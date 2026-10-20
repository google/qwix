# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import functools
from unittest import mock
from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax import numpy as jnp
import numpy as np
from qwix._src.core import dot_general
from qwix._src.core import einsum
from qwix._src.core import qarray

# Unquantized x QArray cases whose scale varies along a contracting axis, so
# that dot_general folds the scale into the unquantized operand.
_FOLD_CASES = {
    # QT backward dlhs = g . W^T with W[K, N] channelwise on N.
    'qt_backward_dlhs': dict(
        lhs_shape=(16, 64),
        rhs_shape=(256, 64),
        dimension_numbers=(((1,), (1,)), ((), ())),
        channelwise_axes=(1,),
    ),
    # QT backward drhs = X^T . g with X[M, K] channelwise on M, on the lhs.
    'qt_backward_drhs': dict(
        lhs_shape=(16, 256),
        rhs_shape=(16, 64),
        dimension_numbers=(((0,), (0,)), ((), ())),
        channelwise_axes=(0,),
        quantize_lhs=True,
    ),
    # Attention P[b,k,g,t,s] . V[b,s,k,d] with per-token, per-head scales.
    'attention_pv': dict(
        lhs_shape=(2, 2, 2, 4, 8),
        rhs_shape=(2, 8, 2, 16),
        dimension_numbers=(((4,), (1,)), ((0, 1), (0, 2))),
        channelwise_axes=(0, 1, 2),
    ),
    # The same with V on the lhs, which pairs its contracting and batch axes
    # with those of P in a different order.
    'attention_pv_v_on_lhs': dict(
        lhs_shape=(2, 8, 2, 16),
        rhs_shape=(2, 2, 2, 4, 8),
        dimension_numbers=(((1,), (4,)), ((0, 2), (0, 1))),
        channelwise_axes=(0, 1, 2),
        quantize_lhs=True,
    ),
    # Tiles along the contracting axis.
    'contracting_tiled': dict(
        lhs_shape=(16, 256),
        rhs_shape=(256, 64),
        dimension_numbers=(((1,), (0,)), ((), ())),
        tiled_axes={0: 64},
    ),
}


def _make_fold_operands(case, float_dtype, qtype, scale_dtype):
  """Returns (lhs, rhs, dimension_numbers) for a case in _FOLD_CASES."""
  case = _FOLD_CASES[case]
  lhs = jax.random.normal(jax.random.key(0), case['lhs_shape'], jnp.float32)
  rhs = jax.random.normal(jax.random.key(1), case['rhs_shape'], jnp.float32)
  how = qarray.HowToQuantize(
      qtype=qtype,
      channelwise_axes=case.get('channelwise_axes', ()),
      tiled_axes=case.get('tiled_axes', {}),
  )
  if case.get('quantize_lhs'):
    lhs = qarray.quantize(lhs.astype(scale_dtype), how)
    rhs = rhs.astype(float_dtype)
  else:
    lhs = lhs.astype(float_dtype)
    rhs = qarray.quantize(rhs.astype(scale_dtype), how)
  return lhs, rhs, case['dimension_numbers']


def _to_f32(x: qarray.MaybeQArray) -> jax.Array:
  """Returns the (dequantized) array in f32 without rounding the scale."""
  if isinstance(x, qarray.QArray):
    return qarray.dequantize(x.astype(jnp.float32))
  return x.astype(jnp.float32)


def _rel_err(x: jax.Array, ref: jax.Array) -> float:
  """Returns the mean absolute error of x relative to the mean |ref|."""
  err = jnp.mean(jnp.abs(x.astype(jnp.float32) - ref)) / jnp.mean(jnp.abs(ref))
  return float(err)


def _ones_fold_operands(float_dtype, scale_dtype, quantize_lhs):
  """Returns all-ones (lhs, rhs) of a [10, 10] matmul that takes the fold path."""
  x = jnp.ones((10, 10), float_dtype)
  # The scale is channelwise along the contracting axis (1 of lhs, 0 of rhs),
  # so it cannot be applied to the output and is folded into x instead.
  q = qarray.QArray(
      jnp.ones((10, 10), jnp.int8),
      jnp.ones((1, 10) if quantize_lhs else (10, 1), scale_dtype),
  )
  return (q, x) if quantize_lhs else (x, q)


class DotGeneralTest(parameterized.TestCase):
  """Small-scale CPU tests for dot_general which doesn't cover numerics."""

  @parameterized.named_parameters(
      dict(
          testcase_name='bf16_f32',
          lhs_dtype=jnp.bfloat16,
          rhs_dtype=jnp.float32,
          expected_output_dtype=jnp.float32,
      ),
      dict(
          testcase_name='bf16_i8bf16',
          lhs_dtype=jnp.bfloat16,
          rhs_dtype=(jnp.int8, jnp.bfloat16),
          expected_output_dtype=jnp.bfloat16,
      ),
      dict(
          testcase_name='bf16_i8f32',
          lhs_dtype=jnp.bfloat16,
          rhs_dtype=(jnp.int8, jnp.float32),
          expected_output_dtype=jnp.float32,
      ),
      dict(
          testcase_name='i4bf16_i8f32',
          lhs_dtype=(jnp.int4, jnp.bfloat16),
          rhs_dtype=(jnp.int8, jnp.float32),
          expected_output_dtype=jnp.float32,
      ),
      dict(
          testcase_name='f8bf16_f8f32',
          lhs_dtype=(jnp.float8_e4m3fn, jnp.bfloat16),
          rhs_dtype=(jnp.float8_e4m3fn, jnp.float32),
          expected_output_dtype=jnp.float32,
      ),
      dict(
          testcase_name='mxfp8_mxfp8',
          lhs_dtype=(jnp.float8_e4m3fn, jnp.bfloat16, 'mxfp8'),
          rhs_dtype=(jnp.float8_e4m3fn, jnp.bfloat16, 'mxfp8'),
          expected_output_dtype=jnp.bfloat16,
      ),
      dict(
          testcase_name='bool_i8bf16',
          lhs_dtype=jnp.bool_,
          rhs_dtype=(jnp.int8, jnp.bfloat16),
          expected_output_dtype=jnp.bfloat16,
      ),
  )
  def test_output_dtype(self, lhs_dtype, rhs_dtype, expected_output_dtype):
    if isinstance(lhs_dtype, tuple):
      kwargs = {}
      if len(lhs_dtype) > 2:
        kwargs['qtype'] = lhs_dtype[2]
      lhs = qarray.QArray(
          jnp.ones((10, 10), lhs_dtype[0]),
          jnp.ones((1, 1), lhs_dtype[1]),
          **kwargs,
      )
    else:
      lhs = jnp.ones((10, 10), lhs_dtype)
    if isinstance(rhs_dtype, tuple):
      kwargs = {}
      if len(rhs_dtype) > 2:
        kwargs['qtype'] = rhs_dtype[2]
      rhs = qarray.QArray(
          jnp.ones((10, 10), rhs_dtype[0]),
          jnp.ones((1, 1), rhs_dtype[1]),
          **kwargs,
      )
    else:
      rhs = jnp.ones((10, 10), rhs_dtype)
    dnums = (([1], [0]), ([], []))

    for preferred_element_type in (None, jnp.bfloat16, jnp.float32):
      if preferred_element_type is not None:
        expected_output_dtype = preferred_element_type

      with self.subTest(f'preferred_element_type={preferred_element_type}'):
        slow_output = jax.eval_shape(
            lambda: dot_general._slow_dot_general(
                lhs,
                rhs,
                dnums,
                preferred_element_type=preferred_element_type,  # pylint: disable=cell-var-from-loop
            )
        )
        fast_output = jax.eval_shape(
            lambda: dot_general._fast_dot_general(
                lhs,
                rhs,
                dnums,
                preferred_element_type=preferred_element_type,  # pylint: disable=cell-var-from-loop
            )
        )
        loop_output = jax.eval_shape(
            lambda: dot_general.loop_dot_general(
                lhs,
                rhs,
                dnums,
                preferred_element_type=preferred_element_type,  # pylint: disable=cell-var-from-loop
            )
        )
        dot_general_output = jax.eval_shape(
            lambda: dot_general.dot_general(
                lhs,
                rhs,
                dnums,
                preferred_element_type=preferred_element_type,  # pylint: disable=cell-var-from-loop
            )
        )
        einsum_output = jax.eval_shape(
            lambda: einsum.einsum(
                'ab,bc->ac',
                lhs,
                rhs,
                preferred_element_type=preferred_element_type,  # pylint: disable=cell-var-from-loop
            )
        )
        self.assertEqual(slow_output.dtype, expected_output_dtype)
        self.assertEqual(fast_output.dtype, expected_output_dtype)
        self.assertEqual(loop_output.dtype, expected_output_dtype)
        self.assertEqual(einsum_output.dtype, expected_output_dtype)
        self.assertEqual(dot_general_output.dtype, expected_output_dtype)

  @mock.patch.object(jax.nn, 'scaled_matmul')
  def test_outer_product(self, mock_scaled_matmul):
    mock_scaled_matmul.return_value = jnp.ones((1, 10, 10), jnp.float32)

    lhs = qarray.QArray(
        jnp.ones((10, 1), jnp.float8_e4m3fn),
        jnp.ones((10, 1), jnp.bfloat16),
        qtype='mxfp8',
    )
    rhs = qarray.QArray(
        jnp.ones((1, 10), jnp.float8_e4m3fn),
        jnp.ones((1, 10), jnp.bfloat16),
        qtype='mxfp8',
    )
    dnums = (((), ()), ((), ()))
    res = dot_general.dot_general(lhs, rhs, dnums)
    self.assertEqual(res.shape, (10, 1, 1, 10))

    mock_scaled_matmul.assert_called_once()

    args, _ = mock_scaled_matmul.call_args
    lhs_3d, rhs_3d, lhs_scale_3d, rhs_scale_3d = args
    self.assertEqual(lhs_3d.shape, (1, 10, 1))
    self.assertEqual(rhs_3d.shape, (1, 10, 1))
    self.assertEqual(lhs_scale_3d.shape, (1, 10, 1))
    self.assertEqual(rhs_scale_3d.shape, (1, 10, 1))

  def test_innermost_tiling_heuristic(self):
    """Verifies that multi-dimensional dot_general picks the innermost contracting reduction axis."""
    dnums = (((2, 3), (0, 1)), ((), ()))

    how_lhs = dot_general.get_how_to_quantize(
        dimension_numbers=dnums,
        ndims=(4, 3),
        for_lhs=True,
        qtype='mxfp8',
        tile_size=32,
    )
    # Contracting axes for LHS are 2 and 3. Axis 3 should be selected.
    self.assertEqual(how_lhs.tiled_axes, {3: 32})

    how_rhs = dot_general.get_how_to_quantize(
        dimension_numbers=dnums,
        ndims=(4, 3),
        for_lhs=False,
        qtype='mxfp8',
        tile_size=32,
    )
    # Contracting axes for RHS are 0 and 1. Axis 1 should be selected.
    self.assertEqual(how_rhs.tiled_axes, {1: 32})

  @parameterized.named_parameters(
      dict(testcase_name='cpu_default', device_kind='cpu', expected=128),
      dict(
          testcase_name='gpu_default', device_kind='NVIDIA H100', expected=128
      ),
      dict(testcase_name='tpu_v5e', device_kind='TPU v5 lite', expected=128),
      dict(testcase_name='tpu_v5p', device_kind='TPU v5p', expected=128),
      dict(
          testcase_name='tpu_v6_lite', device_kind='TPU v6 lite', expected=256
      ),
      dict(testcase_name='tpu_v6e', device_kind='TPU v6e', expected=256),
      dict(testcase_name='tpu7x', device_kind='TPU7x', expected=512),
      dict(
          testcase_name='tpu7x_sim',
          device_kind='TPU7x\nsimdevice',
          expected=512,
      ),
      dict(testcase_name='tpu8i', device_kind='TPU8i', expected=256),
      dict(testcase_name='tpu8t', device_kind='TPU8t', expected=256),
  )
  def test_get_min_tile_size_to_dequant_on_output(
      self, device_kind: str, expected: int
  ):
    with mock.patch.object(
        dot_general, '_get_device_kind', return_value=device_kind
    ):
      self.assertEqual(
          dot_general.get_min_tile_size_to_dequant_on_output(), expected
      )

  def test_min_tile_size_uses_abstract_mesh_device(self):
    fake_mesh = mock.MagicMock()
    fake_mesh.abstract_device.device_kind = 'TPU8i'
    with mock.patch.object(
        jax.sharding, 'get_abstract_mesh', return_value=fake_mesh
    ):
      self.assertEqual(
          dot_general.get_min_tile_size_to_dequant_on_output(), 256
      )

  def test_get_device_kind_fallback_without_abstract_device(self):
    fake_mesh = mock.MagicMock()
    fake_mesh.abstract_device = None
    fake_device = mock.MagicMock()
    fake_device.device_kind = 'TPU v6e'
    with (
        mock.patch.object(
            jax.sharding, 'get_abstract_mesh', return_value=fake_mesh
        ),
        mock.patch.object(jax, 'devices', return_value=[fake_device]),
    ):
      self.assertEqual(
          dot_general.get_min_tile_size_to_dequant_on_output(), 256
      )

  @parameterized.named_parameters(
      dict(
          testcase_name='v5p_tile128_fast',
          device_kind='TPU v5p',
          tile_size=128,
          expect_fast=True,
      ),
      dict(
          testcase_name='v6e_tile128_slow',
          device_kind='TPU v6 lite',
          tile_size=128,
          expect_fast=False,
      ),
      dict(
          testcase_name='v6e_tile256_fast',
          device_kind='TPU v6 lite',
          tile_size=256,
          expect_fast=True,
      ),
      dict(
          testcase_name='tpu7x_tile128_slow',
          device_kind='TPU7x',
          tile_size=128,
          expect_fast=False,
      ),
      dict(
          testcase_name='tpu7x_tile256_slow',
          device_kind='TPU7x',
          tile_size=256,
          expect_fast=False,
      ),
      dict(
          testcase_name='tpu7x_tile512_fast',
          device_kind='TPU7x',
          tile_size=512,
          expect_fast=True,
      ),
      dict(
          testcase_name='tpu8i_tile128_slow',
          device_kind='TPU8i',
          tile_size=128,
          expect_fast=False,
      ),
      dict(
          testcase_name='tpu8i_tile256_fast',
          device_kind='TPU8i',
          tile_size=256,
          expect_fast=True,
      ),
      dict(
          testcase_name='tpu8i_tile512_fast',
          device_kind='TPU8i',
          tile_size=512,
          expect_fast=True,
      ),
      dict(
          testcase_name='tpu7x_fp8_tile128_slow',
          device_kind='TPU7x',
          lhs_qtype=jnp.float8_e4m3fn,
          rhs_qtype=jnp.float8_e4m3fn,
          tile_size=128,
          expect_fast=False,
      ),
      dict(
          testcase_name='tpu7x_fp8_tile256_slow',
          device_kind='TPU7x',
          lhs_qtype=jnp.float8_e4m3fn,
          rhs_qtype=jnp.float8_e4m3fn,
          tile_size=256,
          expect_fast=False,
      ),
      dict(
          testcase_name='tpu7x_fp8_tile64_slow',
          device_kind='TPU7x',
          lhs_qtype=jnp.float8_e4m3fn,
          rhs_qtype=jnp.float8_e4m3fn,
          tile_size=64,
          expect_fast=False,
      ),
  )
  def test_dot_general_tile_size_dispatch(
      self,
      device_kind: str,
      tile_size: int,
      expect_fast: bool,
      lhs_qtype: jax.typing.DTypeLike = jnp.int8,
      rhs_qtype: jax.typing.DTypeLike = jnp.int4,
  ):
    k = 1024
    num_tiles = k // tile_size
    lhs = qarray.QArray(
        jnp.ones((16, k), lhs_qtype),
        jnp.ones((16, num_tiles), jnp.bfloat16),
        qtype=lhs_qtype,
    )
    rhs = qarray.QArray(
        jnp.ones((k, 16), rhs_qtype),
        jnp.ones((num_tiles, 16), jnp.bfloat16),
        qtype=rhs_qtype,
    )
    dnums = (([1], [0]), ([], []))
    with (
        mock.patch.object(
            dot_general, '_get_device_kind', return_value=device_kind
        ),
        mock.patch.object(
            dot_general,
            '_fast_dot_general',
            wraps=dot_general._fast_dot_general,
        ) as mock_fast,
        mock.patch.object(
            dot_general,
            '_slow_dot_general',
            wraps=dot_general._slow_dot_general,
        ) as mock_slow,
    ):
      res = dot_general.dot_general(lhs, rhs, dnums)
      self.assertEqual(res.shape, (16, 16))
      if expect_fast:
        mock_fast.assert_called_once()
        mock_slow.assert_not_called()
      else:
        mock_slow.assert_called_once()
        mock_fast.assert_not_called()

  @parameterized.named_parameters(
      dict(
          testcase_name='w8a16_channelwise_fast',
          lhs_how=None,
          rhs_how=qarray.HowToQuantize(
              qtype=jnp.int8, channelwise_axes=(1,), tiled_axes={}
          ),
          expect='fast',
      ),
      dict(
          testcase_name='w4a16_channelwise_fast',
          lhs_how=None,
          rhs_how=qarray.HowToQuantize(
              qtype=jnp.int4, channelwise_axes=(1,), tiled_axes={}
          ),
          expect='fast',
      ),
      dict(
          testcase_name='w4a16_per_tensor_fast',
          lhs_how=None,
          rhs_how=qarray.HowToQuantize(
              qtype=jnp.int4, channelwise_axes=(), tiled_axes={}
          ),
          expect='fast',
      ),
      dict(
          testcase_name='w4a16_subchannel_slow',
          lhs_how=None,
          rhs_how=qarray.HowToQuantize(
              qtype=jnp.int4, channelwise_axes=(1,), tiled_axes={0: 128}
          ),
          expect='slow',
      ),
      dict(
          testcase_name='w8a16_zero_point_slow',
          lhs_how=None,
          rhs_how=qarray.HowToQuantize(
              qtype=jnp.int8,
              channelwise_axes=(1,),
              tiled_axes={},
              calibration_method='minmax',
          ),
          expect='slow',
      ),
      dict(
          # The scale varies along both the contracting and the remaining axis,
          # so it can neither be applied to the output nor folded into lhs.
          testcase_name='w4a16_channelwise_both_axes_slow',
          lhs_how=None,
          rhs_how=qarray.HowToQuantize(
              qtype=jnp.int4, channelwise_axes=(0, 1), tiled_axes={}
          ),
          expect='slow',
      ),
      dict(
          testcase_name='nf4a16_slow',
          lhs_how=None,
          rhs_how=qarray.HowToQuantize(
              qtype='nf4', channelwise_axes=(1,), tiled_axes={}
          ),
          expect='slow',
      ),
      dict(
          testcase_name='a16_unquantized_slow',
          lhs_how=None,
          rhs_how=None,
          expect='slow',
      ),
      dict(
          testcase_name='w8a8_subchannel_fast',
          lhs_how=qarray.HowToQuantize(
              qtype=jnp.int8, channelwise_axes=(0,), tiled_axes={1: 128}
          ),
          rhs_how=qarray.HowToQuantize(
              qtype=jnp.int8, channelwise_axes=(1,), tiled_axes={0: 128}
          ),
          expect='fast',
      ),
      dict(
          testcase_name='w8a8_small_tile_slow',
          lhs_how=qarray.HowToQuantize(
              qtype=jnp.int8, channelwise_axes=(0,), tiled_axes={1: 64}
          ),
          rhs_how=qarray.HowToQuantize(
              qtype=jnp.int8, channelwise_axes=(1,), tiled_axes={0: 64}
          ),
          expect='slow',
      ),
      dict(
          testcase_name='w8a8_channelwise_fast',
          lhs_how=qarray.HowToQuantize(
              qtype=jnp.int8, channelwise_axes=(0,), tiled_axes={}
          ),
          rhs_how=qarray.HowToQuantize(
              qtype=jnp.int8, channelwise_axes=(1,), tiled_axes={}
          ),
          expect='fast',
      ),
      dict(
          testcase_name='w8a8_act_zero_point_fast',
          lhs_how=qarray.HowToQuantize(
              qtype=jnp.int8,
              channelwise_axes=(0,),
              tiled_axes={},
              calibration_method='minmax',
          ),
          rhs_how=qarray.HowToQuantize(
              qtype=jnp.int8, channelwise_axes=(1,), tiled_axes={}
          ),
          expect='fast',
      ),
      dict(
          testcase_name='raw_int8_x_raw_int8_slow',
          lhs_how=None,
          rhs_how=None,
          lhs_dtype=jnp.int8,
          rhs_dtype=jnp.int8,
          expect='slow',
      ),
      dict(
          testcase_name='raw_fp8_x_w4_subchannel_fast',
          lhs_how=None,
          rhs_how=qarray.HowToQuantize(
              qtype=jnp.int4, channelwise_axes=(1,), tiled_axes={0: 128}
          ),
          lhs_dtype=jnp.float8_e4m3fn,
          expect='fast',
      ),
      # Unquantized x QArray whose scale varies along a contracting axis but
      # not along its remaining axes: fold the scale into the unquantized
      # operand.
      dict(
          testcase_name='w8a16_contracting_channelwise_fold',
          lhs_how=None,
          rhs_how=qarray.HowToQuantize(
              qtype=jnp.int8, channelwise_axes=(0,), tiled_axes={}
          ),
          expect='fold',
      ),
      dict(
          testcase_name='w4a16_contracting_channelwise_fold',
          lhs_how=None,
          rhs_how=qarray.HowToQuantize(
              qtype=jnp.int4, channelwise_axes=(0,), tiled_axes={}
          ),
          expect='fold',
      ),
      dict(
          testcase_name='fp8a16_contracting_channelwise_fold',
          lhs_how=None,
          rhs_how=qarray.HowToQuantize(
              qtype=jnp.float8_e4m3fn, channelwise_axes=(0,), tiled_axes={}
          ),
          expect='fold',
      ),
      dict(
          # Tiles along the contracting axis are broadcast to lhs tile by tile.
          testcase_name='w8a16_contracting_tiled_fold',
          lhs_how=None,
          rhs_how=qarray.HowToQuantize(
              qtype=jnp.int8, channelwise_axes=(), tiled_axes={0: 128}
          ),
          expect='fold',
      ),
      dict(
          testcase_name='w8a16_contracting_channelwise_zero_point_slow',
          lhs_how=None,
          rhs_how=qarray.HowToQuantize(
              qtype=jnp.int8,
              channelwise_axes=(0,),
              tiled_axes={},
              calibration_method='minmax',
          ),
          expect='slow',
      ),
      dict(
          testcase_name='nf4a16_contracting_channelwise_slow',
          lhs_how=None,
          rhs_how=qarray.HowToQuantize(
              qtype='nf4', channelwise_axes=(0,), tiled_axes={}
          ),
          expect='slow',
      ),
      dict(
          # Two QArrays, no unquantized operand to fold the scale into.
          testcase_name='w8a8_contracting_channelwise_slow',
          lhs_how=qarray.HowToQuantize(
              qtype=jnp.int8, channelwise_axes=(0,), tiled_axes={}
          ),
          rhs_how=qarray.HowToQuantize(
              qtype=jnp.int8, channelwise_axes=(0,), tiled_axes={}
          ),
          expect='slow',
      ),
      dict(
          testcase_name='raw_fp8_x_w8_contracting_channelwise_slow',
          lhs_how=None,
          rhs_how=qarray.HowToQuantize(
              qtype=jnp.int8, channelwise_axes=(0,), tiled_axes={}
          ),
          lhs_dtype=jnp.float8_e4m3fn,
          expect='slow',
      ),
      dict(
          testcase_name='a8w16_contracting_channelwise_fold',
          lhs_how=qarray.HowToQuantize(
              qtype=jnp.int8, channelwise_axes=(1,), tiled_axes={}
          ),
          rhs_how=None,
          expect='fold',
      ),
      dict(
          # Attention Q[b,t,k,g,d] . K[b,s,k,d] with per-token, per-head scales
          # [b,s,k,1]: the scale is constant along the contracting axis d. For
          # P . V, where it varies along the contracting axis s, see
          # _FOLD_CASES['attention_pv'].
          testcase_name='attention_qk_per_token_per_head_fast',
          lhs_shape=(2, 4, 2, 2, 16),
          rhs_shape=(2, 8, 2, 16),
          dimension_numbers=(((4,), (3,)), ((0, 2), (0, 2))),
          lhs_how=None,
          rhs_how=qarray.HowToQuantize(
              qtype=jnp.int8, channelwise_axes=(0, 1, 2), tiled_axes={}
          ),
          expect='fast',
      ),
      dict(
          # Per-token scales [b,s,1,1] shared by all heads.
          testcase_name='attention_pv_per_token_fold',
          lhs_shape=(2, 2, 2, 4, 8),
          rhs_shape=(2, 8, 2, 16),
          dimension_numbers=(((4,), (1,)), ((0, 1), (0, 2))),
          lhs_how=None,
          rhs_how=qarray.HowToQuantize(
              qtype=jnp.int4, channelwise_axes=(0, 1), tiled_axes={}
          ),
          expect='fold',
      ),
      dict(
          # The scale also varies along the remaining axis d.
          testcase_name='attention_pv_tiled_head_dim_slow',
          lhs_shape=(2, 2, 2, 4, 8),
          rhs_shape=(2, 8, 2, 16),
          dimension_numbers=(((4,), (1,)), ((0, 1), (0, 2))),
          lhs_how=None,
          rhs_how=qarray.HowToQuantize(
              qtype=jnp.int8, channelwise_axes=(0, 1, 2), tiled_axes={3: 8}
          ),
          expect='slow',
      ),
  )
  @mock.patch.object(dot_general, '_slow_dot_general', autospec=True)
  @mock.patch.object(dot_general, '_fold_scale_dot_general', autospec=True)
  @mock.patch.object(dot_general, '_fast_dot_general', autospec=True)
  def test_dot_general_implementation(
      self,
      mock_fast,
      mock_fold,
      mock_slow,
      *,
      lhs_how: qarray.HowToQuantize | None,
      rhs_how: qarray.HowToQuantize | None,
      expect: str,
      lhs_shape: tuple[int, ...] = (16, 256),
      rhs_shape: tuple[int, ...] = (256, 64),
      dimension_numbers: jax.lax.DotDimensionNumbers = (((1,), (0,)), ((), ())),
      lhs_dtype: jax.typing.DTypeLike = jnp.bfloat16,
      rhs_dtype: jax.typing.DTypeLike = jnp.bfloat16,
  ):
    lhs = jax.random.normal(jax.random.key(0), lhs_shape, jnp.bfloat16)
    rhs = jax.random.normal(jax.random.key(1), rhs_shape, jnp.bfloat16)
    q_lhs = qarray.quantize(lhs, lhs_how) if lhs_how else lhs.astype(lhs_dtype)
    q_rhs = qarray.quantize(rhs, rhs_how) if rhs_how else rhs.astype(rhs_dtype)

    dot_general.dot_general(q_lhs, q_rhs, dimension_numbers)
    mocks = {'fast': mock_fast, 'fold': mock_fold, 'slow': mock_slow}
    for name, mock_impl in mocks.items():
      if name == expect:
        mock_impl.assert_called_once()
      else:
        mock_impl.assert_not_called()

  @parameterized.product(
      lhs_dtype=(jnp.bfloat16, jnp.float32),
      rhs_qtype=(jnp.int8, jnp.int4),
      rhs_scale_dtype=(jnp.bfloat16, jnp.float32),
  )
  def test_unquantized_x_channelwise_fast_matches_slow(
      self, lhs_dtype, rhs_qtype, rhs_scale_dtype
  ):
    lhs = jax.random.normal(jax.random.key(0), (16, 256), lhs_dtype)
    rhs = jax.random.normal(jax.random.key(1), (256, 64), rhs_scale_dtype)
    q_rhs = qarray.quantize(
        rhs,
        qarray.HowToQuantize(
            qtype=rhs_qtype, channelwise_axes=(1,), tiled_axes={}
        ),
    )
    dimension_numbers = (((1,), (0,)), ((), ()))

    out = jax.jit(dot_general.dot_general, static_argnums=2)(
        lhs, q_rhs, dimension_numbers
    )
    slow_out = jax.jit(dot_general._slow_dot_general, static_argnums=2)(
        lhs, q_rhs, dimension_numbers
    )
    # Exact reference: f32 matmul on the f32-dequantized weight.
    ref = jnp.dot(
        lhs.astype(jnp.float32),
        q_rhs.qvalue.astype(jnp.float32) * q_rhs.scale.astype(jnp.float32),
    )

    self.assertEqual(out.dtype, slow_out.dtype)
    self.assertEqual(out.shape, slow_out.shape)
    # The fast path applies the scale after accumulation instead of rounding
    # each dequantized weight, so results are not bit-identical to the slow
    # path, but should be at least as accurate.
    fast_err, slow_err = _rel_err(out, ref), _rel_err(slow_out, ref)
    self.assertLess(fast_err, 1e-2)
    self.assertLessEqual(fast_err, slow_err * 1.5 + 1e-6)

  @parameterized.product(
      case=tuple(_FOLD_CASES),
      float_dtype=(jnp.bfloat16, jnp.float32),
      qtype=(jnp.int8, jnp.int4, jnp.float8_e4m3fn),
      scale_dtype=(jnp.bfloat16, jnp.float32),
  )
  def test_unquantized_x_contracting_channelwise_fold_matches_slow(
      self, case, float_dtype, qtype, scale_dtype
  ):
    lhs, rhs, dimension_numbers = _make_fold_operands(
        case, float_dtype, qtype, scale_dtype
    )

    with mock.patch.object(
        dot_general,
        '_fold_scale_dot_general',
        wraps=dot_general._fold_scale_dot_general,
    ) as mock_fold:
      out = jax.jit(dot_general.dot_general, static_argnums=2)(
          lhs, rhs, dimension_numbers
      )
    mock_fold.assert_called_once()
    slow_out = jax.jit(dot_general._slow_dot_general, static_argnums=2)(
        lhs, rhs, dimension_numbers
    )
    # Exact reference: f32 dot_general on the f32-dequantized operand.
    ref = jax.lax.dot_general(
        _to_f32(lhs),
        _to_f32(rhs),
        dimension_numbers,
        precision=jax.lax.Precision.HIGHEST,
    )

    self.assertEqual(out.dtype, slow_out.dtype)
    self.assertEqual(out.shape, slow_out.shape)
    # The fold path rounds the scaled unquantized operand instead of each
    # dequantized value, so results are not bit-identical to the slow path,
    # but should be about as accurate. Observed on CPU: with a bf16 scale,
    # fold_err / slow_err <= 1.17 (int4, tiled contracting axis); with an f32
    # scale, the slow path reproduces the reference exactly (slow_err == 0)
    # and fold_err <= 1.4e-7, about one ulp of f32, so a few ulps of headroom
    # bound those cases.
    fold_err, slow_err = _rel_err(out, ref), _rel_err(slow_out, ref)
    self.assertLess(fold_err, 1e-2)
    self.assertLessEqual(
        fold_err, slow_err * 1.5 + 4 * jnp.finfo(jnp.float32).eps
    )

  @parameterized.product(
      float_dtype=(jnp.bfloat16, jnp.float32),
      scale_dtype=(jnp.bfloat16, jnp.float32),
      preferred_element_type=(None, jnp.bfloat16, jnp.float32),
      quantize_lhs=(False, True),
  )
  def test_fold_output_dtype_matches_slow(
      self, float_dtype, scale_dtype, preferred_element_type, quantize_lhs
  ):
    lhs, rhs = _ones_fold_operands(float_dtype, scale_dtype, quantize_lhs)
    dnums = (([1], [0]), ([], []))

    def output_of(fn):
      return jax.eval_shape(
          lambda: fn(
              lhs, rhs, dnums, preferred_element_type=preferred_element_type
          )
      )

    expected = output_of(dot_general._slow_dot_general)
    for fn in (dot_general._fold_scale_dot_general, dot_general.dot_general):
      actual = output_of(fn)
      self.assertEqual(actual.dtype, expected.dtype)
      self.assertEqual(actual.shape, expected.shape)

  def test_fold_scale_into(self):
    # x P[b,k,t,s] . q V[b,s,k,d] with scale [b,s,k,1]: contracting s, batch
    # (b,k), so axes (3, 0, 1) of x match axes (1, 0, 2) of q.
    x_axes, scale_axes = (3, 0, 1), (1, 0, 2)
    x = jax.random.normal(jax.random.key(0), (2, 3, 4, 8))
    scale = jax.random.uniform(jax.random.key(1), (2, 8, 3, 1))
    res = dot_general.fold_scale_into(x, scale, x_axes, scale_axes)
    expected = x * jnp.einsum('bskd->bkds', scale)
    np.testing.assert_allclose(res, expected, rtol=1e-6)

    # Two tiles of 4 along the contracting axis s are broadcast tile by tile.
    scale = jax.random.uniform(jax.random.key(2), (1, 2, 1, 1))
    res = dot_general.fold_scale_into(x, scale, x_axes, scale_axes)
    expected = x * jnp.repeat(scale.reshape(2), 4)
    np.testing.assert_allclose(res, expected, rtol=1e-6)

    # q on the lhs: q X[m,k] . x G[m,n] with scale [m,1], contracting m.
    x = jax.random.normal(jax.random.key(3), (8, 5))
    scale = jax.random.uniform(jax.random.key(4), (8, 1))
    res = dot_general.fold_scale_into(x, scale, (0,), (0,))
    np.testing.assert_allclose(res, x * scale, rtol=1e-6)

    # A scale along the remaining axis d of q cannot be folded into x.
    x = jax.random.normal(jax.random.key(0), (2, 3, 4, 8))
    with self.assertRaisesRegex(ValueError, 'Cannot transpose'):
      dot_general.fold_scale_into(x, jnp.ones((2, 8, 3, 2)), x_axes, scale_axes)

  @parameterized.product(
      float_dtype=(jnp.bfloat16, jnp.float32),
      scale_dtype=(jnp.bfloat16, jnp.float32),
      quantize_lhs=(False, True),
  )
  def test_fold_matmul_operand_dtypes(
      self, float_dtype, scale_dtype, quantize_lhs
  ):
    lhs, rhs = _ones_fold_operands(float_dtype, scale_dtype, quantize_lhs)
    jaxpr = jax.make_jaxpr(
        functools.partial(
            dot_general.dot_general, dimension_numbers=(([1], [0]), ([], []))
        )
    )(lhs, rhs)
    dots = [e for e in jaxpr.eqns if e.primitive.name == 'dot_general']
    self.assertLen(dots, 1)
    # The matmul consumes the raw quantized values and x scaled in the
    # promoted dtype of x and the scale.
    scaled_dtype = jnp.promote_types(float_dtype, scale_dtype)
    expected = (
        [jnp.int8, scaled_dtype] if quantize_lhs else [scaled_dtype, jnp.int8]
    )
    self.assertEqual([v.aval.dtype for v in dots[0].invars], expected)

  @parameterized.named_parameters(
      ('f16_x_f16_scale', jnp.float16, jnp.float16, 'slow'),
      ('f16_x_f32_scale', jnp.float16, jnp.float32, 'fold'),
      ('f32_x_f16_scale', jnp.float32, jnp.float16, 'fold'),
      # A weakly typed f32 scale is multiplied into an f16 x in f16.
      ('f16_x_weak_f32_scale', jnp.float16, None, 'slow'),
  )
  def test_fold_dispatch_by_scaled_dtype(
      self, float_dtype, scale_dtype, expect
  ):
    # Channelwise along the contracting axis 0 of rhs.
    lhs = jnp.ones((16, 256), float_dtype)
    rhs = qarray.QArray(
        jnp.ones((256, 64), jnp.int8), jnp.full((256, 1), 1e-3, scale_dtype)
    )
    dnums = (([1], [0]), ([], []))
    impls = {
        'fold': dot_general._fold_scale_dot_general,
        'slow': dot_general._slow_dot_general,
    }
    self.assertIs(
        dot_general._choose_dot_general_impl(lhs, rhs, dnums), impls[expect]
    )

  def test_f16_fold_underflows_in_attention_pv(self):
    # Why _choose_dot_general_impl keeps an f16 x with an f16 scale on the slow
    # path: in attention P[b,k,t,s] . V[b,s,k,d] with per-token, per-head
    # scales, P ~ 1/s times the value scales ~1e-3 is in the f16 subnormal
    # range, so folding the scale into P loses most of its precision, while the
    # dequantized values the slow path rounds stay in range.
    b, k, t, s, d = 1, 2, 4, 1024, 64
    logits = jax.random.normal(jax.random.key(0), (b, k, t, s))
    p = jax.nn.softmax(logits, axis=-1).astype(jnp.float16)
    v = 0.1 * jax.random.normal(jax.random.key(1), (b, s, k, d))
    # Calibrate in f32 and only store the scale in f16.
    v = qarray.quantize(
        v,
        qarray.HowToQuantize(
            qtype=jnp.int8, channelwise_axes=(0, 1, 2), tiled_axes={}
        ),
    ).astype(jnp.float16)
    dnums = (((3,), (1,)), ((0, 1), (0, 2)))
    fold_out = jax.jit(dot_general._fold_scale_dot_general, static_argnums=2)(
        p, v, dnums
    )
    slow_out = jax.jit(dot_general._slow_dot_general, static_argnums=2)(
        p, v, dnums
    )
    ref = jax.lax.dot_general(
        _to_f32(p), _to_f32(v), dnums, precision=jax.lax.Precision.HIGHEST
    )
    fold_err, slow_err = _rel_err(fold_out, ref), _rel_err(slow_out, ref)
    # Observed on CPU: slow_err 2.8e-4 to 4.1e-4, fold_err 5.4e-3 (13-19x).
    self.assertLess(slow_err, 1e-3)
    self.assertGreater(fold_err, 5 * slow_err)

  def test_fold_grad_matches_slow(self):
    lhs, rhs, dimension_numbers = _make_fold_operands(
        'attention_pv', jnp.float32, jnp.int8, jnp.float32
    )

    def loss(fn, x):
      return jnp.sum(fn(x, rhs, dimension_numbers) ** 2)

    with mock.patch.object(
        dot_general,
        '_fold_scale_dot_general',
        wraps=dot_general._fold_scale_dot_general,
    ) as mock_fold:
      grad = jax.grad(functools.partial(loss, dot_general.dot_general))(lhs)
    mock_fold.assert_called_once()
    expected = jax.grad(functools.partial(loss, dot_general._slow_dot_general))(
        lhs
    )
    np.testing.assert_allclose(grad, expected, rtol=1e-5, atol=1e-5)


if __name__ == '__main__':
  absltest.main()
