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

from unittest import mock
from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax import numpy as jnp
from qwix._src.core import dot_general
from qwix._src.core import einsum
from qwix._src.core import qarray


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
      dict(
          testcase_name='w8a16_channelwise_fast',
          lhs_how=None,
          rhs_how=qarray.HowToQuantize(
              qtype=jnp.int8, channelwise_axes=(1,), tiled_axes={}
          ),
          expect_fast=True,
      ),
      dict(
          testcase_name='w4a16_channelwise_fast',
          lhs_how=None,
          rhs_how=qarray.HowToQuantize(
              qtype=jnp.int4, channelwise_axes=(1,), tiled_axes={}
          ),
          expect_fast=True,
      ),
      dict(
          testcase_name='w4a16_per_tensor_fast',
          lhs_how=None,
          rhs_how=qarray.HowToQuantize(
              qtype=jnp.int4, channelwise_axes=(), tiled_axes={}
          ),
          expect_fast=True,
      ),
      dict(
          testcase_name='w4a16_subchannel_slow',
          lhs_how=None,
          rhs_how=qarray.HowToQuantize(
              qtype=jnp.int4, channelwise_axes=(1,), tiled_axes={0: 128}
          ),
          expect_fast=False,
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
          expect_fast=False,
      ),
      dict(
          testcase_name='w4a16_contracting_channelwise_slow',
          lhs_how=None,
          rhs_how=qarray.HowToQuantize(
              qtype=jnp.int4, channelwise_axes=(0, 1), tiled_axes={}
          ),
          expect_fast=False,
      ),
      dict(
          testcase_name='nf4a16_slow',
          lhs_how=None,
          rhs_how=qarray.HowToQuantize(
              qtype='nf4', channelwise_axes=(1,), tiled_axes={}
          ),
          expect_fast=False,
      ),
      dict(
          testcase_name='a16_unquantized_slow',
          lhs_how=None,
          rhs_how=None,
          expect_fast=False,
      ),
      dict(
          testcase_name='w8a8_subchannel_fast',
          lhs_how=qarray.HowToQuantize(
              qtype=jnp.int8, channelwise_axes=(0,), tiled_axes={1: 128}
          ),
          rhs_how=qarray.HowToQuantize(
              qtype=jnp.int8, channelwise_axes=(1,), tiled_axes={0: 128}
          ),
          expect_fast=True,
      ),
      dict(
          testcase_name='w8a8_small_tile_slow',
          lhs_how=qarray.HowToQuantize(
              qtype=jnp.int8, channelwise_axes=(0,), tiled_axes={1: 64}
          ),
          rhs_how=qarray.HowToQuantize(
              qtype=jnp.int8, channelwise_axes=(1,), tiled_axes={0: 64}
          ),
          expect_fast=False,
      ),
      dict(
          testcase_name='w8a8_channelwise_fast',
          lhs_how=qarray.HowToQuantize(
              qtype=jnp.int8, channelwise_axes=(0,), tiled_axes={}
          ),
          rhs_how=qarray.HowToQuantize(
              qtype=jnp.int8, channelwise_axes=(1,), tiled_axes={}
          ),
          expect_fast=True,
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
          expect_fast=True,
      ),
      dict(
          testcase_name='raw_int8_x_raw_int8_slow',
          lhs_how=None,
          rhs_how=None,
          lhs_dtype=jnp.int8,
          rhs_dtype=jnp.int8,
          expect_fast=False,
      ),
      dict(
          testcase_name='raw_fp8_x_w4_subchannel_fast',
          lhs_how=None,
          rhs_how=qarray.HowToQuantize(
              qtype=jnp.int4, channelwise_axes=(1,), tiled_axes={0: 128}
          ),
          lhs_dtype=jnp.float8_e4m3fn,
          expect_fast=True,
      ),
  )
  @mock.patch.object(dot_general, '_slow_dot_general', autospec=True)
  @mock.patch.object(dot_general, '_fast_dot_general', autospec=True)
  def test_dot_general_implementation(
      self,
      mock_fast,
      mock_slow,
      *,
      lhs_how: qarray.HowToQuantize | None,
      rhs_how: qarray.HowToQuantize | None,
      expect_fast: bool,
      lhs_dtype: jax.typing.DTypeLike = jnp.bfloat16,
      rhs_dtype: jax.typing.DTypeLike = jnp.bfloat16,
  ):
    mock_fast.return_value = jnp.ones((16, 64), jnp.bfloat16)
    mock_slow.return_value = jnp.ones((16, 64), jnp.bfloat16)

    lhs = jax.random.normal(jax.random.key(0), (16, 256), jnp.bfloat16)
    rhs = jax.random.normal(jax.random.key(1), (256, 64), jnp.bfloat16)
    q_lhs = qarray.quantize(lhs, lhs_how) if lhs_how else lhs.astype(lhs_dtype)
    q_rhs = qarray.quantize(rhs, rhs_how) if rhs_how else rhs.astype(rhs_dtype)

    dot_general.dot_general(q_lhs, q_rhs, (((1,), (0,)), ((), ())))
    if expect_fast:
      mock_fast.assert_called_once()
      mock_slow.assert_not_called()
    else:
      mock_fast.assert_not_called()
      mock_slow.assert_called_once()

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

    def rel_err(x):
      return jnp.mean(jnp.abs(x.astype(jnp.float32) - ref)) / jnp.mean(
          jnp.abs(ref)
      )

    self.assertEqual(out.dtype, slow_out.dtype)
    self.assertEqual(out.shape, slow_out.shape)
    # The fast path applies the scale after accumulation instead of rounding
    # each dequantized weight, so results are not bit-identical to the slow
    # path, but should be at least as accurate.
    fast_err, slow_err = float(rel_err(out)), float(rel_err(slow_out))
    self.assertLess(fast_err, 1e-2)
    self.assertLessEqual(fast_err, slow_err * 1.5 + 1e-6)


if __name__ == '__main__':
  absltest.main()
