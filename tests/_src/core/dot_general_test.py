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


if __name__ == '__main__':
  absltest.main()
