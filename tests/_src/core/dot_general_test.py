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
from absl import logging
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
          testcase_name='e4m3_int4_sc32',
          fp8_dtype=jnp.float8_e4m3fn,
          w4_dtype=jnp.int4,
          w4_qtype=jnp.int4,
          fp8_tile_size=None,
          swap_operands=False,
      ),
      dict(
          testcase_name='e5m2_uint4_sc32',
          fp8_dtype=jnp.float8_e5m2,
          w4_dtype=jnp.uint4,
          w4_qtype=jnp.uint4,
          fp8_tile_size=128,
          swap_operands=False,
      ),
      dict(
          testcase_name='e4m3_fp4_mxfp4',
          fp8_dtype=jnp.float8_e4m3fn,
          w4_dtype=jnp.float4_e2m1fn,
          w4_qtype='mxfp4',
          fp8_tile_size=256,
          swap_operands=False,
      ),
      dict(
          testcase_name='e4m3_nf4',
          fp8_dtype=jnp.float8_e4m3fn,
          w4_dtype=jnp.uint4,
          w4_qtype='nf4',
          fp8_tile_size=None,
          swap_operands=False,
      ),
      dict(
          testcase_name='int4_sc32_e4m3_swapped',
          fp8_dtype=jnp.float8_e4m3fn,
          w4_dtype=jnp.int4,
          w4_qtype=jnp.int4,
          fp8_tile_size=128,
          swap_operands=True,
      ),
      dict(
          testcase_name='raw_e4m3_int4_sc32',
          fp8_dtype=jnp.float8_e4m3fn,
          w4_dtype=jnp.int4,
          w4_qtype=jnp.int4,
          fp8_tile_size=None,
          swap_operands=False,
          is_fp8_raw_array=True,
      ),
      dict(
          testcase_name='int4_sc32_raw_e4m3_swapped',
          fp8_dtype=jnp.float8_e4m3fn,
          w4_dtype=jnp.int4,
          w4_qtype=jnp.int4,
          fp8_tile_size=None,
          swap_operands=True,
          is_fp8_raw_array=True,
      ),
  )
  def test_w4a8_fine_subchannel_dequant_to_fp8(
      self,
      fp8_dtype,
      w4_dtype,
      w4_qtype,
      fp8_tile_size,
      swap_operands,
      is_fp8_raw_array=False,
  ):
    k = 256
    if is_fp8_raw_array:
      fp8_op = jnp.ones((16, k), fp8_dtype)
    else:
      fp8_k_scales = 1 if fp8_tile_size is None else k // fp8_tile_size
      fp8_op = qarray.QArray(
          jnp.ones((16, k), fp8_dtype),
          jnp.ones((16, fp8_k_scales), jnp.bfloat16),
          qtype=fp8_dtype,
      )
    # tile_size = 32 < MIN_TILE_SIZE_TO_DEQUANT_ON_OUTPUT (128).
    w4_op = qarray.QArray(
        jnp.ones((k, 32), w4_dtype),
        jnp.ones((k // 32, 32), jnp.float32),
        qtype=w4_qtype,
    )
    if swap_operands:
      lhs = w4_op.T
      rhs = fp8_op.T
    else:
      lhs = fp8_op
      rhs = w4_op
    dnums = (([1], [0]), ([], []))

    with (
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
      out = dot_general.dot_general(lhs, rhs, dnums)
      mock_fast.assert_called_once()
      mock_slow.assert_not_called()
      fast_lhs, fast_rhs = mock_fast.call_args.args[:2]
      if swap_operands:
        self.assertIsInstance(fast_lhs, jax.Array)
        self.assertEqual(fast_lhs.dtype, fp8_dtype)
        if is_fp8_raw_array:
          self.assertIsInstance(fast_rhs, jax.Array)
        else:
          self.assertIsInstance(fast_rhs, qarray.QArray)
        self.assertEqual(out.shape, (32, 16))
      else:
        if is_fp8_raw_array:
          self.assertIsInstance(fast_lhs, jax.Array)
        else:
          self.assertIsInstance(fast_lhs, qarray.QArray)
        self.assertIsInstance(fast_rhs, jax.Array)
        self.assertEqual(fast_rhs.dtype, fp8_dtype)
        self.assertEqual(out.shape, (16, 32))
      # Preserve float32 promotion from w4_op.scale.dtype (float32).
      self.assertEqual(out.dtype, jnp.float32)

    for preferred_element_type in (None, jnp.bfloat16, jnp.float32):
      expected_dtype = preferred_element_type or jnp.float32
      dg_out = jax.eval_shape(
          lambda pet=preferred_element_type: dot_general.dot_general(
              lhs, rhs, dnums, preferred_element_type=pet
          )
      )
      loop_out = jax.eval_shape(
          lambda pet=preferred_element_type: dot_general.loop_dot_general(
              lhs, rhs, dnums, preferred_element_type=pet
          )
      )
      self.assertEqual(dg_out.dtype, expected_dtype)
      self.assertEqual(loop_out.dtype, expected_dtype)

  @parameterized.named_parameters(
      dict(
          testcase_name='int8_fine_sc_fp8',
          case='int8_fine_sc_fp8',
      ),
      dict(
          testcase_name='bf16_acts_fine_int4',
          case='bf16_acts_fine_int4',
      ),
      dict(
          testcase_name='asymmetric_fp8_fine_int4',
          case='asymmetric_fp8_fine_int4',
      ),
      dict(
          testcase_name='fine_int4_bf16_acts',
          case='fine_int4_bf16_acts',
      ),
      dict(
          testcase_name='fine_int4_asymmetric_fp8',
          case='fine_int4_asymmetric_fp8',
      ),
  )
  def test_negative_cases_fallback_to_slow(self, case):
    if case == 'int8_fine_sc_fp8':
      lhs = qarray.QArray(
          jnp.ones((16, 256), jnp.int8),
          jnp.ones((16, 8), jnp.float32),
      )
      rhs = qarray.QArray(
          jnp.ones((256, 32), jnp.float8_e4m3fn),
          jnp.ones((1, 32), jnp.bfloat16),
      )
    elif case == 'bf16_acts_fine_int4':
      lhs = jnp.ones((16, 256), jnp.bfloat16)
      rhs = qarray.QArray(
          jnp.ones((256, 32), jnp.int4),
          jnp.ones((8, 32), jnp.float32),
      )
    elif case == 'asymmetric_fp8_fine_int4':
      lhs = qarray.QArray(
          jnp.ones((16, 256), jnp.float8_e4m3fn),
          jnp.ones((1, 1), jnp.bfloat16),
          zero_point=jnp.zeros((1, 1), jnp.float8_e4m3fn),
      )
      rhs = qarray.QArray(
          jnp.ones((256, 32), jnp.int4),
          jnp.ones((8, 32), jnp.float32),
      )
    elif case == 'fine_int4_bf16_acts':
      lhs = qarray.QArray(
          jnp.ones((16, 256), jnp.int4),
          jnp.ones((16, 8), jnp.float32),
      )
      rhs = jnp.ones((256, 32), jnp.bfloat16)
    elif case == 'fine_int4_asymmetric_fp8':
      lhs = qarray.QArray(
          jnp.ones((16, 256), jnp.int4),
          jnp.ones((16, 8), jnp.float32),
      )
      rhs = qarray.QArray(
          jnp.ones((256, 32), jnp.float8_e4m3fn),
          jnp.ones((1, 1), jnp.bfloat16),
          zero_point=jnp.zeros((1, 1), jnp.float8_e4m3fn),
      )
    else:
      raise ValueError(f'Unknown case: {case}')

    dnums = (([1], [0]), ([], []))
    with (
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
      _ = dot_general.dot_general(lhs, rhs, dnums)
      mock_slow.assert_called_once()
      mock_fast.assert_not_called()

  def test_get_fp8_dtype(self):
    self.assertEqual(
        dot_general._get_fp8_dtype(jnp.ones((2, 2), jnp.float8_e4m3fn)),
        jnp.float8_e4m3fn,
    )
    self.assertEqual(
        dot_general._get_fp8_dtype(jnp.ones((2, 2), jnp.float8_e5m2)),
        jnp.float8_e5m2,
    )
    self.assertIsNone(
        dot_general._get_fp8_dtype(jnp.ones((2, 2), jnp.float32)),
    )
    self.assertEqual(
        dot_general._get_fp8_dtype(
            qarray.QArray(
                jnp.ones((2, 2), jnp.float8_e4m3fn),
                jnp.ones((1, 1), jnp.bfloat16),
            )
        ),
        jnp.float8_e4m3fn,
    )
    self.assertIsNone(
        dot_general._get_fp8_dtype(
            qarray.QArray(
                jnp.ones((2, 2), jnp.float8_e4m3fn),
                jnp.ones((1, 1), jnp.bfloat16),
                zero_point=jnp.zeros((1, 1), jnp.float8_e4m3fn),
            )
        ),
    )

  def test_can_dequant_operand_on_output_validates_qarray(self):
    # Invalid QArray with mismatched rank between qvalue and scale raises
    # ValueError.
    invalid_qarray = qarray.QArray(
        jnp.ones((16, 16), jnp.int4),
        jnp.ones((16,), jnp.float32),
    )
    with self.assertRaises(ValueError):
      dot_general._can_dequant_operand_on_output(invalid_qarray, [1])

  def test_maybe_dequant_4bit_to_fp8_both_operands_fine_subchannel_no_op(self):
    # When both operands have fine subchannels (< 128), neither can be
    # dequantized on output. _maybe_dequant_4bit_to_fp8 should not dequantize
    # either operand and should return use_fast=False.
    fine_fp8 = qarray.QArray(
        jnp.ones((16, 256), jnp.float8_e4m3fn),
        jnp.ones((16, 8), jnp.bfloat16),
        qtype=jnp.float8_e4m3fn,
    )
    fine_int4 = qarray.QArray(
        jnp.ones((256, 32), jnp.int4),
        jnp.ones((8, 32), jnp.float32),
        qtype=jnp.int4,
    )
    dnums = (([1], [0]), ([], []))
    res_lhs, res_rhs, use_fast = dot_general._maybe_dequant_4bit_to_fp8(
        fine_fp8, fine_int4, dnums
    )
    self.assertFalse(use_fast)
    self.assertIs(res_lhs, fine_fp8)
    self.assertIs(res_rhs, fine_int4)
    self.assertIsInstance(res_rhs, qarray.QArray)
    self.assertEqual(res_rhs.qvalue.dtype, jnp.int4)

  def test_maybe_dequant_4bit_to_fp8_both_operands_coarse_no_op(self):
    # When both operands can be dequantized on output (e.g. coarse tile size >=
    # 128 or per-channel/per-tensor scale), neither operand should be
    # dequantized on input.
    coarse_int4 = qarray.QArray(
        jnp.ones((16, 256), jnp.int4),
        jnp.ones((16, 2), jnp.float32),
        qtype=jnp.int4,
    )
    coarse_fp8 = qarray.QArray(
        jnp.ones((256, 32), jnp.float8_e4m3fn),
        jnp.ones((2, 32), jnp.bfloat16),
        qtype=jnp.float8_e4m3fn,
    )
    dnums = (([1], [0]), ([], []))
    res_lhs, res_rhs, use_fast = dot_general._maybe_dequant_4bit_to_fp8(
        coarse_int4, coarse_fp8, dnums
    )
    self.assertTrue(use_fast)
    self.assertIs(res_lhs, coarse_int4)
    self.assertIs(res_rhs, coarse_fp8)
    self.assertIsInstance(res_lhs, qarray.QArray)
    self.assertEqual(res_lhs.qvalue.dtype, jnp.int4)

  @parameterized.named_parameters(
      dict(
          testcase_name='typical_scale_1e_3_to_1e_2',
          minval=1e-3,
          maxval=1e-2,
          max_rel_mae=0.06,
      ),
      dict(
          # In the subnormal/small scale range (1e-4 to 1e-3), dequantized
          # weights land in or near the subnormal range of float8_e4m3fn
          # (min normal 2^-6 = 0.015625, smallest subnormal 2^-9 = 0.001953125),
          # causing loss of precision (rel_mae up to ~25%) when casting directly
          # to raw FP8 without a retained coarse scale.
          testcase_name='subnormal_scale_1e_4_to_1e_3',
          minval=1e-4,
          maxval=1e-3,
          max_rel_mae=0.25,
      ),
  )
  def test_w4a8_random_numerics(self, minval, maxval, max_rel_mae):
    key = jax.random.PRNGKey(0)
    k1, k2, k3 = jax.random.split(key, 3)
    k = 256
    fp8_qval = jax.random.uniform(k1, (16, k), minval=-4.0, maxval=4.0).astype(
        jnp.float8_e4m3fn
    )
    fp8_scale = jnp.ones((16, 1), jnp.bfloat16) * 0.5
    fp8_op = qarray.QArray(fp8_qval, fp8_scale, qtype=jnp.float8_e4m3fn)

    # Use random per-tile scales at specified magnitude range.
    w4_qval = jax.random.randint(k2, (k, 32), -8, 8, dtype=jnp.int4)
    w4_scale = jax.random.uniform(
        k3, (k // 32, 32), minval=minval, maxval=maxval
    )
    w4_op = qarray.QArray(w4_qval, w4_scale, qtype=jnp.int4)

    dnums = (([1], [0]), ([], []))
    res_dg = dot_general.dot_general(fp8_op, w4_op, dnums)
    res_loop = dot_general.loop_dot_general(fp8_op, w4_op, dnums)
    res_slow = dot_general._slow_dot_general(fp8_op, w4_op, dnums)

    # Loop dot_general and dot_general match closely.
    self.assertTrue(jnp.allclose(res_dg, res_loop, rtol=1e-3, atol=1e-3))

    # Assert relative MAE bound against _slow_dot_general.
    rel_mae = jnp.mean(jnp.abs(res_dg - res_slow)) / jnp.mean(jnp.abs(res_slow))
    logging.info(
        'test_w4a8_random_numerics: scale range [%e, %e], measured rel_mae'
        ' = %f (bound <= %f)',
        minval,
        maxval,
        rel_mae,
        max_rel_mae,
    )
    self.assertLessEqual(rel_mae, max_rel_mae)

  def test_w4a8_overflow_no_nan(self):
    k = 256
    # Construct a 4-bit QArray with large scale such that dequantized |x| > 448
    # (e.g. 7 * 100 = 700). convert_to clips dequantized values to 448
    # (finfo(float8_e4m3fn).max).
    w4_qval = jnp.full((k, 32), 7, dtype=jnp.int4)
    w4_scale = jnp.full((k // 32, 32), 100.0, dtype=jnp.float32)
    w4_op = qarray.QArray(w4_qval, w4_scale, qtype=jnp.int4)

    fp8_op = qarray.QArray(
        jnp.ones((16, k), jnp.float8_e4m3fn),
        jnp.ones((16, 1), jnp.bfloat16),
        qtype=jnp.float8_e4m3fn,
    )
    dnums = (([1], [0]), ([], []))
    res = dot_general.dot_general(fp8_op, w4_op, dnums)
    self.assertFalse(jnp.any(jnp.isnan(res)))
    # 256 contracting elements * (1.0 * 448.0 clipped) = 114688.0.
    expected = jnp.full((16, 32), 256.0 * 448.0, dtype=jnp.float32)
    self.assertTrue(jnp.allclose(res, expected, rtol=1e-3, atol=1e-3))


if __name__ == '__main__':
  absltest.main()
