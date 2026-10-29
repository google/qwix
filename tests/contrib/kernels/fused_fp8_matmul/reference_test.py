# Copyright 2026 Google LLC
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

"""Tests for the numerics and XLA reference of the fused FP8 matmul."""

from absl.testing import absltest
import jax
import jax.numpy as jnp
import numpy as np
from qwix.contrib.kernels.fused_fp8_matmul import reference
from qwix.contrib.kernels.fused_fp8_matmul import test_utils


class ReferenceVjpTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    m, k, n = 512, 1024, 2048
    self.lhs, self.rhs = test_utils.random_inputs(m, k, n, jnp.bfloat16)
    self.dout = jax.random.normal(jax.random.PRNGKey(3), (m, n), jnp.bfloat16)
    _, self.residuals = reference.fused_fp8_matmul_reference(self.lhs, self.rhs)

  def test_residual_layout(self):
    m, k = self.lhs.shape
    n = self.rhs.shape[1]
    lhs_q, lhs_scale, rhs_q, rhs_scale = self.residuals
    self.assertEqual((lhs_q.shape, lhs_q.dtype), ((m, k), jnp.float8_e4m3fn))
    self.assertEqual((lhs_scale.shape, lhs_scale.dtype), ((m, 1), jnp.float32))
    self.assertEqual((rhs_q.shape, rhs_q.dtype), ((k, n), jnp.float8_e4m3fn))
    self.assertEqual((rhs_scale.shape, rhs_scale.dtype), ((n,), jnp.float32))

  def test_dlhs_reference_close_to_f32_matmul(self):
    _, _, rhs_q, rhs_scale = self.residuals
    dlhs = jax.jit(reference.fused_fp8_dlhs_reference)(
        self.dout, rhs_q, rhs_scale
    )
    self.assertEqual(
        (dlhs.shape, dlhs.dtype), (self.lhs.shape, self.dout.dtype)
    )
    rhs = rhs_q.astype(jnp.float32) * rhs_scale
    expected = jnp.matmul(
        self.dout.astype(jnp.float32),
        rhs.T,
        precision=jax.lax.Precision.HIGHEST,
    )
    # Only the cotangent is quantized here (the weight is already FP8 in both
    # products), so the gap is one operand's worth of FP8 rounding.
    self.assertLess(
        test_utils.relative_error(dlhs, expected), test_utils.FP8_NOISE_TOL
    )

  def test_drhs_reference_is_f32_product_of_dequantized_lhs(self):
    lhs_q, lhs_scale, _, _ = self.residuals
    drhs = jax.jit(reference.fused_fp8_drhs_reference)(
        lhs_q, lhs_scale, self.dout
    )
    self.assertEqual(
        (drhs.shape, drhs.dtype), (self.rhs.shape, self.dout.dtype)
    )
    expected = jnp.matmul(
        (lhs_q.astype(jnp.float32) * lhs_scale).T,
        self.dout.astype(jnp.float32),
        precision=jax.lax.Precision.HIGHEST,
    )
    # Nothing is quantized in this product; only bf16 rounding differs.
    self.assertLess(
        test_utils.relative_error(drhs, expected),
        test_utils.BF16_ROUNDING_TOL,
    )


class QuantizationTest(absltest.TestCase):

  def test_quantize_rows_fp8(self):
    x = jax.random.normal(jax.random.PRNGKey(0), (64, 256), jnp.bfloat16)
    xq, scale = reference.quantize_rows_fp8(x)
    self.assertEqual((xq.shape, xq.dtype), (x.shape, jnp.float8_e4m3fn))
    self.assertEqual((scale.shape, scale.dtype), ((64, 1), jnp.float32))
    # Every row's absmax lands on the FP8 max (up to bf16 rounding of the
    # scale, i.e. within one FP8 ulp), and nothing overflows to NaN.
    xq_f32 = np.asarray(xq.astype(jnp.float32))
    self.assertTrue(np.all(np.isfinite(xq_f32)))
    np.testing.assert_allclose(
        np.abs(xq_f32).max(axis=1), reference.FP8_MAX, rtol=0.08
    )
    np.testing.assert_allclose(
        xq_f32 * np.asarray(scale),
        np.asarray(x, np.float32),
        rtol=test_utils.FP8_RECONSTRUCTION_RTOL,
    )

  def test_zero_row_does_not_divide_by_zero(self):
    x = jnp.zeros((8, 128), jnp.bfloat16)
    xq, scale = reference.quantize_rows_fp8(x)
    self.assertTrue(bool(jnp.all(jnp.isfinite(scale))))
    np.testing.assert_array_equal(np.asarray(xq.astype(jnp.float32)), 0.0)

  def test_qarray_rhs_is_used_as_is(self):
    rhs = test_utils.random_inputs(8, 256, 128, jnp.bfloat16)[1]
    rhs_q = test_utils.quantize_per_column(rhs)
    q, scale = reference.rhs_as_fp8(rhs_q)
    self.assertIs(q, rhs_q.qvalue)
    self.assertEqual((scale.shape, scale.dtype), ((128,), jnp.float32))

  def test_plain_rhs_takes_the_qwix_quantization_path(self):
    rhs = test_utils.random_inputs(8, 256, 128, jnp.bfloat16)[1]
    q, scale = reference.rhs_as_fp8(rhs)
    q_qwix, scale_qwix = reference.rhs_as_fp8(
        test_utils.quantize_per_column(rhs)
    )
    np.testing.assert_array_equal(
        np.asarray(q.astype(jnp.float32)),
        np.asarray(q_qwix.astype(jnp.float32)),
    )
    np.testing.assert_array_equal(np.asarray(scale), np.asarray(scale_qwix))


if __name__ == "__main__":
  absltest.main()
