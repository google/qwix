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

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax import lax
import jax.numpy as jnp
import metrax
import numpy as np
from qwix._src.core import dot_general
from qwix._src.core import dot_general_qt
from qwix._src.core import multipass
from qwix._src.core import qarray


def compute_snr_db(true_val: jax.Array, approx_val: jax.Array) -> float:
  """Computes Signal-to-Noise Ratio (SNR) in decibels (dB) via metrax.SNR."""
  return float(
      metrax.SNR.from_model_output(
          predictions=approx_val.astype(jnp.float32),
          targets=true_val.astype(jnp.float32),
      ).compute()
  )


def compute_relative_error(true_val: jax.Array, approx_val: jax.Array) -> float:
  """Computes relative L2 error ||true - approx|| / ||true||."""
  true_f = true_val.astype(jnp.float32)
  approx_f = approx_val.astype(jnp.float32)
  norm_diff = jnp.linalg.norm(true_f - approx_f)
  norm_true = jnp.linalg.norm(true_f)
  return float(norm_diff / jnp.maximum(norm_true, 1e-12))


class MultiPassTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.rng = jax.random.PRNGKey(42)

  def test_residual_decompose(self):
    key = self.rng
    x = jax.random.normal(key, (4, 32), dtype=jnp.float32)
    how = dot_general.get_how_to_quantize(
        dimension_numbers=(((1,), (0,)), ((), ())),
        ndims=(2, 2),
        for_lhs=True,
        qtype='mxfp8_16',
        tile_size=16,
    )
    passes = multipass.residual_decompose(x, how, n_passes=2)
    self.assertLen(passes, 2)
    q0, q1 = passes
    self.assertIsInstance(q0, qarray.QArray)
    self.assertIsInstance(q1, qarray.QArray)

    # Dequantized values
    x0 = qarray.dequantize(q0)
    x1 = qarray.dequantize(q1)
    reconstructed = x0 + x1

    err_single = compute_relative_error(x, x0)
    err_double = compute_relative_error(x, reconstructed)
    self.assertLess(err_double, err_single)

    snr_single = compute_snr_db(x, x0)
    snr_double = compute_snr_db(x, reconstructed)
    self.assertGreater(snr_double, snr_single)

  def test_residual_decomposition_n_passes_progression(self):
    """Verifies that each residual pass reduces error and increases SNR."""
    key = self.rng
    x = jax.random.normal(key, (8, 64), dtype=jnp.float32)
    how = dot_general.get_how_to_quantize(
        dimension_numbers=(((1,), (0,)), ((), ())),
        ndims=(2, 2),
        for_lhs=True,
        qtype='mxfp8_16',
        tile_size=16,
    )
    n_passes = 4
    passes = multipass.residual_decompose(x, how, n_passes=n_passes)
    self.assertLen(passes, n_passes)

    reconstructed = jnp.zeros_like(x)
    prev_err = float('inf')
    prev_snr = -float('inf')
    prev_scale_mean = float('inf')

    for p in range(n_passes):
      deq_p = qarray.dequantize(passes[p])
      reconstructed = reconstructed + deq_p

      err_p = compute_relative_error(x, reconstructed)
      snr_p = compute_snr_db(x, reconstructed)
      scale_mean = float(jnp.mean(passes[p].scale))

      # Strict error decrease and SNR increase with each added pass
      self.assertLess(err_p, prev_err)
      self.assertGreater(snr_p, prev_snr)
      # Residual scale must drop substantially with each pass (~16x for FP8)
      self.assertLess(scale_mean, prev_scale_mean / 4.0)

      prev_err = err_p
      prev_snr = snr_p
      prev_scale_mean = scale_mean

    # 4 passes of FP8 should achieve near-lossless reconstruction
    # (> 85 dB SNR with float32 eps).
    self.assertGreater(prev_snr, 85.0)

  @parameterized.parameters(
      ('mxfp8_16', 16),
  )
  def test_exact_quantized_input_zero_residual(self, qtype, tile_size):
    """Verifies that on-grid inputs produce zero residual in pass 1."""
    k = self.rng
    raw = jax.random.normal(k, (8, 64), dtype=jnp.float32)
    how = dot_general.get_how_to_quantize(
        dimension_numbers=(((1,), (0,)), ((), ())),
        ndims=(2, 2),
        for_lhs=True,
        qtype=qtype,
        tile_size=tile_size,
    )
    # Project raw values onto the exact quantization grid
    exact_vals = qarray.dequantize(qarray.quantize(raw, how))

    # Decompose the already-quantized values
    passes = multipass.residual_decompose(exact_vals, how, n_passes=2)
    self.assertLen(passes, 2)
    q0, q1 = passes

    # Pass 0 must exactly equal exact_vals
    x0 = qarray.dequantize(q0)
    np.testing.assert_allclose(x0, exact_vals, rtol=1e-5, atol=1e-5)

    # Pass 1 must have identically zero qvalue and dequantize to 0.0
    np.testing.assert_array_equal(q1.qvalue, 0)
    x1 = qarray.dequantize(q1)
    np.testing.assert_allclose(x1, 0.0, atol=1e-7)

  @parameterized.parameters(
      'triangular',
      'full_cross',
      'lhs_high_precision',
      'rhs_high_precision',
  )
  def test_exact_algebraic_component_equivalence(self, mode):
    """Verifies that multipass_dot matches the exact sum of component GEMMs."""
    k1, k2 = jax.random.split(self.rng)
    lhs = jax.random.normal(k1, (8, 32), dtype=jnp.float32)
    rhs = jax.random.normal(k2, (32, 8), dtype=jnp.float32)
    dnums = (((1,), (0,)), ((), ()))

    lhs_how = dot_general.get_how_to_quantize(
        dimension_numbers=dnums,
        ndims=(2, 2),
        for_lhs=True,
        qtype='mxfp8_16',
        tile_size=16,
    )
    rhs_how = dot_general.get_how_to_quantize(
        dimension_numbers=dnums,
        ndims=(2, 2),
        for_lhs=False,
        qtype='mxfp8_16',
        tile_size=16,
    )

    actual = multipass.multipass_dot(
        lhs,
        rhs,
        dimension_numbers=dnums,
        mode=mode,
        lhs_how=lhs_how,
        rhs_how=rhs_how,
    )

    # Compute manual reference terms from dequantized component passes
    a_passes = multipass.residual_decompose(lhs, lhs_how, n_passes=2)
    b_passes = multipass.residual_decompose(rhs, rhs_how, n_passes=2)
    a0 = qarray.dequantize(a_passes[0])
    a1 = qarray.dequantize(a_passes[1])
    b0 = qarray.dequantize(b_passes[0])
    b1 = qarray.dequantize(b_passes[1])

    c00 = lax.dot_general(a0, b0, dnums)
    c01 = lax.dot_general(a0, b1, dnums)
    c10 = lax.dot_general(a1, b0, dnums)
    c11 = lax.dot_general(a1, b1, dnums)

    if mode == 'triangular':
      expected = c00 + c01 + c10
    elif mode == 'full_cross':
      expected = c00 + c01 + c10 + c11
    elif mode == 'lhs_high_precision':
      expected = c00 + c10
    elif mode == 'rhs_high_precision':
      expected = c00 + c01
    else:
      raise ValueError(f'Unexpected mode: {mode}')

    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)

  def test_exact_dropped_cross_residual(self):
    """Verifies full_cross - triangular is identically equal to A_1 @ B_1."""
    k1, k2 = jax.random.split(self.rng)
    lhs = jax.random.normal(k1, (8, 32), dtype=jnp.float32)
    rhs = jax.random.normal(k2, (32, 8), dtype=jnp.float32)
    dnums = (((1,), (0,)), ((), ()))

    lhs_how = dot_general.get_how_to_quantize(
        dimension_numbers=dnums,
        ndims=(2, 2),
        for_lhs=True,
        qtype='mxfp8_16',
        tile_size=16,
    )
    rhs_how = dot_general.get_how_to_quantize(
        dimension_numbers=dnums,
        ndims=(2, 2),
        for_lhs=False,
        qtype='mxfp8_16',
        tile_size=16,
    )

    res_tri = multipass.multipass_dot(
        lhs, rhs, dnums, mode='triangular', lhs_how=lhs_how, rhs_how=rhs_how
    )
    res_full = multipass.multipass_dot(
        lhs, rhs, dnums, mode='full_cross', lhs_how=lhs_how, rhs_how=rhs_how
    )

    a_passes = multipass.residual_decompose(lhs, lhs_how, n_passes=2)
    b_passes = multipass.residual_decompose(rhs, rhs_how, n_passes=2)
    a1 = qarray.dequantize(a_passes[1])
    b1 = qarray.dequantize(b_passes[1])
    a1_b1 = lax.dot_general(a1, b1, dnums)

    # Difference must match A1 @ B1 to floating point precision
    diff = res_full - res_tri
    np.testing.assert_allclose(diff, a1_b1, rtol=1e-5, atol=1e-5)

    # Second order term magnitude must be ~10^-4 or smaller relative to full
    rel_second_order = jnp.linalg.norm(a1_b1) / jnp.linalg.norm(res_full)
    self.assertLess(float(rel_second_order), 1e-3)

  @parameterized.parameters(
      'gaussian',
      'uniform',
      'heavy_tailed',
      'outliers',
  )
  def test_multipass_snr_hierarchy_across_distributions(self, dist):
    """Verifies strict SNR monotonic improvement across distributions."""
    k1, k2, k3, k4 = jax.random.split(self.rng, 4)
    shape_lhs = (16, 64)
    shape_rhs = (64, 16)
    dnums = (((1,), (0,)), ((), ()))

    if dist == 'gaussian':
      lhs = jax.random.normal(k1, shape_lhs, dtype=jnp.float32)
      rhs = jax.random.normal(k2, shape_rhs, dtype=jnp.float32)
    elif dist == 'uniform':
      lhs = jax.random.uniform(k1, shape_lhs, minval=-3.0, maxval=3.0)
      rhs = jax.random.uniform(k2, shape_rhs, minval=-3.0, maxval=3.0)
    elif dist == 'heavy_tailed':
      lhs = jax.random.laplace(k1, shape_lhs, dtype=jnp.float32)
      rhs = jax.random.laplace(k2, shape_rhs, dtype=jnp.float32)
    elif dist == 'outliers':
      lhs = jax.random.normal(k1, shape_lhs, dtype=jnp.float32)
      rhs = jax.random.normal(k2, shape_rhs, dtype=jnp.float32)
      # Add 2% large outliers (30x scale)
      mask_l = jax.random.bernoulli(k3, p=0.02, shape=shape_lhs)
      mask_r = jax.random.bernoulli(k4, p=0.02, shape=shape_rhs)
      lhs = jnp.where(mask_l, lhs * 30.0, lhs)
      rhs = jnp.where(mask_r, rhs * 30.0, rhs)
    else:
      raise ValueError(f'Unknown distribution: {dist}')

    true_res = lax.dot_general(lhs, rhs, dnums)

    # 1-pass baseline
    how_l = dot_general.get_how_to_quantize(
        dimension_numbers=dnums,
        ndims=(2, 2),
        for_lhs=True,
        qtype='mxfp8_16',
        tile_size=16,
    )
    how_r = dot_general.get_how_to_quantize(
        dimension_numbers=dnums,
        ndims=(2, 2),
        for_lhs=False,
        qtype='mxfp8_16',
        tile_size=16,
    )
    res_1p = lax.dot_general(
        qarray.dequantize(qarray.quantize(lhs, how_l)),
        qarray.dequantize(qarray.quantize(rhs, how_r)),
        dnums,
    )
    snr_1p = compute_snr_db(true_res, res_1p)

    # 2-pass (lhs_high_precision)
    res_2p = multipass.multipass_dot(
        lhs,
        rhs,
        dnums,
        mode='lhs_high_precision',
        lhs_how=how_l,
        rhs_how=how_r,
    )
    snr_2p = compute_snr_db(true_res, res_2p)

    # 3-pass (triangular)
    res_3p = multipass.multipass_dot(
        lhs,
        rhs,
        dnums,
        mode='triangular',
        lhs_how=how_l,
        rhs_how=how_r,
    )
    snr_3p = compute_snr_db(true_res, res_3p)

    # 4-pass (full_cross)
    res_4p = multipass.multipass_dot(
        lhs,
        rhs,
        dnums,
        mode='full_cross',
        lhs_how=how_l,
        rhs_how=how_r,
    )
    snr_4p = compute_snr_db(true_res, res_4p)

    # Strict SNR progression: 1p < 2p < 3p <= 4p
    self.assertGreater(snr_2p, snr_1p)
    self.assertGreater(snr_3p, snr_2p)
    self.assertGreaterEqual(snr_4p, snr_3p - 0.05)
    # Triangular should deliver substantial gain over 1-pass (> 20 dB gain)
    self.assertGreater(snr_3p - snr_1p, 20.0)

  @parameterized.parameters(
      ((2, 8, 32), (2, 32, 16), (((2,), (1,)), ((0,), (0,))), (2, 8, 16)),
      (
          (2, 4, 8, 16),
          (2, 4, 16, 8),
          (((3,), (2,)), ((0, 1), (0, 1))),
          (2, 4, 8, 8),
      ),
      ((4, 8, 16), (8, 16, 4), (((1, 2), (0, 1)), ((), ())), (4, 4)),
  )
  def test_batched_and_higher_rank_matmuls(
      self, lhs_shape, rhs_shape, dnums, expected_shape
  ):
    """Verifies multipass_dot works on batched and multi-axis contractions."""
    k1, k2 = jax.random.split(self.rng)
    lhs = jax.random.normal(k1, lhs_shape, dtype=jnp.float32)
    rhs = jax.random.normal(k2, rhs_shape, dtype=jnp.float32)

    true_res = lax.dot_general(lhs, rhs, dnums)

    res = multipass.multipass_dot(
        lhs,
        rhs,
        dimension_numbers=dnums,
        mode='triangular',
        lhs_qtype='mxfp8_16',
        rhs_qtype='mxfp8_16',
        tile_size=16,
    )

    self.assertEqual(res.shape, expected_shape)
    self.assertFalse(jnp.isnan(res).any())

    snr = compute_snr_db(true_res, res)
    self.assertGreater(snr, 45.0)

  @parameterized.parameters(
      'triangular',
      'full_cross',
      'lhs_high_precision',
      'rhs_high_precision',
  )
  def test_dot_general_qt_multipass_fwd(self, mode):
    k1, k2 = jax.random.split(self.rng)
    lhs = jax.random.normal(k1, (4, 32), dtype=jnp.float32)
    rhs = jax.random.normal(k2, (32, 4), dtype=jnp.float32)
    dnums = (((1,), (0,)), ((), ()))

    config = dot_general_qt.DotGeneralQtConfig(
        lhs_qtype='mxfp8_16',
        rhs_qtype='mxfp8_16',
        tile_size=16,
        multipass_mode=mode,
    )
    res = dot_general_qt.dot_general_qt(lhs, rhs, dnums, config)
    self.assertEqual(res.shape, (4, 4))
    self.assertFalse(jnp.isnan(res).any())

  def test_gradient_accuracy_and_convergence_vs_float32(self):
    """Verifies that multi-pass backward gradients converge to float32."""
    k1, k2, k3 = jax.random.split(self.rng, 3)
    lhs = jax.random.normal(k1, (16, 32), dtype=jnp.float32)
    rhs = jax.random.normal(k2, (32, 16), dtype=jnp.float32)
    target = jax.random.normal(k3, (16, 16), dtype=jnp.float32)
    dnums = (((1,), (0,)), ((), ()))

    # Loss function with continuous non-trivial incoming gradient
    def true_loss(a, b):
      out = lax.dot_general(a, b, dnums)
      return 0.5 * jnp.sum((out - target) ** 2)

    true_ga, true_gb = jax.grad(true_loss, argnums=(0, 1))(lhs, rhs)

    # 1. Single pass fwd & bwd config (quantizing grad and residual in bwd)
    cfg_1p = dot_general_qt.DotGeneralQtConfig(
        lhs_qtype='mxfp8_16',
        rhs_qtype='mxfp8_16',
        tile_size=16,
        dlhs_grad_qtype='mxfp8_16',
        drhs_grad_qtype='mxfp8_16',
        dlhs_residual_qtype='mxfp8_16',
        drhs_residual_qtype='mxfp8_16',
        dlhs_tile_size=16,
        drhs_tile_size=16,
    )

    def loss_1p(a, b):
      out = dot_general_qt.dot_general_qt(a, b, dnums, cfg_1p)
      return 0.5 * jnp.sum((out - target) ** 2)

    ga_1p, gb_1p = jax.grad(loss_1p, argnums=(0, 1))(lhs, rhs)
    snr_ga_1p = compute_snr_db(true_ga, ga_1p)
    snr_gb_1p = compute_snr_db(true_gb, gb_1p)

    # 2. Multi-pass triangular fwd & bwd config
    cfg_tri = dot_general_qt.DotGeneralQtConfig(
        lhs_qtype='mxfp8_16',
        rhs_qtype='mxfp8_16',
        tile_size=16,
        multipass_mode='triangular',
        dlhs_grad_qtype='mxfp8_16',
        drhs_grad_qtype='mxfp8_16',
        dlhs_residual_qtype='mxfp8_16',
        drhs_residual_qtype='mxfp8_16',
        dlhs_tile_size=16,
        drhs_tile_size=16,
        dlhs_multipass_mode='triangular',
        drhs_multipass_mode='triangular',
    )

    def loss_tri(a, b):
      out = dot_general_qt.dot_general_qt(a, b, dnums, cfg_tri)
      return 0.5 * jnp.sum((out - target) ** 2)

    ga_tri, gb_tri = jax.grad(loss_tri, argnums=(0, 1))(lhs, rhs)
    snr_ga_tri = compute_snr_db(true_ga, ga_tri)
    snr_gb_tri = compute_snr_db(true_gb, gb_tri)

    self.assertFalse(jnp.isnan(ga_tri).any())
    self.assertFalse(jnp.isnan(gb_tri).any())
    # Multi-pass gradient should achieve high fidelity (> 45 dB SNR)
    self.assertGreater(snr_ga_tri, 45.0)
    self.assertGreater(snr_gb_tri, 45.0)
    # Multi-pass gradient should have substantially higher SNR than single pass
    self.assertGreater(snr_ga_tri, snr_ga_1p + 15.0)
    self.assertGreater(snr_gb_tri, snr_gb_1p + 15.0)

  @parameterized.parameters(
      'triangular',
      'full_cross',
  )
  def test_numerical_edge_cases(self, mode):
    """Verifies behavior on zero matrices, asymmetric zeros, and wide dynamic range."""
    dnums = (((1,), (0,)), ((), ()))

    # Zero matrices
    zeros_l = jnp.zeros((4, 16), dtype=jnp.float32)
    zeros_r = jnp.zeros((16, 4), dtype=jnp.float32)
    res_zero = multipass.multipass_dot(
        zeros_l, zeros_r, dnums, mode=mode, tile_size=16
    )
    np.testing.assert_array_equal(res_zero, 0.0)
    self.assertFalse(jnp.isnan(res_zero).any())

    # One side zero
    k1 = self.rng
    normal_l = jax.random.normal(k1, (4, 16), dtype=jnp.float32)
    res_one_zero = multipass.multipass_dot(
        normal_l, zeros_r, dnums, mode=mode, tile_size=16
    )
    np.testing.assert_array_equal(res_one_zero, 0.0)
    self.assertFalse(jnp.isnan(res_one_zero).any())

    # Extreme scaling (1e-5 to 1e5)
    k2, k3 = jax.random.split(self.rng)
    small_l = jax.random.normal(k2, (4, 16), dtype=jnp.float32) * 1e-4
    large_r = jax.random.normal(k3, (16, 4), dtype=jnp.float32) * 1e4
    res_scaled = multipass.multipass_dot(
        small_l, large_r, dnums, mode=mode, tile_size=16
    )
    self.assertFalse(jnp.isnan(res_scaled).any())
    self.assertFalse(jnp.isinf(res_scaled).any())
    true_scaled = lax.dot_general(small_l, large_r, dnums)
    snr = compute_snr_db(true_scaled, res_scaled)
    self.assertGreater(snr, 40.0)

  def test_bilinear_scaling_and_transposition_symmetry(self):
    """Verifies transposition symmetry and exact bilinear scaling of multi-pass."""
    k1, k2 = jax.random.split(self.rng)
    lhs = jax.random.normal(k1, (16, 32), dtype=jnp.float32)
    rhs = jax.random.normal(k2, (32, 16), dtype=jnp.float32)
    dnums = (((1,), (0,)), ((), ()))

    res = multipass.multipass_dot(
        lhs, rhs, dnums, mode='triangular', tile_size=16
    )

    # Transposition symmetry: (A @ B).T == B.T @ A.T
    res_t = multipass.multipass_dot(
        rhs.T, lhs.T, dnums, mode='triangular', tile_size=16
    )
    np.testing.assert_allclose(res.T, res_t, rtol=1e-5, atol=1e-5)

    # Power-of-2 scaling linearity: multipass(2 * A, B) == 2 * multipass(A, B)
    res_scaled = multipass.multipass_dot(
        lhs * 2.0, rhs, dnums, mode='triangular', tile_size=16
    )
    np.testing.assert_allclose(res * 2.0, res_scaled, rtol=1e-5, atol=1e-5)

  def test_multipass_jit_compatibility(self):
    """Verifies that multipass_dot compiles under jax.jit and matches eager."""
    k1, k2 = jax.random.split(self.rng)
    lhs = jax.random.normal(k1, (8, 32), dtype=jnp.float32)
    rhs = jax.random.normal(k2, (32, 8), dtype=jnp.float32)
    dnums = (((1,), (0,)), ((), ()))

    eager_res = multipass.multipass_dot(
        lhs, rhs, dnums, mode='triangular', tile_size=16
    )

    jitted_dot = jax.jit(
        multipass.multipass_dot,
        static_argnames=(
            'dimension_numbers',
            'mode',
            'lhs_qtype',
            'rhs_qtype',
            'tile_size',
        ),
    )
    jit_res = jitted_dot(lhs, rhs, dnums, mode='triangular', tile_size=16)

    np.testing.assert_array_equal(eager_res, jit_res)

  def test_transformer_attention_multipass_accuracy(self):
    """Verifies multi-pass accuracy on 4D Transformer attention operations."""
    # (batch=2, num_heads=4, seq_len=16, head_dim=32)
    k1, k2, k3 = jax.random.split(self.rng, 3)
    query = jax.random.normal(k1, (2, 4, 16, 32), dtype=jnp.float32)
    key = jax.random.normal(k2, (2, 4, 16, 32), dtype=jnp.float32)
    value = jax.random.normal(k3, (2, 4, 16, 32), dtype=jnp.float32)

    # Q @ K.T: contracting head_dim (axis 3)
    qk_dnums = (((3,), (3,)), ((0, 1), (0, 1)))
    true_logits = lax.dot_general(query, key, qk_dnums)

    # 1-pass baseline
    how_q = dot_general.get_how_to_quantize(
        dimension_numbers=qk_dnums,
        ndims=(4, 4),
        for_lhs=True,
        qtype='mxfp8_16',
        tile_size=16,
    )
    how_k = dot_general.get_how_to_quantize(
        dimension_numbers=qk_dnums,
        ndims=(4, 4),
        for_lhs=False,
        qtype='mxfp8_16',
        tile_size=16,
    )
    logits_1p = lax.dot_general(
        qarray.dequantize(qarray.quantize(query, how_q)),
        qarray.dequantize(qarray.quantize(key, how_k)),
        qk_dnums,
    )
    snr_1p = compute_snr_db(true_logits, logits_1p)

    # Multi-pass triangular (3-pass)
    logits_tri = multipass.multipass_dot(
        query,
        key,
        dimension_numbers=qk_dnums,
        mode='triangular',
        lhs_how=how_q,
        rhs_how=how_k,
    )
    snr_tri = compute_snr_db(true_logits, logits_tri)

    self.assertEqual(logits_tri.shape, (2, 4, 16, 16))
    self.assertGreater(snr_tri, 50.0)
    self.assertGreater(snr_tri - snr_1p, 20.0)

    # Attn_weights @ V: contracting seq_len (axis 3 of weights, axis 2 of V)
    weights = jax.nn.softmax(logits_tri, axis=-1)
    av_dnums = (((3,), (2,)), ((0, 1), (0, 1)))
    true_context = lax.dot_general(weights, value, av_dnums)

    context_tri = multipass.multipass_dot(
        weights,
        value,
        dimension_numbers=av_dnums,
        mode='triangular',
        lhs_qtype='mxfp8_16',
        rhs_qtype='mxfp8_16',
        tile_size=16,
    )
    self.assertEqual(context_tri.shape, (2, 4, 16, 32))
    snr_context = compute_snr_db(true_context, context_tri)
    self.assertGreater(snr_context, 50.0)

  def test_normal_distribution_sqnr_benchmark(self):
    """Verifies SQNR and relative error on N(0, 1) inputs against reference table.

    Cross-checks the QWIX multi-pass and asymmetric implementation against the
    published benchmarks in zfc_emulation_utils/README.md Table 2.1.
    """
    k1, k2 = jax.random.split(self.rng)
    shape_l = (256, 512)
    shape_r = (512, 256)
    dnums = (((1,), (0,)), ((), ()))

    # Continuous standard normal Gaussian inputs N(0, 1)
    lhs = jax.random.normal(k1, shape_l, dtype=jnp.float32)
    rhs = jax.random.normal(k2, shape_r, dtype=jnp.float32)
    ref_f32 = lax.dot_general(lhs, rhs, dnums)

    # 1. Single-Pass FP8 (1 pass, block size 16) - target ~28.41 dB
    how_l_fp8 = dot_general.get_how_to_quantize(
        dimension_numbers=dnums,
        ndims=(2, 2),
        for_lhs=True,
        qtype='mxfp8_16',
        tile_size=16,
    )
    how_r_fp8 = dot_general.get_how_to_quantize(
        dimension_numbers=dnums,
        ndims=(2, 2),
        for_lhs=False,
        qtype='mxfp8_16',
        tile_size=16,
    )
    res_fp8_1p = lax.dot_general(
        qarray.dequantize(qarray.quantize(lhs, how_l_fp8)),
        qarray.dequantize(qarray.quantize(rhs, how_r_fp8)),
        dnums,
    )
    snr_fp8_1p = float(compute_snr_db(ref_f32, res_fp8_1p))
    err_fp8_1p = float(compute_relative_error(ref_f32, res_fp8_1p))

    # 2. Asymmetric FP8 2-Pass (2 passes, block size 16) - target ~31.56 dB
    res_fp8_2p = multipass.multipass_dot(
        lhs,
        rhs,
        dnums,
        mode='lhs_high_precision',
        lhs_how=how_l_fp8,
        rhs_how=how_r_fp8,
    )
    snr_fp8_2p = float(compute_snr_db(ref_f32, res_fp8_2p))
    err_fp8_2p = float(compute_relative_error(ref_f32, res_fp8_2p))

    # 3. Multi-Pass FP8 Triangular (3 passes, block size 16) - target ~54.84 dB
    res_fp8_tri = multipass.multipass_dot(
        lhs,
        rhs,
        dnums,
        mode='triangular',
        lhs_how=how_l_fp8,
        rhs_how=how_r_fp8,
    )
    snr_fp8_tri = float(compute_snr_db(ref_f32, res_fp8_tri))
    err_fp8_tri = float(compute_relative_error(ref_f32, res_fp8_tri))

    # 4. Multi-Pass FP8 Full Cross (4 passes, block size 16) - target ~55.59 dB
    res_fp8_full = multipass.multipass_dot(
        lhs,
        rhs,
        dnums,
        mode='full_cross',
        lhs_how=how_l_fp8,
        rhs_how=how_r_fp8,
    )
    snr_fp8_full = float(compute_snr_db(ref_f32, res_fp8_full))
    err_fp8_full = float(compute_relative_error(ref_f32, res_fp8_full))

    # Also evaluate with bfloat16 accumulation (matching zfc_emulation_utils)
    res_fp8_tri_bf16 = multipass.multipass_dot(
        lhs,
        rhs,
        dnums,
        mode='triangular',
        lhs_how=how_l_fp8,
        rhs_how=how_r_fp8,
        preferred_element_type=jnp.bfloat16,
    )
    snr_fp8_tri_bf16 = float(compute_snr_db(ref_f32, res_fp8_tri_bf16))
    err_fp8_tri_bf16 = float(compute_relative_error(ref_f32, res_fp8_tri_bf16))

    print('=== SQNR BENCHMARKS ===')
    print(f'FP8 1-pass: SNR={snr_fp8_1p:.2f} dB, err={err_fp8_1p:.4f}')
    print(f'FP8 2-pass: SNR={snr_fp8_2p:.2f} dB, err={err_fp8_2p:.4f}')
    print(f'FP8 3-pass (tri): SNR={snr_fp8_tri:.2f} dB, err={err_fp8_tri:.4f}')
    print(f'FP8 4-pass (full): SNR={snr_fp8_full:.2f} dB')
    print(f'FP8 3-pass (bf16): SNR={snr_fp8_tri_bf16:.2f} dB')

    # Reference values aligned with zfc_emulation_utils/README.md
    # 1-pass FP8: ~28.4 dB, relative error ~0.038
    self.assertAlmostEqual(snr_fp8_1p, 28.41, delta=1.5)
    self.assertAlmostEqual(err_fp8_1p, 0.0380, delta=0.01)

    # 2-pass Asymmetric FP8: ~31.5 dB (+3.1 dB over 1-pass)
    self.assertAlmostEqual(snr_fp8_2p, 31.56, delta=1.5)
    self.assertAlmostEqual(err_fp8_2p, 0.0264, delta=0.01)
    self.assertGreater(snr_fp8_2p, snr_fp8_1p + 2.0)

    # 3-pass Triangular FP8: > 54.0 dB (fp32), > 49.0 dB (bf16)
    self.assertGreater(snr_fp8_tri, 54.0)
    self.assertGreater(snr_fp8_tri_bf16, 49.0)
    self.assertLess(err_fp8_tri, 0.002)
    self.assertLess(err_fp8_tri_bf16, 0.005)

    # 4-pass Full Cross FP8: > 55.0 dB in float32
    self.assertGreater(snr_fp8_full, 55.0)
    self.assertLess(err_fp8_full, 0.002)
    self.assertGreaterEqual(snr_fp8_full, snr_fp8_tri - 0.2)

    # Strict SNR progression hierarchy across passes
    self.assertGreater(snr_fp8_2p, snr_fp8_1p)
    self.assertGreater(snr_fp8_tri, snr_fp8_2p)
    self.assertGreaterEqual(snr_fp8_full, snr_fp8_tri - 0.2)

  @parameterized.parameters(
      ((8, 16), (16, 8)),
      ((16, 32), (32, 16)),
      ((32, 64), (64, 32)),
      ((64, 64), (64, 64)),
  )
  def test_karatsuba_exact_int8_multiplication(self, shape_l, shape_r):
    """Verifies Karatsuba 3-pass produces 100% bit-exact results to signed int8 matmul."""
    k1, k2 = jax.random.split(self.rng)
    a = jax.random.randint(k1, shape_l, minval=-128, maxval=128).astype(
        jnp.int32
    )
    b = jax.random.randint(k2, shape_r, minval=-128, maxval=128).astype(
        jnp.int32
    )
    ref = jnp.matmul(a, b)

    dnums = (((1,), (0,)), ((), ()))
    res = multipass.karatsuba_signed_int8_dot(a, b, dnums)
    np.testing.assert_array_equal(res, ref)
    self.assertEqual(int(jnp.max(jnp.abs(res - ref))), 0)

  @parameterized.parameters(
      ((8, 16), (16, 8)),
      ((16, 32), (32, 16)),
      ((32, 64), (64, 32)),
      ((64, 64), (64, 64)),
  )
  def test_naive_exact_int8_multiplication(self, shape_l, shape_r):
    """Verifies Naive 4-pass produces 100% bit-exact results to signed int8 matmul."""
    k1, k2 = jax.random.split(self.rng)
    a = jax.random.randint(k1, shape_l, minval=-128, maxval=128).astype(
        jnp.int32
    )
    b = jax.random.randint(k2, shape_r, minval=-128, maxval=128).astype(
        jnp.int32
    )
    ref = jnp.matmul(a, b)

    dnums = (((1,), (0,)), ((), ()))
    res = multipass.naive_signed_int8_dot(a, b, dnums)
    np.testing.assert_array_equal(res, ref)
    self.assertEqual(int(jnp.max(jnp.abs(res - ref))), 0)

  def test_karatsuba_and_naive_equivalence_and_extremes(self):
    """Verifies Karatsuba and Naive 4-pass match bit-for-bit on extreme boundaries."""
    # Test boundary values: -128, -127, -8, -1, 0, 1, 7, 127
    vals = jnp.array([-128, -127, -8, -1, 0, 1, 7, 127], dtype=jnp.int32)
    a = jnp.tile(vals[:, None], (1, 8))
    b = jnp.tile(vals[None, :], (8, 1))
    dnums = (((1,), (0,)), ((), ()))

    ref = jnp.matmul(a, b)
    res_kara = multipass.karatsuba_signed_int8_dot(a, b, dnums)
    res_naive = multipass.naive_signed_int8_dot(a, b, dnums)

    np.testing.assert_array_equal(res_kara, ref)
    np.testing.assert_array_equal(res_naive, ref)
    np.testing.assert_array_equal(res_kara, res_naive)

  @parameterized.parameters(
      ((8, 16), (16, 8)),
      ((16, 32), (32, 16)),
      ((32, 64), (64, 32)),
  )
  def test_asymmetric_exact_int4_int8_multiplication(self, shape_l, shape_r):
    """Verifies 2-pass asymmetric signed INT4 x INT8 matmul produces exact integer products."""
    k1, k2 = jax.random.split(self.rng)
    # Case 1: LHS is INT4 [-8, 7], RHS is INT8 [-128, 127]
    a_4 = jax.random.randint(k1, shape_l, minval=-8, maxval=8).astype(jnp.int32)
    b_8 = jax.random.randint(k2, shape_r, minval=-128, maxval=128).astype(
        jnp.int32
    )
    ref1 = jnp.matmul(a_4, b_8)
    dnums = (((1,), (0,)), ((), ()))
    res1 = multipass.asymmetric_signed_int4_int8_dot(
        a_4, b_8, dnums, int4_operand='lhs'
    )
    np.testing.assert_array_equal(res1, ref1)
    self.assertEqual(int(jnp.max(jnp.abs(res1 - ref1))), 0)

    # Case 2: LHS is INT8 [-128, 127], RHS is INT4 [-8, 7]
    a_8 = jax.random.randint(k1, shape_l, minval=-128, maxval=128).astype(
        jnp.int32
    )
    b_4 = jax.random.randint(k2, shape_r, minval=-8, maxval=8).astype(jnp.int32)
    ref2 = jnp.matmul(a_8, b_4)
    res2 = multipass.asymmetric_signed_int4_int8_dot(
        a_8, b_4, dnums, int4_operand='rhs'
    )
    np.testing.assert_array_equal(res2, ref2)
    self.assertEqual(int(jnp.max(jnp.abs(res2 - ref2))), 0)

  def test_triangular_signed_int8_dot(self):
    """Verifies 3-pass triangular int8 matmul drops exactly p00 and achieves >33 dB SQNR."""
    k1, k2 = jax.random.split(self.rng)
    shape_l = (32, 64)
    shape_r = (64, 32)
    a = jax.random.randint(k1, shape_l, minval=-128, maxval=128).astype(
        jnp.int32
    )
    b = jax.random.randint(k2, shape_r, minval=-128, maxval=128).astype(
        jnp.int32
    )
    ref = jnp.matmul(a, b)
    dnums = (((1,), (0,)), ((), ()))
    res = multipass.triangular_signed_int8_dot(a, b, dnums)

    # Algebraic identity check: ref - res must be exactly p00 = a_l * b_l
    a_l = ((a.astype(jnp.uint8) & 0x0F).astype(jnp.int32)) - 8
    b_l = ((b.astype(jnp.uint8) & 0x0F).astype(jnp.int32)) - 8
    p00 = jnp.matmul(a_l, b_l)
    diff = ref - res
    np.testing.assert_array_equal(diff, p00)

    # SQNR must exceed 33.0 dB
    sqnr = compute_snr_db(ref, res)
    self.assertGreater(sqnr, 33.0)

  @parameterized.parameters('karatsuba', 'naive', 'triangular', 'asymmetric')
  def test_mxint8_multipass_dot(self, method):
    """Verifies microscaled mxint8 GEMM across all 4 multi-pass methods."""
    k1, k2 = jax.random.split(self.rng)
    lhs = jax.random.normal(k1, (16, 64), dtype=jnp.float32)
    rhs = jax.random.normal(k2, (64, 16), dtype=jnp.float32)
    dnums = (((1,), (0,)), ((), ()))

    res_multipass = multipass.mxint8_multipass_dot(
        lhs, rhs, dnums, method=method, tile_size=32
    )
    self.assertEqual(res_multipass.shape, (16, 16))
    self.assertFalse(jnp.isnan(res_multipass).any())

    true_dot = lax.dot_general(lhs, rhs, dnums)
    snr = compute_snr_db(true_dot, res_multipass)
    if method in ('karatsuba', 'naive'):
      self.assertGreater(snr, 36.0)
    elif method == 'triangular':
      self.assertGreater(snr, 33.0)
    elif method == 'asymmetric':
      self.assertGreater(snr, 16.0)

  def test_residual_decompose_hybrid(self):
    """Verifies heterogeneous residual decomposition (mxfp8_16 Pass 1 + mxfp4 Pass 2)."""
    k1, _ = jax.random.split(self.rng)
    x = jax.random.normal(k1, (32, 64), dtype=jnp.float32)
    dnums = (((1,), (0,)), ((), ()))

    how_p1 = dot_general.get_how_to_quantize(
        dimension_numbers=dnums,
        ndims=(2, 2),
        for_lhs=True,
        qtype='mxfp8_16',
        tile_size=16,
    )
    how_p2 = dot_general.get_how_to_quantize(
        dimension_numbers=dnums,
        ndims=(2, 2),
        for_lhs=True,
        qtype='mxfp4',
        tile_size=32,
    )

    q0, q1 = multipass.residual_decompose_hybrid(x, (how_p1, how_p2))
    self.assertEqual(q0.qtype, 'mxfp8_16')
    self.assertEqual(q1.qtype, 'mxfp4')

    x0 = qarray.dequantize(q0)
    x1 = qarray.dequantize(q1)
    reconstructed = x0 + x1

    snr_p1 = compute_snr_db(x, x0)
    snr_hybrid = compute_snr_db(x, reconstructed)

    self.assertGreater(snr_p1, 26.0)
    self.assertGreater(snr_hybrid, 42.0)
    self.assertGreater(snr_hybrid - snr_p1, 14.0)

  @parameterized.parameters(
      'triangular',
      'full_cross',
      'lhs_high_precision',
      'rhs_high_precision',
      'reconstructed',
  )
  def test_hybrid_fp8_fp4_multipass_dot_modes(self, mode):
    """Verifies hybrid FP8+FP4 multi-pass GEMM across execution modes."""
    k1, k2 = jax.random.split(self.rng)
    lhs = jax.random.normal(k1, (16, 64), dtype=jnp.float32)
    rhs = jax.random.normal(k2, (64, 16), dtype=jnp.float32)
    dnums = (((1,), (0,)), ((), ()))

    res = multipass.hybrid_fp8_fp4_multipass_dot(lhs, rhs, dnums, mode=mode)
    self.assertEqual(res.shape, (16, 16))
    self.assertFalse(jnp.isnan(res).any())

    ref = lax.dot_general(lhs, rhs, dnums)
    snr = compute_snr_db(ref, res)

    if mode in ('full_cross', 'reconstructed'):
      self.assertGreater(snr, 44.0)
    elif mode == 'triangular':
      self.assertGreater(snr, 42.0)
    elif mode in ('lhs_high_precision', 'rhs_high_precision'):
      self.assertGreater(snr, 30.5)

  def test_hybrid_fp8_fp4_dropped_cross_residual(self):
    """Verifies full_cross - triangular matches exactly A_1 @ B_1 in hybrid mode."""
    k1, k2 = jax.random.split(self.rng)
    lhs = jax.random.normal(k1, (16, 64), dtype=jnp.float32)
    rhs = jax.random.normal(k2, (64, 16), dtype=jnp.float32)
    dnums = (((1,), (0,)), ((), ()))

    res_tri = multipass.hybrid_fp8_fp4_multipass_dot(
        lhs, rhs, dnums, mode='triangular'
    )
    res_full = multipass.hybrid_fp8_fp4_multipass_dot(
        lhs, rhs, dnums, mode='full_cross'
    )

    lhs_how_p1 = dot_general.get_how_to_quantize(
        dimension_numbers=dnums,
        ndims=(2, 2),
        for_lhs=True,
        qtype='mxfp8_16',
        tile_size=16,
    )
    lhs_how_p2 = dot_general.get_how_to_quantize(
        dimension_numbers=dnums,
        ndims=(2, 2),
        for_lhs=True,
        qtype='mxfp4',
        tile_size=32,
    )
    rhs_how_p1 = dot_general.get_how_to_quantize(
        dimension_numbers=dnums,
        ndims=(2, 2),
        for_lhs=False,
        qtype='mxfp8_16',
        tile_size=16,
    )
    rhs_how_p2 = dot_general.get_how_to_quantize(
        dimension_numbers=dnums,
        ndims=(2, 2),
        for_lhs=False,
        qtype='mxfp4',
        tile_size=32,
    )

    _, a1_q = multipass.residual_decompose_hybrid(lhs, (lhs_how_p1, lhs_how_p2))
    _, b1_q = multipass.residual_decompose_hybrid(rhs, (rhs_how_p1, rhs_how_p2))
    a1 = qarray.dequantize(a1_q)
    b1 = qarray.dequantize(b1_q)
    a1_b1 = lax.dot_general(a1, b1, dnums)

    diff = res_full - res_tri
    np.testing.assert_allclose(diff, a1_b1, rtol=1e-5, atol=1e-5)


if __name__ == '__main__':
  absltest.main()
