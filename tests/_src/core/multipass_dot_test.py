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
from qwix._src.core import multipass_dot
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


def is_tpu() -> bool:
  devices = jax.devices()
  return bool(devices and devices[0].platform == 'tpu')


def extract_all_equations(jaxpr: jax.core.Jaxpr) -> list[jax.core.JaxprEqn]:
  """Recursively extracts all equations from a Jaxpr including nested JIT calls."""
  eqns = []
  for eqn in jaxpr.eqns:
    eqns.append(eqn)
    if 'jaxpr' in eqn.params:
      closed_jpr = eqn.params['jaxpr']
      if hasattr(closed_jpr, 'jaxpr'):
        eqns.extend(extract_all_equations(closed_jpr.jaxpr))
    if 'call_jaxpr' in eqn.params:
      eqns.extend(extract_all_equations(eqn.params['call_jaxpr']))
  return eqns


def get_graph_gemms_and_casts(
    fn, *args
) -> tuple[list[jax.core.JaxprEqn], set[jnp.dtype]]:
  """Extracts all GEMM equations and cast target dtypes from a traced JAX function."""
  jaxpr = jax.make_jaxpr(fn)(*args)
  all_eqns = extract_all_equations(
      jaxpr.jaxpr if hasattr(jaxpr, 'jaxpr') else jaxpr
  )
  gemms = [
      eqn
      for eqn in all_eqns
      if eqn.primitive.name in ('dot_general', 'scaled_matmul_wrapper')
  ]
  casts = [
      eqn for eqn in all_eqns if eqn.primitive.name == 'convert_element_type'
  ]
  cast_dtypes = {jnp.dtype(eqn.params['new_dtype']) for eqn in casts}
  return gemms, cast_dtypes


class MultiPassDotTest(parameterized.TestCase):

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
        qtype=jnp.float8_e4m3fn,
        tile_size=None,
    )
    passes = multipass_dot.residual_decompose(x, how, n_passes=2)
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

  def test_quantized_input_raises_error(self):
    """Verifies that passing QArray to multipass raises ValueError."""
    key = self.rng
    x = jax.random.normal(key, (4, 32), dtype=jnp.float32)
    how = dot_general.get_how_to_quantize(
        dimension_numbers=(((1,), (0,)), ((), ())),
        ndims=(2, 2),
        for_lhs=True,
        qtype=jnp.float8_e4m3fn,
        tile_size=None,
    )
    qx = qarray.quantize(x, how)
    with self.assertRaisesRegex(ValueError, 'must strictly be unquantized'):
      multipass_dot.residual_decompose(qx, how)
    with self.assertRaisesRegex(ValueError, 'must strictly be unquantized'):
      multipass_dot.multipass_dot_general(qx, x)
    with self.assertRaisesRegex(ValueError, 'must strictly be unquantized'):
      multipass_dot.multipass_dot_general(x, qx)

  def test_residual_decomposition_n_passes_progression(self):
    """Verifies that each residual pass reduces error and increases SNR."""
    key = self.rng
    x = jax.random.normal(key, (8, 64), dtype=jnp.float32)
    how = dot_general.get_how_to_quantize(
        dimension_numbers=(((1,), (0,)), ((), ())),
        ndims=(2, 2),
        for_lhs=True,
        qtype=jnp.float8_e4m3fn,
        tile_size=None,
    )
    n_passes = 4
    passes = multipass_dot.residual_decompose(x, how, n_passes=n_passes)
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
      (jnp.float8_e4m3fn, None),
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
    passes = multipass_dot.residual_decompose(exact_vals, how, n_passes=2)
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
      'three_pass_fp8',
      'four_pass_fp8',
      'two_pass_lhs_fp8',
      'two_pass_rhs_fp8',
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
        qtype=jnp.float8_e4m3fn,
        tile_size=None,
    )
    rhs_how = dot_general.get_how_to_quantize(
        dimension_numbers=dnums,
        ndims=(2, 2),
        for_lhs=False,
        qtype=jnp.float8_e4m3fn,
        tile_size=None,
    )

    actual = multipass_dot.multipass_dot_general(
        lhs,
        rhs,
        dimension_numbers=dnums,
        multipass_mode=mode,
    )

    # Compute manual reference terms from component passes
    a_passes = multipass_dot.residual_decompose(lhs, lhs_how, n_passes=2)
    b_passes = multipass_dot.residual_decompose(rhs, rhs_how, n_passes=2)

    c00 = dot_general.dot_general(a_passes[0], b_passes[0], dnums)
    c01 = dot_general.dot_general(a_passes[0], b_passes[1], dnums)
    c10 = dot_general.dot_general(a_passes[1], b_passes[0], dnums)
    c11 = dot_general.dot_general(a_passes[1], b_passes[1], dnums)

    if mode == 'three_pass_fp8':
      expected = c00 + c01 + c10
    elif mode == 'four_pass_fp8':
      expected = c00 + c01 + c10 + c11
    elif mode == 'two_pass_lhs_fp8':
      expected = c00 + c10
    elif mode == 'two_pass_rhs_fp8':
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
        qtype=jnp.float8_e4m3fn,
        tile_size=None,
    )
    rhs_how = dot_general.get_how_to_quantize(
        dimension_numbers=dnums,
        ndims=(2, 2),
        for_lhs=False,
        qtype=jnp.float8_e4m3fn,
        tile_size=None,
    )

    res_tri = multipass_dot.multipass_dot_general(
        lhs,
        rhs,
        dnums,
        multipass_mode='three_pass_fp8',
    )
    res_full = multipass_dot.multipass_dot_general(
        lhs,
        rhs,
        dnums,
        multipass_mode='four_pass_fp8',
    )

    a_passes = multipass_dot.residual_decompose(lhs, lhs_how, n_passes=2)
    b_passes = multipass_dot.residual_decompose(rhs, rhs_how, n_passes=2)
    a1_b1 = dot_general.dot_general(a_passes[1], b_passes[1], dnums)

    # Difference must match A1 @ B1 to floating point precision
    diff = res_full - res_tri
    np.testing.assert_allclose(diff, a1_b1, rtol=1e-5, atol=1e-5)

    # Second order term magnitude must be ~10^-4 or smaller relative to full
    rel_second_order = jnp.linalg.norm(a1_b1) / jnp.linalg.norm(res_full)
    self.assertLess(float(rel_second_order), 1e-3)

  @parameterized.named_parameters(
      (
          'batched_matmul_3d',
          (2, 8, 32),
          (2, 32, 16),
          (((2,), (1,)), ((0,), (0,))),
          (2, 8, 16),
      ),
      (
          'multi_axis_contraction_weight_grad',
          (2, 4, 8),
          (2, 4, 16),
          (((0, 1), (0, 1)), ((), ())),
          (8, 16),
      ),
      (
          'high_rank_attention_proj_4d',
          (2, 4, 8, 16),
          (16, 32),
          (((3,), (0,)), ((), ())),
          (2, 4, 8, 32),
      ),
  )
  def test_multipass_dot_general_arbitrary_dimensions(
      self, lhs_shape, rhs_shape, dnums, expected_shape
  ):
    """Verifies multipass_dot_general works with arbitrary dimension numbers."""
    k1, k2 = jax.random.split(self.rng)
    lhs = jax.random.normal(k1, lhs_shape, dtype=jnp.float32)
    rhs = jax.random.normal(k2, rhs_shape, dtype=jnp.float32)

    # 1. Direct call to multipass_dot_general
    out = multipass_dot.multipass_dot_general(
        lhs, rhs, dimension_numbers=dnums, multipass_mode='three_pass_fp8'
    )
    self.assertEqual(out.shape, expected_shape)

    # 2. Dispatch via dot_general.dot_general with multipass_mode
    out_dg = dot_general.dot_general(
        lhs, rhs, dimension_numbers=dnums, multipass_mode='three_pass_fp8'
    )
    self.assertEqual(out_dg.shape, expected_shape)
    np.testing.assert_allclose(out, out_dg, rtol=1e-5, atol=1e-5)

    # 3. Verify high numerical accuracy against unquantized FP32 reference
    ref = jax.lax.dot_general(lhs, rhs, dimension_numbers=dnums)
    self.assertEqual(ref.shape, expected_shape)
    snr = compute_snr_db(ref, out)
    self.assertGreater(snr, 35.0)

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
        qtype=jnp.float8_e4m3fn,
        tile_size=None,
    )
    how_r = dot_general.get_how_to_quantize(
        dimension_numbers=dnums,
        ndims=(2, 2),
        for_lhs=False,
        qtype=jnp.float8_e4m3fn,
        tile_size=None,
    )
    res_1p = lax.dot_general(
        qarray.dequantize(qarray.quantize(lhs, how_l)),
        qarray.dequantize(qarray.quantize(rhs, how_r)),
        dnums,
    )
    snr_1p = compute_snr_db(true_res, res_1p)

    # 2-pass (two_pass_lhs_fp8)
    res_2p = multipass_dot.multipass_dot_general(
        lhs,
        rhs,
        dnums,
        multipass_mode='two_pass_lhs_fp8',
    )
    snr_2p = compute_snr_db(true_res, res_2p)

    # 3-pass (three_pass_fp8)
    res_3p = multipass_dot.multipass_dot_general(
        lhs,
        rhs,
        dnums,
        multipass_mode='three_pass_fp8',
    )
    snr_3p = compute_snr_db(true_res, res_3p)

    # 4-pass (four_pass_fp8)
    res_4p = multipass_dot.multipass_dot_general(
        lhs,
        rhs,
        dnums,
        multipass_mode='four_pass_fp8',
    )
    snr_4p = compute_snr_db(true_res, res_4p)

    # Strict SNR progression: 1p < 2p < 3p <= 4p
    self.assertGreater(snr_2p, snr_1p)
    self.assertGreater(snr_3p, snr_2p)
    self.assertGreaterEqual(snr_4p, snr_3p - 0.2)
    # Triangular delivers substantial gain over 1-pass (>20dB CPU, >17dB TPU).
    min_gain = 17.0 if is_tpu() else 20.0
    self.assertGreater(snr_3p - snr_1p, min_gain)

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

    res = multipass_dot.multipass_dot_general(
        lhs,
        rhs,
        dimension_numbers=dnums,
        multipass_mode='three_pass_fp8',
        tile_size=None,
    )

    self.assertEqual(res.shape, expected_shape)
    self.assertFalse(jnp.isnan(res).any())

    snr = compute_snr_db(true_res, res)
    self.assertGreater(snr, 45.0)

  @parameterized.parameters(
      'three_pass_fp8',
      'four_pass_fp8',
      'two_pass_lhs_fp8',
      'two_pass_rhs_fp8',
  )
  def test_dot_general_qt_multipass_fwd(self, mode):
    k1, k2 = jax.random.split(self.rng)
    lhs = jax.random.normal(k1, (4, 32), dtype=jnp.float32)
    rhs = jax.random.normal(k2, (32, 4), dtype=jnp.float32)
    dnums = (((1,), (0,)), ((), ()))

    config = dot_general_qt.DotGeneralQtConfig(
        multipass_mode=mode,
    )
    res = dot_general_qt.dot_general_qt(lhs, rhs, dnums, config)
    self.assertEqual(res.shape, (4, 4))
    self.assertFalse(jnp.isnan(res).any())

  @parameterized.parameters(
      'three_pass_fp8',
      'four_pass_fp8',
      'two_pass_lhs_fp8',
      'two_pass_rhs_fp8',
  )
  def test_dot_general_multipass_mode_dispatch(self, mode):
    """Verifies that dot_general.dot_general correctly dispatches multipass_mode."""
    k1, k2 = jax.random.split(self.rng)
    lhs = jax.random.normal(k1, (8, 32), dtype=jnp.float32)
    rhs = jax.random.normal(k2, (32, 8), dtype=jnp.float32)
    dnums = (((1,), (0,)), ((), ()))

    res_dot_general = dot_general.dot_general(
        lhs, rhs, dnums, multipass_mode=mode
    )
    res_multipass = multipass_dot.multipass_dot_general(
        lhs, rhs, dnums, multipass_mode=mode
    )

    np.testing.assert_array_equal(res_dot_general, res_multipass)

  def test_fp8_accumulation_path(self):
    """Verifies standard FP8 preserves hardware FP8 dot_general."""
    dnums = (((1,), (0,)), ((), ()))
    x = jnp.ones((4, 256), dtype=jnp.bfloat16)
    y = jnp.ones((256, 4), dtype=jnp.bfloat16)

    # 1. Multi-pass with standard FP8 (default: jnp.float8_e4m3fn,
    # tile_size=None): Tracing with jax.make_jaxpr verifies the dot_general
    # primitive receives float8_e4m3fn inputs directly.
    jaxpr_mp_fp8 = jax.make_jaxpr(
        lambda a, b: multipass_dot.multipass_dot_general(
            a,
            b,
            dnums,
            multipass_mode='three_pass_fp8',
        )
    )(x, y)
    fp8_dot_eqns = [
        eqn for eqn in jaxpr_mp_fp8.eqns if eqn.primitive.name == 'dot_general'
    ]
    self.assertNotEmpty(fp8_dot_eqns)
    for eqn in fp8_dot_eqns:
      self.assertEqual(eqn.invars[0].aval.dtype, jnp.float8_e4m3fn)
      self.assertEqual(eqn.invars[1].aval.dtype, jnp.float8_e4m3fn)

    # 2. Multi-pass with FP8 and subchannel tile_size=256 (>= 128 threshold):
    # Also preserves hardware FP8 dot_general.
    jaxpr_mp_fp8_subchan = jax.make_jaxpr(
        lambda a, b: multipass_dot.multipass_dot_general(
            a,
            b,
            dnums,
            multipass_mode='three_pass_fp8',
            tile_size=256,
        )
    )(x, y)
    subchan_dot_eqns = [
        eqn
        for eqn in jaxpr_mp_fp8_subchan.eqns
        if eqn.primitive.name == 'dot_general'
    ]
    self.assertNotEmpty(subchan_dot_eqns)
    for eqn in subchan_dot_eqns:
      self.assertEqual(eqn.invars[0].aval.dtype, jnp.float8_e4m3fn)
      self.assertEqual(eqn.invars[1].aval.dtype, jnp.float8_e4m3fn)

  def test_dot_general_qt_multipass_calibrates_stats(self):
    """Verifies that lhs/rhs_collect_quant_stat are invoked when multipass_mode is set."""
    k1, k2 = jax.random.split(self.rng)
    lhs = jax.random.normal(k1, (4, 32), dtype=jnp.float32)
    rhs = jax.random.normal(k2, (32, 4), dtype=jnp.float32)
    dnums = (((1,), (0,)), ((), ()))

    lhs_stat_called = []
    rhs_stat_called = []

    def lhs_stat_hook(calib):
      lhs_stat_called.append(calib)
      return calib

    def rhs_stat_hook(calib):
      rhs_stat_called.append(calib)
      return calib

    config = dot_general_qt.DotGeneralQtConfig(
        lhs_qtype=jnp.float8_e4m3fn,
        rhs_qtype=jnp.float8_e4m3fn,
        multipass_mode='three_pass_fp8',
        lhs_collect_quant_stat=lhs_stat_hook,
        rhs_collect_quant_stat=rhs_stat_hook,
    )
    res = dot_general_qt.dot_general_qt(lhs, rhs, dnums, config)
    self.assertEqual(res.shape, (4, 4))
    self.assertLen(lhs_stat_called, 1)
    self.assertLen(rhs_stat_called, 1)

  def test_dot_general_qt_fwd_multipass_bwd_singlepass_quantized_residual(self):
    """Verifies backward pass correctly receives quantized QArray residual when dlhs_multipass_mode is None."""
    k1, k2, k3 = jax.random.split(self.rng, 3)
    lhs = jax.random.normal(k1, (16, 32), dtype=jnp.float32)
    rhs = jax.random.normal(k2, (32, 16), dtype=jnp.float32)
    target = jax.random.normal(k3, (16, 16), dtype=jnp.float32)
    dnums = (((1,), (0,)), ((), ()))

    # Fwd has multipass_mode='three_pass_fp8', but backward has single pass
    # (dlhs/drhs_multipass_mode=None) and use_original_residuals=False.
    cfg = dot_general_qt.DotGeneralQtConfig(
        multipass_mode='three_pass_fp8',
        use_original_residuals=False,
        dlhs_grad_qtype=jnp.float8_e4m3fn,
        drhs_grad_qtype=jnp.float8_e4m3fn,
        dlhs_residual_qtype=jnp.float8_e4m3fn,
        drhs_residual_qtype=jnp.float8_e4m3fn,
    )

    def loss_fn(a, b):
      out = dot_general_qt.dot_general_qt(a, b, dnums, cfg)
      return 0.5 * jnp.sum((out - target) ** 2)

    ga, gb = jax.grad(loss_fn, argnums=(0, 1))(lhs, rhs)
    self.assertFalse(jnp.isnan(ga).any())
    self.assertFalse(jnp.isnan(gb).any())
    self.assertEqual(ga.shape, lhs.shape)
    self.assertEqual(gb.shape, rhs.shape)

  def test_gradient_accuracy_and_convergence_vs_float32(self):
    """Verifies that multi-pass backward gradients converge to float32."""
    k1, k2, k3 = jax.random.split(self.rng, 3)
    lhs = jax.random.normal(k1, (16, 32), dtype=jnp.float32)
    rhs = jax.random.normal(k2, (32, 16), dtype=jnp.float32)
    target = jax.random.normal(k3, (16, 16), dtype=jnp.float32)
    dnums = (((1,), (0,)), ((), ()))

    # Loss function with fp32 non-trivial incoming gradient
    def true_loss(a, b):
      out = lax.dot_general(a, b, dnums)
      return 0.5 * jnp.sum((out - target) ** 2)

    true_ga, true_gb = jax.grad(true_loss, argnums=(0, 1))(lhs, rhs)

    # 1. Single pass fwd & bwd config (quantizing grad and residual in bwd)
    cfg_1p = dot_general_qt.DotGeneralQtConfig(
        lhs_qtype=jnp.float8_e4m3fn,
        rhs_qtype=jnp.float8_e4m3fn,
        dlhs_grad_qtype=jnp.float8_e4m3fn,
        drhs_grad_qtype=jnp.float8_e4m3fn,
        dlhs_residual_qtype=jnp.float8_e4m3fn,
        drhs_residual_qtype=jnp.float8_e4m3fn,
    )

    def loss_1p(a, b):
      out = dot_general_qt.dot_general_qt(a, b, dnums, cfg_1p)
      return 0.5 * jnp.sum((out - target) ** 2)

    ga_1p, gb_1p = jax.grad(loss_1p, argnums=(0, 1))(lhs, rhs)
    snr_ga_1p = compute_snr_db(true_ga, ga_1p)
    snr_gb_1p = compute_snr_db(true_gb, gb_1p)

    # 2. Multi-pass triangular fwd & bwd config
    cfg_tri = dot_general_qt.DotGeneralQtConfig(
        multipass_mode='three_pass_fp8',
        dlhs_grad_qtype=jnp.float8_e4m3fn,
        drhs_grad_qtype=jnp.float8_e4m3fn,
        dlhs_residual_qtype=jnp.float8_e4m3fn,
        drhs_residual_qtype=jnp.float8_e4m3fn,
        dlhs_multipass_mode='three_pass_fp8',
        drhs_multipass_mode='three_pass_fp8',
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
      'three_pass_fp8',
      'four_pass_fp8',
  )
  def test_numerical_edge_cases(self, mode):
    """Verifies behavior on zero matrices, asymmetric zeros, and wide dynamic range."""
    dnums = (((1,), (0,)), ((), ()))

    # Zero matrices
    zeros_l = jnp.zeros((4, 16), dtype=jnp.float32)
    zeros_r = jnp.zeros((16, 4), dtype=jnp.float32)
    res_zero = multipass_dot.multipass_dot_general(
        zeros_l, zeros_r, dnums, mode=mode
    )
    np.testing.assert_array_equal(res_zero, 0.0)
    self.assertFalse(jnp.isnan(res_zero).any())

    # One side zero
    k1 = self.rng
    normal_l = jax.random.normal(k1, (4, 16), dtype=jnp.float32)
    res_one_zero = multipass_dot.multipass_dot_general(
        normal_l, zeros_r, dnums, mode=mode
    )
    np.testing.assert_array_equal(res_one_zero, 0.0)
    self.assertFalse(jnp.isnan(res_one_zero).any())

    # Extreme scaling (1e-5 to 1e5)
    k2, k3 = jax.random.split(self.rng)
    small_l = jax.random.normal(k2, (4, 16), dtype=jnp.float32) * 1e-4
    large_r = jax.random.normal(k3, (16, 4), dtype=jnp.float32) * 1e4
    res_scaled = multipass_dot.multipass_dot_general(
        small_l, large_r, dnums, mode=mode
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

    res = multipass_dot.multipass_dot_general(
        lhs, rhs, dnums, multipass_mode='three_pass_fp8'
    )

    # Transposition symmetry: (A @ B).T == B.T @ A.T
    res_t = multipass_dot.multipass_dot_general(
        rhs.T, lhs.T, dnums, multipass_mode='three_pass_fp8'
    )
    np.testing.assert_allclose(res.T, res_t, rtol=1e-5, atol=1e-5)

    # Power-of-2 scaling linearity: multipass(2 * A, B) == 2 * multipass(A, B)
    res_scaled = multipass_dot.multipass_dot_general(
        lhs * 2.0, rhs, dnums, multipass_mode='three_pass_fp8'
    )
    np.testing.assert_allclose(res * 2.0, res_scaled, rtol=1e-5, atol=1e-5)

  def test_multipass_jit_compatibility(self):
    """Verifies that multipass_dot compiles under jax.jit and matches eager."""
    k1, k2 = jax.random.split(self.rng)
    lhs = jax.random.normal(k1, (8, 32), dtype=jnp.float32)
    rhs = jax.random.normal(k2, (32, 8), dtype=jnp.float32)
    dnums = (((1,), (0,)), ((), ()))

    eager_res = multipass_dot.multipass_dot_general(
        lhs, rhs, dnums, multipass_mode='three_pass_fp8'
    )

    jitted_dot = jax.jit(
        multipass_dot.multipass_dot_general,
        static_argnames=(
            'dimension_numbers',
            'multipass_mode',
            'tile_size',
        ),
    )
    jit_res = jitted_dot(lhs, rhs, dnums, multipass_mode='three_pass_fp8')

    np.testing.assert_allclose(eager_res, jit_res, rtol=1e-5, atol=1e-5)

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
        qtype=jnp.float8_e4m3fn,
        tile_size=None,
    )
    how_k = dot_general.get_how_to_quantize(
        dimension_numbers=qk_dnums,
        ndims=(4, 4),
        for_lhs=False,
        qtype=jnp.float8_e4m3fn,
        tile_size=None,
    )
    logits_1p = lax.dot_general(
        qarray.dequantize(qarray.quantize(query, how_q)),
        qarray.dequantize(qarray.quantize(key, how_k)),
        qk_dnums,
    )
    snr_1p = compute_snr_db(true_logits, logits_1p)

    # Multi-pass triangular (3-pass)
    logits_tri = multipass_dot.multipass_dot_general(
        query,
        key,
        dimension_numbers=qk_dnums,
        multipass_mode='three_pass_fp8',
    )
    snr_tri = compute_snr_db(true_logits, logits_tri)

    self.assertEqual(logits_tri.shape, (2, 4, 16, 16))
    tri_thresh = 48.0 if is_tpu() else 50.0
    self.assertGreater(snr_tri, tri_thresh)
    min_gain = 18.0 if is_tpu() else 20.0
    self.assertGreater(snr_tri - snr_1p, min_gain)

    # Attn_weights @ V: contracting seq_len (axis 3 of weights, axis 2 of V)
    weights = jax.nn.softmax(logits_tri, axis=-1)
    av_dnums = (((3,), (2,)), ((0, 1), (0, 1)))
    true_context = lax.dot_general(weights, value, av_dnums)

    context_tri = multipass_dot.multipass_dot_general(
        weights,
        value,
        dimension_numbers=av_dnums,
        multipass_mode='three_pass_fp8',
        tile_size=None,
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

    # FP32 standard normal Gaussian inputs N(0, 1)
    lhs = jax.random.normal(k1, shape_l, dtype=jnp.float32)
    rhs = jax.random.normal(k2, shape_r, dtype=jnp.float32)
    ref_f32 = lax.dot_general(lhs, rhs, dnums)

    # 1. Single-Pass FP8 (1 pass) - target ~28.41 dB
    how_l_fp8 = dot_general.get_how_to_quantize(
        dimension_numbers=dnums,
        ndims=(2, 2),
        for_lhs=True,
        qtype=jnp.float8_e4m3fn,
        tile_size=None,
    )
    how_r_fp8 = dot_general.get_how_to_quantize(
        dimension_numbers=dnums,
        ndims=(2, 2),
        for_lhs=False,
        qtype=jnp.float8_e4m3fn,
        tile_size=None,
    )
    res_fp8_1p = lax.dot_general(
        qarray.dequantize(qarray.quantize(lhs, how_l_fp8)),
        qarray.dequantize(qarray.quantize(rhs, how_r_fp8)),
        dnums,
    )
    snr_fp8_1p = float(compute_snr_db(ref_f32, res_fp8_1p))
    err_fp8_1p = float(compute_relative_error(ref_f32, res_fp8_1p))

    # 2. Asymmetric FP8 2-Pass (2 passes) - target ~31.56 dB
    res_fp8_2p = multipass_dot.multipass_dot_general(
        lhs,
        rhs,
        dnums,
        multipass_mode='two_pass_lhs_fp8',
    )
    snr_fp8_2p = float(compute_snr_db(ref_f32, res_fp8_2p))
    err_fp8_2p = float(compute_relative_error(ref_f32, res_fp8_2p))

    # 3. Multi-Pass FP8 Triangular (3 passes) - target ~54.84 dB
    res_fp8_tri = multipass_dot.multipass_dot_general(
        lhs,
        rhs,
        dnums,
        multipass_mode='three_pass_fp8',
    )
    snr_fp8_tri = float(compute_snr_db(ref_f32, res_fp8_tri))
    err_fp8_tri = float(compute_relative_error(ref_f32, res_fp8_tri))

    # 4. Multi-Pass FP8 Full Cross (4 passes) - target ~55.59 dB
    res_fp8_full = multipass_dot.multipass_dot_general(
        lhs,
        rhs,
        dnums,
        multipass_mode='four_pass_fp8',
    )
    snr_fp8_full = float(compute_snr_db(ref_f32, res_fp8_full))
    err_fp8_full = float(compute_relative_error(ref_f32, res_fp8_full))

    # Also evaluate with bfloat16 accumulation (matching zfc_emulation_utils)
    res_fp8_tri_bf16 = multipass_dot.multipass_dot_general(
        lhs,
        rhs,
        dnums,
        multipass_mode='three_pass_fp8',
        preferred_element_type=jnp.bfloat16,
    )
    snr_fp8_tri_bf16 = float(compute_snr_db(ref_f32, res_fp8_tri_bf16))
    err_fp8_tri_bf16 = float(compute_relative_error(ref_f32, res_fp8_tri_bf16))

    print('=== SQNR BENCHMARKS ===')
    print(f'FP8 1-pass: SNR={snr_fp8_1p:.2f} dB, err={err_fp8_1p:.4f}')
    print(f'FP8 2-pass: SNR={snr_fp8_2p:.2f} dB, err={err_fp8_2p:.4f}')
    print(f'FP8 3-pass (tri): SNR={snr_fp8_tri:.2f} dB, err={err_fp8_tri:.4f}')
    print(
        f'FP8 4-pass (full): SNR={snr_fp8_full:.2f} dB, err={err_fp8_full:.4f}'
    )
    print(
        f'FP8 3-pass (bf16): SNR={snr_fp8_tri_bf16:.2f} dB,'
        f' err={err_fp8_tri_bf16:.4f}'
    )

    # Reference values aligned with zfc_emulation_utils/README.md Table 2.1
    # 1-pass FP8: ~28.4 dB, relative error ~0.038
    self.assertAlmostEqual(snr_fp8_1p, 28.47, delta=1.5)
    self.assertAlmostEqual(err_fp8_1p, 0.0377, delta=0.01)

    # 2-pass Asymmetric FP8: ~31.5 dB (+3.1 dB over 1-pass)
    self.assertAlmostEqual(snr_fp8_2p, 31.54, delta=1.5)
    self.assertAlmostEqual(err_fp8_2p, 0.0265, delta=0.01)
    self.assertGreater(snr_fp8_2p, snr_fp8_1p + 2.0)

    # 3-pass Triangular FP8: ~58.9 dB (CPU) / ~46.1 dB (TPU v6e)
    tri_target = 46.11 if is_tpu() else 58.90
    err_tri_target = 0.0050 if is_tpu() else 0.0011
    tri_bf16_target = 44.47 if is_tpu() else 50.22
    err_tri_bf16_target = 0.0060 if is_tpu() else 0.0031
    self.assertAlmostEqual(snr_fp8_tri, tri_target, delta=1.5)
    self.assertAlmostEqual(snr_fp8_tri_bf16, tri_bf16_target, delta=1.5)
    self.assertAlmostEqual(err_fp8_tri, err_tri_target, delta=0.002)
    self.assertAlmostEqual(err_fp8_tri_bf16, err_tri_bf16_target, delta=0.002)

    # 4-pass Full Cross FP8: ~61.0 dB (CPU) / ~46.2 dB (TPU v6e)
    full_target = 46.19 if is_tpu() else 61.02
    err_full_target = 0.0049 if is_tpu() else 0.0009
    self.assertAlmostEqual(snr_fp8_full, full_target, delta=1.5)
    self.assertAlmostEqual(err_fp8_full, err_full_target, delta=0.002)

    # Strict SNR progression hierarchy across passes
    self.assertGreater(snr_fp8_2p, snr_fp8_1p)
    self.assertGreater(snr_fp8_tri, snr_fp8_2p)
    self.assertGreaterEqual(snr_fp8_full, snr_fp8_tri - 0.2)

  def test_qtype_specified_with_multipass_mode_raises_error(self):
    """Verifies that passing lhs_qtype or rhs_qtype with multipass_mode raises ValueError."""
    lhs = jnp.ones((4, 4), dtype=jnp.float32)
    rhs = jnp.ones((4, 4), dtype=jnp.float32)
    with self.assertRaisesRegex(ValueError, 'must not be specified'):
      multipass_dot.multipass_dot_general(
          lhs, rhs, multipass_mode='three_pass_fp8', lhs_qtype=jnp.float8_e4m3fn
      )
    with self.assertRaisesRegex(ValueError, 'must not be specified'):
      multipass_dot.multipass_dot_general(
          lhs, rhs, multipass_mode='three_pass_fp8', rhs_qtype=jnp.float8_e4m3fn
      )

  def test_how_specified_with_multipass_mode_raises_error(self):
    """Verifies that passing lhs_how or rhs_how with multipass_mode raises ValueError."""
    lhs = jnp.ones((4, 4), dtype=jnp.float32)
    rhs = jnp.ones((4, 4), dtype=jnp.float32)
    how = dot_general.get_how_to_quantize(
        dimension_numbers=(((1,), (0,)), ((), ())),
        ndims=(2, 2),
        for_lhs=True,
        qtype=jnp.float8_e4m3fn,
        tile_size=None,
    )
    with self.assertRaisesRegex(ValueError, 'must not be specified'):
      multipass_dot.multipass_dot_general(
          lhs, rhs, multipass_mode='three_pass_fp8', lhs_how=how
      )
    with self.assertRaisesRegex(ValueError, 'must not be specified'):
      multipass_dot.multipass_dot_general(
          lhs, rhs, multipass_mode='three_pass_fp8', rhs_how=how
      )

  def test_fp8_multipass_graph_mechanics_and_gemm_ops(self):
    """Verifies low-level graph mechanics and number of GEMM ops for FP8 modes."""
    lhs = jnp.ones((16, 32), dtype=jnp.float32)
    rhs = jnp.ones((32, 16), dtype=jnp.float32)
    dnums = (((1,), (0,)), ((), ()))

    expected_gemm_counts = {
        'two_pass_lhs_fp8': 2,
        'two_pass_rhs_fp8': 2,
        'three_pass_fp8': 3,
        'four_pass_fp8': 4,
    }

    for mode, expected_count in expected_gemm_counts.items():
      fn = lambda x, y, m=mode: multipass_dot.multipass_dot_general(
          x, y, dnums, multipass_mode=m, tile_size=None
      )
      gemms, cast_dtypes = get_graph_gemms_and_casts(fn, lhs, rhs)
      self.assertLen(gemms, expected_count)
      for g in gemms:
        in_dtypes = [getattr(v.aval, 'dtype', None) for v in g.invars[:2]]
        self.assertEqual(in_dtypes, [jnp.float8_e4m3fn, jnp.float8_e4m3fn])
      self.assertIn(jnp.dtype(jnp.float8_e4m3fn), cast_dtypes)

  def test_preferred_element_type_graph_gemm_ops(self):
    """Verifies that preferred_element_type propagates to GEMMs in the graph."""
    lhs = jnp.ones((16, 32), dtype=jnp.float32)
    rhs = jnp.ones((32, 16), dtype=jnp.float32)
    dnums = (((1,), (0,)), ((), ()))

    fn = lambda x, y: multipass_dot.multipass_dot_general(
        x,
        y,
        dnums,
        multipass_mode='three_pass_fp8',
        tile_size=None,
        preferred_element_type=jnp.bfloat16,
    )
    gemms, _ = get_graph_gemms_and_casts(fn, lhs, rhs)
    self.assertLen(gemms, 3)
    for g in gemms:
      self.assertEqual(g.params.get('preferred_element_type'), jnp.bfloat16)
      in_dtypes = [getattr(v.aval, 'dtype', None) for v in g.invars[:2]]
      self.assertEqual(in_dtypes, [jnp.float8_e4m3fn, jnp.float8_e4m3fn])

  def test_preferred_element_type_accumulation_modes(self):
    """Verifies FP8 across FP32 acc, BF16 rounding, and BF16 accumulator.

    Verifies the accumulation analysis where:
    - Unconstrained FP32 accumulation achieves ~58.9 dB (3-pass) / 61.0 dB
    (4-pass).
    - Output rounded to BF16 achieves ~54.0 dB (3-pass) / 55.6 dB (4-pass).
    - Internal BF16 accumulator (preferred_element_type=bfloat16) drops to ~50.2
    dB
      due to exponent alignment swamping of the 2^-4 residual passes.
    """
    k1, k2 = jax.random.split(self.rng)
    shape_l = (256, 512)
    shape_r = (512, 256)
    dnums = (((1,), (0,)), ((), ()))

    lhs = jax.random.normal(k1, shape_l, dtype=jnp.float32)
    rhs = jax.random.normal(k2, shape_r, dtype=jnp.float32)
    ref_f32 = lax.dot_general(lhs, rhs, dnums)

    how_l = dot_general.get_how_to_quantize(
        dimension_numbers=dnums,
        ndims=(2, 2),
        for_lhs=True,
        qtype=jnp.float8_e4m3fn,
        tile_size=None,
    )
    how_r = dot_general.get_how_to_quantize(
        dimension_numbers=dnums,
        ndims=(2, 2),
        for_lhs=False,
        qtype=jnp.float8_e4m3fn,
        tile_size=None,
    )

    # 1. Single-pass FP8 across accumulation regimes: all ~28.4 dB
    res_1p_f32 = lax.dot_general(
        qarray.dequantize(qarray.quantize(lhs, how_l)),
        qarray.dequantize(qarray.quantize(rhs, how_r)),
        dnums,
    )
    snr_1p_f32 = compute_snr_db(ref_f32, res_1p_f32)
    snr_1p_bf16_round = compute_snr_db(ref_f32, res_1p_f32.astype(jnp.bfloat16))
    self.assertAlmostEqual(snr_1p_f32, 28.44 if is_tpu() else 28.47, delta=0.15)
    self.assertAlmostEqual(
        snr_1p_bf16_round, 28.43 if is_tpu() else 28.47, delta=0.15
    )

    # 2. Asymmetric 2-pass FP8 (LHS): all ~31.5 dB
    res_2p_f32 = multipass_dot.multipass_dot_general(
        lhs,
        rhs,
        dnums,
        multipass_mode='two_pass_lhs_fp8',
    )
    snr_2p_f32 = compute_snr_db(ref_f32, res_2p_f32)
    snr_2p_bf16_round = compute_snr_db(ref_f32, res_2p_f32.astype(jnp.bfloat16))
    res_2p_bf16_acc = multipass_dot.multipass_dot_general(
        lhs,
        rhs,
        dnums,
        multipass_mode='two_pass_lhs_fp8',
        preferred_element_type=jnp.bfloat16,
    )
    snr_2p_bf16_acc = compute_snr_db(ref_f32, res_2p_bf16_acc)
    self.assertAlmostEqual(snr_2p_f32, 31.49 if is_tpu() else 31.54, delta=0.15)
    self.assertAlmostEqual(
        snr_2p_bf16_round, 31.47 if is_tpu() else 31.52, delta=0.15
    )
    self.assertAlmostEqual(
        snr_2p_bf16_acc, 31.45 if is_tpu() else 31.50, delta=0.15
    )

    # 3. Multi-pass 3-pass Triangular FP8:
    res_tri_f32 = multipass_dot.multipass_dot_general(
        lhs,
        rhs,
        dnums,
        multipass_mode='three_pass_fp8',
    )
    snr_tri_f32 = compute_snr_db(ref_f32, res_tri_f32)
    snr_tri_bf16_round = compute_snr_db(
        ref_f32, res_tri_f32.astype(jnp.bfloat16)
    )

    res_tri_bf16_acc = multipass_dot.multipass_dot_general(
        lhs,
        rhs,
        dnums,
        multipass_mode='three_pass_fp8',
        preferred_element_type=jnp.bfloat16,
    )
    snr_tri_bf16_acc = compute_snr_db(ref_f32, res_tri_bf16_acc)

    # Tight check across CPU and TPU
    self.assertAlmostEqual(snr_tri_f32, 46.11 if is_tpu() else 58.90, delta=0.5)
    self.assertAlmostEqual(
        snr_tri_bf16_round, 45.55 if is_tpu() else 53.97, delta=0.5
    )
    self.assertAlmostEqual(
        snr_tri_bf16_acc, 44.47 if is_tpu() else 50.22, delta=0.5
    )
    if not is_tpu():
      self.assertGreater(snr_tri_bf16_round, snr_tri_bf16_acc + 3.0)

    # 4. Multi-pass 4-pass Full Cross FP8:
    res_full_f32 = multipass_dot.multipass_dot_general(
        lhs,
        rhs,
        dnums,
        multipass_mode='four_pass_fp8',
    )
    snr_full_f32 = compute_snr_db(ref_f32, res_full_f32)
    snr_full_bf16_round = compute_snr_db(
        ref_f32, res_full_f32.astype(jnp.bfloat16)
    )

    res_full_bf16_acc = multipass_dot.multipass_dot_general(
        lhs,
        rhs,
        dnums,
        multipass_mode='four_pass_fp8',
        preferred_element_type=jnp.bfloat16,
    )
    snr_full_bf16_acc = compute_snr_db(ref_f32, res_full_bf16_acc)

    self.assertAlmostEqual(
        snr_full_f32, 46.19 if is_tpu() else 61.02, delta=0.5
    )
    self.assertAlmostEqual(
        snr_full_bf16_round, 45.61 if is_tpu() else 54.56, delta=0.5
    )
    self.assertAlmostEqual(
        snr_full_bf16_acc, 44.50 if is_tpu() else 50.28, delta=0.5
    )

  def test_ghostfish_tpu_v6e_multipass_fp8(self):
    """Verifies multi-pass FP8 execution on TPU v6e and CPU reference."""
    k1, k2 = jax.random.split(self.rng)
    shape_l = (128, 256)
    shape_r = (256, 128)
    dnums = (((1,), (0,)), ((), ()))

    lhs = jax.random.normal(k1, shape_l, dtype=jnp.bfloat16)
    rhs = jax.random.normal(k2, shape_r, dtype=jnp.bfloat16)
    ref = lax.dot_general(lhs, rhs, dnums)

    # 1. Pure FP8 (1-Pass) on TPU hardware accumulator
    how_l_pure = dot_general.get_how_to_quantize(
        dimension_numbers=dnums,
        ndims=(2, 2),
        for_lhs=True,
        qtype='float8_e4m3fn',
        tile_size=None,
    )
    how_r_pure = dot_general.get_how_to_quantize(
        dimension_numbers=dnums,
        ndims=(2, 2),
        for_lhs=False,
        qtype='float8_e4m3fn',
        tile_size=None,
    )
    q_lhs_pure = qarray.quantize(lhs, how_l_pure)
    q_rhs_pure = qarray.quantize(rhs, how_r_pure)
    res_pure_fp8 = jax.jit(
        dot_general.dot_general, static_argnames=('dimension_numbers',)
    )(q_lhs_pure, q_rhs_pure, dnums)
    snr_pure_fp8 = compute_snr_db(ref, res_pure_fp8)

    # 2. Test multi-pass modes under jax.jit on TPU
    jitted_dot = jax.jit(
        multipass_dot.multipass_dot_general,
        static_argnames=(
            'dimension_numbers',
            'multipass_mode',
            'tile_size',
        ),
    )

    # 1-pass, 2-pass, 3-pass triangular, and 4-pass full cross
    res_1p = jitted_dot(
        lhs, rhs, dnums, multipass_mode='two_pass_lhs_fp8', tile_size=None
    )
    res_2p = jitted_dot(
        lhs,
        rhs,
        dnums,
        multipass_mode='two_pass_lhs_fp8',
        tile_size=16,
    )
    res_tri = jitted_dot(
        lhs,
        rhs,
        dnums,
        multipass_mode='three_pass_fp8',
        tile_size=16,
    )
    res_full = jitted_dot(
        lhs,
        rhs,
        dnums,
        multipass_mode='four_pass_fp8',
        tile_size=16,
    )

    self.assertEqual(res_pure_fp8.shape, (128, 128))
    self.assertEqual(res_tri.shape, (128, 128))
    self.assertEqual(res_full.shape, (128, 128))
    self.assertTrue(bool(jnp.all(jnp.isfinite(res_pure_fp8))))
    self.assertTrue(bool(jnp.all(jnp.isfinite(res_tri))))
    self.assertTrue(bool(jnp.all(jnp.isfinite(res_full))))

    snr_1p = compute_snr_db(ref, res_1p)
    snr_2p = compute_snr_db(ref, res_2p)
    snr_tri = compute_snr_db(ref, res_tri)
    snr_full = compute_snr_db(ref, res_full)

    # On physical TPU v6e hardware with 8-way YOLO accumulator truncation:
    # Pure FP8 achieves ~28.4 dB, while 3-pass triangular achieves > 45.0 dB.
    self.assertGreater(snr_pure_fp8, 25.0)
    self.assertGreater(snr_1p, 25.0)
    self.assertGreater(snr_2p, 30.0)
    self.assertGreater(snr_tri, 45.0)
    self.assertGreaterEqual(snr_full, snr_tri - 0.5)

  def test_multipass_fp8_sqnr_benchmark_table(self):
    """Verifies SQNR and error on N(0, 1) inputs and outputs benchmark table.

    ### Multi-Pass FP8 SQNR Benchmark Table (512x512, Channel-wise / Tensor
    Scaling)

    | Strategy | GEMMs | Compute DType | Execution Target | Accumulation Mode |
    SQNR (dB) | Rel Error (%) |
    | :---| :---: | :---| :---| :---| :---: | :---: |
    | Baseline BF16 (Reference) | 1 | bfloat16 | TPU / CPU | Output Rounded to
    BF16 | 55.62 dB | 0.17% |
    | Pure FP8 (1-Pass) | 1 | float8_e4m3fn | CPU Reference | FP32 Accumulation
    (Rounded to BF16) | 28.43 dB | 3.79% |
    | Pure FP8 (1-Pass) | 1 | float8_e4m3fn | Physical TPU | Hardware
    Accumulator | 28.39 dB | 3.80% |
    | Triangular Multi-Pass FP8 | 3 | float8_e4m3fn | CPU Reference | FP32
    Accumulation (Rounded to BF16) | 54.84 dB | 0.18% |
    | Triangular Multi-Pass FP8 | 3 | float8_e4m3fn | Physical TPU | Hardware
    Accumulator | 47.48 dB | 0.42% |
    | Full Cross Multi-Pass FP8 | 4 | float8_e4m3fn | CPU Reference | FP32
    Accumulation (Rounded to BF16) | 55.59 dB | 0.17% |
    | Full Cross Multi-Pass FP8 | 4 | float8_e4m3fn | Physical TPU | Hardware
    Accumulator | 47.60 dB | 0.42% |
    """
    k1, k2 = jax.random.split(self.rng)
    shape_l = (256, 512)
    shape_r = (512, 256)
    dnums = (((1,), (0,)), ((), ()))

    # FP32 standard normal Gaussian inputs N(0, 1)
    lhs = jax.random.normal(k1, shape_l, dtype=jnp.float32)
    rhs = jax.random.normal(k2, shape_r, dtype=jnp.float32)
    ref_f32 = lax.dot_general(lhs, rhs, dnums)

    how_l_fp8 = dot_general.get_how_to_quantize(
        dimension_numbers=dnums,
        ndims=(2, 2),
        for_lhs=True,
        qtype=jnp.float8_e4m3fn,
        tile_size=None,
    )
    how_r_fp8 = dot_general.get_how_to_quantize(
        dimension_numbers=dnums,
        ndims=(2, 2),
        for_lhs=False,
        qtype=jnp.float8_e4m3fn,
        tile_size=None,
    )

    # Baseline BF16 (Reference rounded to BF16)
    ref_bf16 = ref_f32.astype(jnp.bfloat16)
    snr_bf16 = compute_snr_db(ref_f32, ref_bf16)
    err_bf16 = compute_relative_error(ref_f32, ref_bf16)

    # 1. Pure FP8 (1-Pass)
    res_1p_f32 = lax.dot_general(
        qarray.dequantize(qarray.quantize(lhs, how_l_fp8)),
        qarray.dequantize(qarray.quantize(rhs, how_r_fp8)),
        dnums,
    )
    snr_1p_f32 = compute_snr_db(ref_f32, res_1p_f32)
    err_1p_f32 = compute_relative_error(ref_f32, res_1p_f32)
    snr_1p_round_bf16 = compute_snr_db(ref_f32, res_1p_f32.astype(jnp.bfloat16))
    err_1p_round_bf16 = compute_relative_error(
        ref_f32, res_1p_f32.astype(jnp.bfloat16)
    )

    # Pure FP8 on Hardware Accumulator (physical TPU / fast path)
    how_l_pure = dot_general.get_how_to_quantize(
        dimension_numbers=dnums,
        ndims=(2, 2),
        for_lhs=True,
        qtype='float8_e4m3fn',
        tile_size=None,
    )
    how_r_pure = dot_general.get_how_to_quantize(
        dimension_numbers=dnums,
        ndims=(2, 2),
        for_lhs=False,
        qtype='float8_e4m3fn',
        tile_size=None,
    )
    res_1p_hw = dot_general.dot_general(
        qarray.quantize(lhs, how_l_pure),
        qarray.quantize(rhs, how_r_pure),
        dnums,
    )
    snr_1p_hw = compute_snr_db(ref_f32, res_1p_hw)
    err_1p_hw = compute_relative_error(ref_f32, res_1p_hw)

    # 2. Asymmetric 2-Pass FP8 (LHS High-Precision)
    res_2p_lhs_f32 = multipass_dot.multipass_dot_general(
        lhs,
        rhs,
        dnums,
        multipass_mode='two_pass_lhs_fp8',
    )
    snr_2p_lhs_f32 = compute_snr_db(ref_f32, res_2p_lhs_f32)
    err_2p_lhs_f32 = compute_relative_error(ref_f32, res_2p_lhs_f32)
    snr_2p_lhs_round_bf16 = compute_snr_db(
        ref_f32, res_2p_lhs_f32.astype(jnp.bfloat16)
    )
    err_2p_lhs_round_bf16 = compute_relative_error(
        ref_f32, res_2p_lhs_f32.astype(jnp.bfloat16)
    )

    # 3. Asymmetric 2-Pass FP8 (RHS High-Precision)
    res_2p_rhs_f32 = multipass_dot.multipass_dot_general(
        lhs,
        rhs,
        dnums,
        multipass_mode='two_pass_rhs_fp8',
    )
    snr_2p_rhs_f32 = compute_snr_db(ref_f32, res_2p_rhs_f32)
    err_2p_rhs_f32 = compute_relative_error(ref_f32, res_2p_rhs_f32)
    snr_2p_rhs_round_bf16 = compute_snr_db(
        ref_f32, res_2p_rhs_f32.astype(jnp.bfloat16)
    )
    err_2p_rhs_round_bf16 = compute_relative_error(
        ref_f32, res_2p_rhs_f32.astype(jnp.bfloat16)
    )

    # 4. Triangular Multi-Pass FP8 (3-Pass)
    res_tri_f32 = multipass_dot.multipass_dot_general(
        lhs,
        rhs,
        dnums,
        multipass_mode='three_pass_fp8',
    )
    snr_tri_f32 = compute_snr_db(ref_f32, res_tri_f32)
    err_tri_f32 = compute_relative_error(ref_f32, res_tri_f32)
    snr_tri_round_bf16 = compute_snr_db(
        ref_f32, res_tri_f32.astype(jnp.bfloat16)
    )
    err_tri_round_bf16 = compute_relative_error(
        ref_f32, res_tri_f32.astype(jnp.bfloat16)
    )

    res_tri_bf16_acc = multipass_dot.multipass_dot_general(
        lhs,
        rhs,
        dnums,
        multipass_mode='three_pass_fp8',
        preferred_element_type=jnp.bfloat16,
    )
    snr_tri_bf16_acc = compute_snr_db(ref_f32, res_tri_bf16_acc)
    err_tri_bf16_acc = compute_relative_error(ref_f32, res_tri_bf16_acc)

    # 5. Full Cross Multi-Pass FP8 (4-Pass)
    res_full_f32 = multipass_dot.multipass_dot_general(
        lhs,
        rhs,
        dnums,
        multipass_mode='four_pass_fp8',
    )
    snr_full_f32 = compute_snr_db(ref_f32, res_full_f32)
    err_full_f32 = compute_relative_error(ref_f32, res_full_f32)
    snr_full_round_bf16 = compute_snr_db(
        ref_f32, res_full_f32.astype(jnp.bfloat16)
    )
    err_full_round_bf16 = compute_relative_error(
        ref_f32, res_full_f32.astype(jnp.bfloat16)
    )

    res_full_bf16_acc = multipass_dot.multipass_dot_general(
        lhs,
        rhs,
        dnums,
        multipass_mode='four_pass_fp8',
        preferred_element_type=jnp.bfloat16,
    )
    snr_full_bf16_acc = compute_snr_db(ref_f32, res_full_bf16_acc)
    err_full_bf16_acc = compute_relative_error(ref_f32, res_full_bf16_acc)

    rows = [
        (
            'Baseline BF16 (Reference)',
            1,
            'Output Rounded to BF16',
            snr_bf16,
            err_bf16,
        ),
        ('Pure FP8 (1-Pass)', 1, 'FP32 Accumulation', snr_1p_f32, err_1p_f32),
        (
            'Pure FP8 (1-Pass)',
            1,
            'Output Rounded to BF16',
            snr_1p_round_bf16,
            err_1p_round_bf16,
        ),
        (
            'Pure FP8 (1-Pass)',
            1,
            'Hardware Accumulator' if is_tpu() else 'Hardware FP8 Path',
            snr_1p_hw,
            err_1p_hw,
        ),
        (
            'Asymmetric 2-Pass FP8 (LHS)',
            2,
            'FP32 Accumulation',
            snr_2p_lhs_f32,
            err_2p_lhs_f32,
        ),
        (
            'Asymmetric 2-Pass FP8 (LHS)',
            2,
            'Output Rounded to BF16',
            snr_2p_lhs_round_bf16,
            err_2p_lhs_round_bf16,
        ),
        (
            'Asymmetric 2-Pass FP8 (RHS)',
            2,
            'FP32 Accumulation',
            snr_2p_rhs_f32,
            err_2p_rhs_f32,
        ),
        (
            'Asymmetric 2-Pass FP8 (RHS)',
            2,
            'Output Rounded to BF16',
            snr_2p_rhs_round_bf16,
            err_2p_rhs_round_bf16,
        ),
        (
            'Triangular Multi-Pass FP8',
            3,
            'FP32 Accumulation',
            snr_tri_f32,
            err_tri_f32,
        ),
        (
            'Triangular Multi-Pass FP8',
            3,
            'Output Rounded to BF16',
            snr_tri_round_bf16,
            err_tri_round_bf16,
        ),
        (
            'Triangular Multi-Pass FP8',
            3,
            'BF16 Accumulator (swamped)',
            snr_tri_bf16_acc,
            err_tri_bf16_acc,
        ),
        (
            'Full Cross Multi-Pass FP8',
            4,
            'FP32 Accumulation',
            snr_full_f32,
            err_full_f32,
        ),
        (
            'Full Cross Multi-Pass FP8',
            4,
            'Output Rounded to BF16',
            snr_full_round_bf16,
            err_full_round_bf16,
        ),
        (
            'Full Cross Multi-Pass FP8',
            4,
            'BF16 Accumulator (swamped)',
            snr_full_bf16_acc,
            err_full_bf16_acc,
        ),
    ]

    print('\n=== MULTI-PASS FP8 SQNR BENCHMARK TABLE ===')
    print(
        '| Strategy | GEMMs | Accumulation Mode | SQNR (dB) | Rel Error (%) |'
    )
    print('| :---| :---: | :---| :---: | :---: |')
    for strat, gemms, acc_mode, snr, err in rows:
      print(
          f'| {strat} | {gemms} | {acc_mode} | {snr:.2f} dB |'
          f' {err * 100:.2f}% |'
      )
    print('===========================================\n')

    # Reference value checks
    self.assertAlmostEqual(snr_bf16, 55.66 if is_tpu() else 55.67, delta=0.15)
    self.assertAlmostEqual(snr_1p_f32, 28.44 if is_tpu() else 28.47, delta=0.15)
    self.assertAlmostEqual(
        snr_1p_round_bf16, 28.43 if is_tpu() else 28.47, delta=0.15
    )
    self.assertAlmostEqual(
        err_1p_f32, 0.0378 if is_tpu() else 0.0377, delta=0.002
    )
    self.assertAlmostEqual(snr_1p_hw, 28.46 if is_tpu() else 28.54, delta=0.15)

    self.assertAlmostEqual(
        snr_2p_lhs_f32, 31.49 if is_tpu() else 31.54, delta=0.15
    )
    self.assertAlmostEqual(
        snr_2p_lhs_round_bf16, 31.47 if is_tpu() else 31.52, delta=0.15
    )
    self.assertAlmostEqual(
        err_2p_lhs_f32, 0.0266 if is_tpu() else 0.0265, delta=0.002
    )
    self.assertAlmostEqual(
        snr_2p_rhs_f32, 31.43 if is_tpu() else 31.48, delta=0.15
    )
    self.assertAlmostEqual(
        snr_2p_rhs_round_bf16, 31.41 if is_tpu() else 31.46, delta=0.15
    )

    self.assertAlmostEqual(snr_tri_f32, 46.11 if is_tpu() else 58.90, delta=0.5)
    self.assertAlmostEqual(
        snr_tri_round_bf16, 45.55 if is_tpu() else 53.97, delta=0.5
    )
    self.assertAlmostEqual(
        snr_tri_bf16_acc, 44.47 if is_tpu() else 50.22, delta=0.5
    )

    self.assertAlmostEqual(
        snr_full_f32, 46.19 if is_tpu() else 61.02, delta=0.5
    )
    self.assertAlmostEqual(
        snr_full_round_bf16, 45.61 if is_tpu() else 54.56, delta=0.5
    )
    self.assertAlmostEqual(
        snr_full_bf16_acc, 44.50 if is_tpu() else 50.28, delta=0.5
    )

    # Monotonicity checks
    self.assertGreater(snr_2p_lhs_f32, snr_1p_f32)
    self.assertGreater(snr_tri_f32, snr_2p_lhs_f32)
    if not is_tpu():
      self.assertGreater(snr_tri_round_bf16, snr_tri_bf16_acc + 3.0)


if __name__ == '__main__':
  absltest.main()
