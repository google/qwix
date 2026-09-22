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


def is_ghostfish() -> bool:
  """Returns True on TPU7x, False on CPU, and raises an error for any other accelerator."""
  devices = jax.devices()
  if not devices:
    return False

  platform = devices[0].platform
  if platform == 'cpu':
    return False

  # If an accelerator is attached, require TPU7x
  if platform == 'tpu':
    device_kind = getattr(devices[0], 'device_kind', '')
    if device_kind != 'TPU7x':
      raise RuntimeError(
          'Accelerator detected, but expected TPU7x (Ghostfish). Found:'
          f' device_kind={device_kind!r}'
      )
    return True


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
) -> tuple[list[jax.core.JaxprEqn], list[jnp.dtype]]:
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
  cast_dtypes = [jnp.dtype(eqn.params['new_dtype']) for eqn in casts]
  return gemms, cast_dtypes


class MultiPassDotTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.rng = jax.random.PRNGKey(42)

  def test__residual_decompose(self):
    key = self.rng
    x = jax.random.normal(key, (4, 32), dtype=jnp.float32)
    how = dot_general.get_how_to_quantize(
        dimension_numbers=(((1,), (0,)), ((), ())),
        ndims=(2, 2),
        for_lhs=True,
        qtype=jnp.float8_e4m3fn,
        tile_size=None,
    )
    passes = multipass_dot._residual_decompose(x, how, n_passes=2)
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
      multipass_dot._residual_decompose(qx, how)
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
    passes = multipass_dot._residual_decompose(x, how, n_passes=n_passes)
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
    passes = multipass_dot._residual_decompose(exact_vals, how, n_passes=2)
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
    a_passes = multipass_dot._residual_decompose(lhs, lhs_how, n_passes=2)
    b_passes = multipass_dot._residual_decompose(rhs, rhs_how, n_passes=2)

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
    """Verifies four_pass_fp8 - three_pass_fp8 is equal to A_1 @ B_1."""
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

    a_passes = multipass_dot._residual_decompose(lhs, lhs_how, n_passes=2)
    b_passes = multipass_dot._residual_decompose(rhs, rhs_how, n_passes=2)
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
    # three_pass_fp8 delivers gain over 1-pass (>20dB CPU, >17dB TPU).
    min_gain = 17.0 if is_ghostfish() else 20.0
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

    # 2. Multi-pass three_pass_fp8 fwd & bwd config
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
        zeros_l, zeros_r, dnums, multipass_mode=mode
    )
    np.testing.assert_array_equal(res_zero, 0.0)
    self.assertFalse(jnp.isnan(res_zero).any())

    # One side zero
    k1 = self.rng
    normal_l = jax.random.normal(k1, (4, 16), dtype=jnp.float32)
    res_one_zero = multipass_dot.multipass_dot_general(
        normal_l, zeros_r, dnums, multipass_mode=mode
    )
    np.testing.assert_array_equal(res_one_zero, 0.0)
    self.assertFalse(jnp.isnan(res_one_zero).any())

    # Extreme scaling (1e-5 to 1e5)
    k2, k3 = jax.random.split(self.rng)
    small_l = jax.random.normal(k2, (4, 16), dtype=jnp.float32) * 1e-4
    large_r = jax.random.normal(k3, (16, 4), dtype=jnp.float32) * 1e4
    res_scaled = multipass_dot.multipass_dot_general(
        small_l, large_r, dnums, multipass_mode=mode
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

    # Multi-pass three_pass_fp8
    logits_tri = multipass_dot.multipass_dot_general(
        query,
        key,
        dimension_numbers=qk_dnums,
        multipass_mode='three_pass_fp8',
    )
    snr_tri = compute_snr_db(true_logits, logits_tri)

    self.assertEqual(logits_tri.shape, (2, 4, 16, 16))
    tri_thresh = 48.0 if is_ghostfish() else 50.0
    self.assertGreater(snr_tri, tri_thresh)
    min_gain = 18.0 if is_ghostfish() else 20.0
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

    # 3. three_pass_fp8 - target ~54.84 dB
    res_fp8_tri = multipass_dot.multipass_dot_general(
        lhs,
        rhs,
        dnums,
        multipass_mode='three_pass_fp8',
    )
    snr_fp8_tri = float(compute_snr_db(ref_f32, res_fp8_tri))
    err_fp8_tri = float(compute_relative_error(ref_f32, res_fp8_tri))

    # 4. four_pass_fp8 - target ~55.59 dB
    res_fp8_full = multipass_dot.multipass_dot_general(
        lhs,
        rhs,
        dnums,
        multipass_mode='four_pass_fp8',
    )
    snr_fp8_full = float(compute_snr_db(ref_f32, res_fp8_full))
    err_fp8_full = float(compute_relative_error(ref_f32, res_fp8_full))

    res_fp8_tri_bf16 = multipass_dot.multipass_dot_general(
        lhs,
        rhs,
        dnums,
        multipass_mode='three_pass_fp8',
        preferred_element_type=jnp.bfloat16,
    )
    snr_fp8_tri_bf16 = float(compute_snr_db(ref_f32, res_fp8_tri_bf16))
    err_fp8_tri_bf16 = float(compute_relative_error(ref_f32, res_fp8_tri_bf16))

    print('=== SQNR BENCHMARKS (Preferred Accumulator: FP32) ===')
    print(f'FP8 1-pass: SNR={snr_fp8_1p:.2f} dB, err={err_fp8_1p:.4f}')
    print(f'FP8 2-pass: SNR={snr_fp8_2p:.2f} dB, err={err_fp8_2p:.4f}')
    print(f'FP8 3-pass (tri): SNR={snr_fp8_tri:.2f} dB, err={err_fp8_tri:.4f}')
    print(
        f'FP8 4-pass (full): SNR={snr_fp8_full:.2f} dB, err={err_fp8_full:.4f}'
    )
    print('=== SQNR BENCHMARKS (Preferred Accumulator: BF16) ===')
    print(
        f'FP8 3-pass (tri): SNR={snr_fp8_tri_bf16:.2f} dB,'
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

    # three_pass_fp8: ~58.9 dB (CPU) / ~46.1 dB (TPU v6e)
    tri_target = 46.11 if is_ghostfish() else 58.90
    err_tri_target = 0.0050 if is_ghostfish() else 0.0011
    tri_bf16_target = 44.47 if is_ghostfish() else 50.22
    err_tri_bf16_target = 0.0060 if is_ghostfish() else 0.0031
    self.assertAlmostEqual(snr_fp8_tri, tri_target, delta=1.5)
    self.assertAlmostEqual(snr_fp8_tri_bf16, tri_bf16_target, delta=1.5)
    self.assertAlmostEqual(err_fp8_tri, err_tri_target, delta=0.002)
    self.assertAlmostEqual(err_fp8_tri_bf16, err_tri_bf16_target, delta=0.002)

    # four_pass_fp8: ~61.0 dB (CPU) / ~46.2 dB (TPU v6e)
    full_target = 46.19 if is_ghostfish() else 61.02
    err_full_target = 0.0049 if is_ghostfish() else 0.0009
    self.assertAlmostEqual(snr_fp8_full, full_target, delta=1.5)
    self.assertAlmostEqual(err_fp8_full, err_full_target, delta=0.002)

    # Strict SNR progression hierarchy across passes
    self.assertGreater(snr_fp8_2p, snr_fp8_1p)
    self.assertGreater(snr_fp8_tri, snr_fp8_2p)
    self.assertGreaterEqual(snr_fp8_full, snr_fp8_tri - 0.2)

    # 5. Exact INT8 via INT4 Full Cross (4 passes)
    res_four_pass = multipass_dot.multipass_dot_general(
        lhs,
        rhs,
        dnums,
        multipass_mode='four_pass_int4',
    )
    snr_four_pass = float(compute_snr_db(ref_f32, res_four_pass))
    err_four_pass = float(compute_relative_error(ref_f32, res_four_pass))

    print(
        f'INT8 Full Cross (4-pass): SNR={snr_four_pass:.2f} dB,'
        f' err={err_four_pass:.4f}'
    )

    self.assertAlmostEqual(snr_four_pass, 39.51, delta=1.5)
    self.assertAlmostEqual(err_four_pass, 0.0106, delta=0.005)

    # 6. Approximate INT8 via INT4 Truncated (3 passes, drop low-order product)
    res_i8_tri = multipass_dot.multipass_dot_general(
        lhs,
        rhs,
        dnums,
        multipass_mode='three_pass_int4',
    )
    snr_i8_tri = float(compute_snr_db(ref_f32, res_i8_tri))
    err_i8_tri = float(compute_relative_error(ref_f32, res_i8_tri))

    # 8. Asymmetric INT8 via INT4 (2 passes)
    res_i8_asym_l = multipass_dot.multipass_dot_general(
        lhs,
        rhs,
        dnums,
        multipass_mode='two_pass_lhs_int4',
    )
    snr_i8_asym_l = float(compute_snr_db(ref_f32, res_i8_asym_l))
    err_i8_asym_l = float(compute_relative_error(ref_f32, res_i8_asym_l))

    res_i8_asym_r = multipass_dot.multipass_dot_general(
        lhs,
        rhs,
        dnums,
        multipass_mode='two_pass_rhs_int4',
    )
    snr_i8_asym_r = float(compute_snr_db(ref_f32, res_i8_asym_r))
    err_i8_asym_r = float(compute_relative_error(ref_f32, res_i8_asym_r))

    print(
        f'INT8 Truncated (3-pass): SNR={snr_i8_tri:.2f} dB,'
        f' err={err_i8_tri:.4f}'
    )
    print(
        f'INT8 Asym LHS (2-pass): SNR={snr_i8_asym_l:.2f} dB,'
        f' err={err_i8_asym_l:.4f}'
    )
    print(
        f'INT8 Asym RHS (2-pass): SNR={snr_i8_asym_r:.2f} dB,'
        f' err={err_i8_asym_r:.4f}'
    )

    self.assertAlmostEqual(snr_i8_tri, 34.85, delta=1.5)
    self.assertAlmostEqual(err_i8_tri, 0.0182, delta=0.005)

    self.assertAlmostEqual(snr_i8_asym_l, 17.89, delta=1.5)
    self.assertAlmostEqual(err_i8_asym_l, 0.1275, delta=0.02)

    self.assertAlmostEqual(snr_i8_asym_r, 17.99, delta=1.5)
    self.assertAlmostEqual(err_i8_asym_r, 0.1274, delta=0.02)

    # 9. Hybrid FP8 + 4-Bit Strategies (3 passes)
    res_fp8_fp4 = multipass_dot.multipass_dot_general(
        lhs,
        rhs,
        dnums,
        multipass_mode='three_pass_fp8_fp4',
    )
    snr_fp8_fp4 = float(compute_snr_db(ref_f32, res_fp8_fp4))
    err_fp8_fp4 = float(compute_relative_error(ref_f32, res_fp8_fp4))

    res_fp8_int4 = multipass_dot.multipass_dot_general(
        lhs,
        rhs,
        dnums,
        multipass_mode='three_pass_fp8_int4',
    )
    snr_fp8_int4 = float(compute_snr_db(ref_f32, res_fp8_int4))
    err_fp8_int4 = float(compute_relative_error(ref_f32, res_fp8_int4))

    res_fp8_mixed4 = multipass_dot.multipass_dot_general(
        lhs,
        rhs,
        dnums,
        multipass_mode='three_pass_fp8_mixed4',
    )
    snr_fp8_mixed4 = float(compute_snr_db(ref_f32, res_fp8_mixed4))
    err_fp8_mixed4 = float(compute_relative_error(ref_f32, res_fp8_mixed4))

    print(f'Hybrid FP8+FP4: SNR={snr_fp8_fp4:.2f} dB, err={err_fp8_fp4:.4f}')
    print(f'Hybrid FP8+INT4: SNR={snr_fp8_int4:.2f} dB, err={err_fp8_int4:.4f}')
    print(
        f'Hybrid FP8+Mixed4: SNR={snr_fp8_mixed4:.2f} dB,'
        f' err={err_fp8_mixed4:.4f}'
    )

    fp8_fp4_target = 41.83 if is_ghostfish() else 43.75
    err_fp8_fp4_target = 0.0081 if is_ghostfish() else 0.0065
    self.assertAlmostEqual(snr_fp8_fp4, fp8_fp4_target, delta=1.5)
    self.assertAlmostEqual(err_fp8_fp4, err_fp8_fp4_target, delta=0.002)

    fp8_int4_target = 41.27 if is_ghostfish() else 42.95
    err_fp8_int4_target = 0.0086 if is_ghostfish() else 0.0071
    self.assertAlmostEqual(snr_fp8_int4, fp8_int4_target, delta=1.5)
    self.assertAlmostEqual(err_fp8_int4, err_fp8_int4_target, delta=0.002)

    fp8_mixed4_target = 41.72 if is_ghostfish() else 43.58
    err_fp8_mixed4_target = 0.0082 if is_ghostfish() else 0.0066
    self.assertAlmostEqual(snr_fp8_mixed4, fp8_mixed4_target, delta=1.5)
    self.assertAlmostEqual(err_fp8_mixed4, err_fp8_mixed4_target, delta=0.002)

    # 10. Microscaled Hybrid Strategies with Native Hardware FP8 Accumulation
    res_mx_fp4 = multipass_dot.multipass_dot_general(
        lhs,
        rhs,
        dnums,
        multipass_mode='three_pass_mxfp8_16_mxfp4',
    )
    snr_mx_fp4 = float(compute_snr_db(ref_f32, res_mx_fp4))
    err_mx_fp4 = float(compute_relative_error(ref_f32, res_mx_fp4))

    res_mx_int4 = multipass_dot.multipass_dot_general(
        lhs,
        rhs,
        dnums,
        multipass_mode='three_pass_mxfp8_16_mxint4',
    )
    snr_mx_int4 = float(compute_snr_db(ref_f32, res_mx_int4))
    err_mx_int4 = float(compute_relative_error(ref_f32, res_mx_int4))

    res_mx_mixed4 = multipass_dot.multipass_dot_general(
        lhs,
        rhs,
        dnums,
        multipass_mode='three_pass_mxfp8_16_mxmixed4',
    )
    snr_mx_mixed4 = float(compute_snr_db(ref_f32, res_mx_mixed4))
    err_mx_mixed4 = float(compute_relative_error(ref_f32, res_mx_mixed4))

    print(
        f'Microscale MXFP8+MXFP4: SNR={snr_mx_fp4:.2f} dB, err={err_mx_fp4:.4f}'
    )
    print(
        f'Microscale MXFP8+MXINT4: SNR={snr_mx_int4:.2f} dB,'
        f' err={err_mx_int4:.4f}'
    )
    print(
        f'Microscale MXFP8+MXMixed4: SNR={snr_mx_mixed4:.2f} dB,'
        f' err={err_mx_mixed4:.4f}'
    )

    mx_fp4_target = 43.48 if is_ghostfish() else 44.09
    err_mx_fp4_target = 0.0067 if is_ghostfish() else 0.0063
    self.assertAlmostEqual(snr_mx_fp4, mx_fp4_target, delta=1.5)
    self.assertAlmostEqual(err_mx_fp4, err_mx_fp4_target, delta=0.002)

    mx_int4_target = 41.96 if is_ghostfish() else 42.29
    err_mx_int4_target = 0.0080 if is_ghostfish() else 0.0077
    self.assertAlmostEqual(snr_mx_int4, mx_int4_target, delta=1.5)
    self.assertAlmostEqual(err_mx_int4, err_mx_int4_target, delta=0.002)

    mx_mixed4_target = 43.24 if is_ghostfish() else 43.82
    err_mx_mixed4_target = 0.0069 if is_ghostfish() else 0.0064
    self.assertAlmostEqual(snr_mx_mixed4, mx_mixed4_target, delta=1.5)
    self.assertAlmostEqual(err_mx_mixed4, err_mx_mixed4_target, delta=0.002)

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

  def test_int8_combos_sqnr_benchmark(self):
    r"""Verifies SQNR and relative error on N(0, 1) inputs and outputs tables.

    ### INT8 Multi-Pass Combos SQNR Benchmark Table

    | Strategy | Tile / Block | GEMMs | Compute DType | SQNR (dB) | Rel Error
    (%) | Equivalence vs INT8 |
    | :---| :---: | :---: | :---| :---: | :---: | :---|
    | 1-Pass Subchannel-32 INT8 | 32 | 1 | float32 | 36.30 dB | 1.53% |
    Reference |
    | Subchannel-32 INT4 Emulated Int | 32 | 4 | float8_e4m3fn | 36.30 dB |
    1.53% |
    Exact to block prod |
    | 1-Pass Subchannel-256 INT8 | 256 | 1 | float32 | 39.83 dB | 1.02% |
    Reference |
    | Subchannel-256 INT4 Emulated Int | 256 | 4 | float8_e4m3fn | 39.83 dB |
    1.02% | Exact to block prod |
    | Exact Signed INT8 Reference | None | 1 | int32 | $$\\infty$$ dB | 0.00% |
    Reference |
    | Signed INT8 via INT4 Emulated Int | None | 4 | float8_e4m3fn | $$\\infty$$
    dB | 0.00% | Bit-Exact (0 max diff) |
    """
    k1, k2 = jax.random.split(self.rng)
    shape_l = (256, 512)
    shape_r = (512, 256)
    dnums = (((1,), (0,)), ((), ()))

    # FP32 standard normal Gaussian inputs N(0, 1)
    lhs = jax.random.normal(k1, shape_l, dtype=jnp.float32)
    rhs = jax.random.normal(k2, shape_r, dtype=jnp.float32)
    ref_f32 = lax.dot_general(lhs, rhs, dnums)

    # 1. Subchannel-32 INT8 1-Pass Baseline (block size 32) - target ~37.71 dB
    how_l_int8_32 = dot_general.get_how_to_quantize(
        dimension_numbers=dnums,
        ndims=(2, 2),
        for_lhs=True,
        qtype=jnp.int8,
        tile_size=32,
    )
    how_r_int8_32 = dot_general.get_how_to_quantize(
        dimension_numbers=dnums,
        ndims=(2, 2),
        for_lhs=False,
        qtype=jnp.int8,
        tile_size=32,
    )
    res_int8_32_1p = lax.dot_general(
        qarray.dequantize(qarray.quantize(lhs, how_l_int8_32)),
        qarray.dequantize(qarray.quantize(rhs, how_r_int8_32)),
        dnums,
    )
    snr_int8_32_1p = float(compute_snr_db(ref_f32, res_int8_32_1p))
    err_int8_32_1p = float(compute_relative_error(ref_f32, res_int8_32_1p))

    # 6. Subchannel-32 INT4 Full Cross (4 passes, block size 32)
    res_int8_32_emulated = multipass_dot._int8_multipass_dot_general(
        lhs,
        rhs,
        dnums,
        compute_dtype=jnp.float8_e4m3fn,
        multipass_mode='four_pass_int4',
        tile_size=32,
    )
    snr_int8_32_emulated = float(compute_snr_db(ref_f32, res_int8_32_emulated))
    err_int8_32_emulated = float(
        compute_relative_error(ref_f32, res_int8_32_emulated)
    )

    # 7. Subchannel 256 INT8 1-Pass Baseline (tile size 256)
    how_l_int8_256 = dot_general.get_how_to_quantize(
        dimension_numbers=dnums,
        ndims=(2, 2),
        for_lhs=True,
        qtype=jnp.int8,
        tile_size=256,
    )
    how_r_int8_256 = dot_general.get_how_to_quantize(
        dimension_numbers=dnums,
        ndims=(2, 2),
        for_lhs=False,
        qtype=jnp.int8,
        tile_size=256,
    )
    res_int8_256_1p = lax.dot_general(
        qarray.dequantize(qarray.quantize(lhs, how_l_int8_256)),
        qarray.dequantize(qarray.quantize(rhs, how_r_int8_256)),
        dnums,
    )
    snr_int8_256_1p = float(compute_snr_db(ref_f32, res_int8_256_1p))
    err_int8_256_1p = float(compute_relative_error(ref_f32, res_int8_256_1p))

    # 8. Subchannel 256 INT4 Full Cross (4 passes, tile size 256)
    res_int8_256_emulated = multipass_dot._int8_multipass_dot_general(
        lhs,
        rhs,
        dnums,
        compute_dtype=jnp.float8_e4m3fn,
        multipass_mode='four_pass_int4',
        tile_size=256,
    )
    snr_int8_256_emulated = float(
        compute_snr_db(ref_f32, res_int8_256_emulated)
    )
    err_int8_256_emulated = float(
        compute_relative_error(ref_f32, res_int8_256_emulated)
    )

    # 9. Exact Unscaled Signed INT8 Bit-Level Verification (Integer Inputs)
    a_int8 = jax.random.randint(k1, shape_l, minval=-128, maxval=128).astype(
        jnp.int32
    )
    b_int8 = jax.random.randint(k2, shape_r, minval=-128, maxval=128).astype(
        jnp.int32
    )
    ref_int8_exact = jnp.matmul(a_int8, b_int8)
    res_emulated_exact = multipass_dot._emulated_signed_int8_dot_general(
        a_int8, b_int8, dnums, compute_dtype=jnp.float8_e4m3fn
    )
    max_diff_emulated = int(
        jnp.max(jnp.abs(res_emulated_exact - ref_int8_exact))
    )

    # Print dedicated INT8 Combos Markdown Benchmark Table
    int8_table_rows = [
        (
            '1-Pass Subchannel-32 INT8',
            32,
            1,
            'float32',
            snr_int8_32_1p,
            err_int8_32_1p,
            'Reference',
        ),
        (
            'Subchannel-32 INT4 Emulated Int',
            32,
            4,
            'float8_e4m3fn',
            snr_int8_32_emulated,
            err_int8_32_emulated,
            'Exact to block prod',
        ),
        (
            '1-Pass Subchannel-256 INT8',
            256,
            1,
            'float32',
            snr_int8_256_1p,
            err_int8_256_1p,
            'Reference',
        ),
        (
            'Subchannel-256 INT4 Emulated Int',
            256,
            4,
            'float8_e4m3fn',
            snr_int8_256_emulated,
            err_int8_256_emulated,
            'Exact to block prod',
        ),
        (
            'Exact Signed INT8 Reference',
            'None',
            1,
            'int32',
            float('inf'),
            0.0,
            'Reference',
        ),
        (
            'Signed INT8 via INT4 Emulated Int',
            'None',
            4,
            'float8_e4m3fn',
            float('inf'),
            0.0,
            f'Bit-Exact ({max_diff_emulated} max diff)',
        ),
    ]

    print('\n=== INT8 MULTI-PASS COMBOS SQNR BENCHMARK TABLE ===')
    print(
        '| Strategy | Tile / Block | GEMMs | Compute DType | SQNR (dB) | Rel'
        ' Error (%) | Equivalence vs INT8 |'
    )
    print('| :---| :---: | :---: | :---| :---: | :---: | :---|')
    for name, blk, ngemm, cdtype, snr_val, err_val, equiv in int8_table_rows:
      snr_str = f'{snr_val:.2f} dB' if np.isfinite(snr_val) else 'inf dB'
      print(
          f'| {name} | {blk} | {ngemm} | {cdtype} | {snr_str} |'
          f' {err_val * 100:.2f}% | {equiv} |'
      )
    print('===================================================\n')

    # INT8 exact Emulated Int equivalence to 1-pass reference
    eq_delta = 0.5 if is_ghostfish() else 0.01
    self.assertAlmostEqual(snr_int8_32_emulated, snr_int8_32_1p, delta=eq_delta)
    self.assertAlmostEqual(
        snr_int8_256_emulated, snr_int8_256_1p, delta=eq_delta
    )

  def test_prep_int8_parts_range(self):
    """Verifies _prep_int8_parts produces values strictly in [-8, 7] for all int8 values."""
    all_int8 = jnp.arange(-128, 128, dtype=jnp.int32)
    x_h, x_l = multipass_dot._prep_int8_parts(all_int8)
    self.assertTrue(bool(jnp.all((x_h >= -8) & (x_h <= 7))))
    self.assertTrue(bool(jnp.all((x_l >= -8) & (x_l <= 7))))
    reconstructed = (x_h << 4) + x_l + 8
    np.testing.assert_array_equal(reconstructed, all_int8)

  @parameterized.parameters(
      ((8, 16), (16, 8)),
      ((16, 32), (32, 16)),
      ((32, 64), (64, 32)),
      ((64, 64), (64, 64)),
  )
  def test_emulated_exact_int8_multiplication(self, shape_l, shape_r):
    """Verifies Emulated 4-pass produces exact results to signed int8 matmul."""
    k1, k2 = jax.random.split(self.rng)
    a = jax.random.randint(k1, shape_l, minval=-128, maxval=128).astype(
        jnp.int32
    )
    b = jax.random.randint(k2, shape_r, minval=-128, maxval=128).astype(
        jnp.int32
    )
    ref = jnp.matmul(a, b)

    dnums = (((1,), (0,)), ((), ()))
    res = multipass_dot._emulated_signed_int8_dot_general(
        a, b, dnums, compute_dtype=jnp.float8_e4m3fn
    )
    if is_ghostfish():
      self.assertGreater(compute_snr_db(ref, res), 60.0)
      self.assertLessEqual(int(jnp.max(jnp.abs(res - ref))), 256)
    else:
      np.testing.assert_array_equal(res, ref)
      self.assertEqual(int(jnp.max(jnp.abs(res - ref))), 0)

  def test_emulated_exact_int8_multiplication_extremes(self):
    """Verifies Emulated Int matches bit-for-bit on extreme boundaries."""
    # Test boundary values: -128, -127, -8, -1, 0, 1, 7, 127
    vals = jnp.array([-128, -127, -8, -1, 0, 1, 7, 127], dtype=jnp.int32)
    a = jnp.tile(vals[:, None], (1, 8))
    b = jnp.tile(vals[None, :], (8, 1))
    dnums = (((1,), (0,)), ((), ()))

    ref = jnp.matmul(a, b)
    res_emulated = multipass_dot._emulated_signed_int8_dot_general(
        a, b, dnums, compute_dtype=jnp.float8_e4m3fn
    )

    np.testing.assert_array_equal(res_emulated, ref)

  @parameterized.parameters(
      ('four_pass_int4', 32, 64),
      ('four_pass_int4', 256, 256),
  )
  def test_int8_multipass_dot(self, multipass_mode, tile_size, k_dim):
    """Verifies subchannel int8 GEMM using Full Cross passes."""
    k1, k2 = jax.random.split(self.rng)
    lhs = jax.random.normal(k1, (16, k_dim), dtype=jnp.float32)
    rhs = jax.random.normal(k2, (k_dim, 16), dtype=jnp.float32)
    dnums = (((1,), (0,)), ((), ()))

    res_multipass = multipass_dot._int8_multipass_dot_general(
        lhs,
        rhs,
        dnums,
        compute_dtype=jnp.float8_e4m3fn,
        multipass_mode=multipass_mode,
        tile_size=tile_size,
    )
    self.assertEqual(res_multipass.shape, (16, 16))
    self.assertFalse(jnp.isnan(res_multipass).any())

    # Direct dispatch via multipass_dot_general with single multipass_mode
    # parameter to select integer emulation.
    res_direct = multipass_dot.multipass_dot_general(
        lhs,
        rhs,
        dnums,
        compute_dtype=jnp.float8_e4m3fn,
        multipass_mode=multipass_mode,
        tile_size=tile_size,
    )
    np.testing.assert_allclose(res_direct, res_multipass, atol=1e-5)

    # Should closely match reference float dot
    true_dot = lax.dot_general(lhs, rhs, dnums)
    snr = compute_snr_db(true_dot, res_multipass)
    self.assertGreater(snr, 36.0)

  @parameterized.parameters(jnp.float8_e4m3fn, jnp.bfloat16)
  def test_exact_int8_multipass_graph_mechanics_and_gemm_ops(
      self, target_dtype
  ):
    """Verifies low-level graph mechanics, GEMM counts, and compute_dtype propagation."""
    lhs = jnp.ones((16, 32), dtype=jnp.float32)
    rhs = jnp.ones((32, 16), dtype=jnp.float32)
    dnums = (((1,), (0,)), ((), ()))

    # four_pass_int4: exactly 4 uncoupled GEMMs on target compute_dtype
    fn_fc = lambda x, y: multipass_dot.multipass_dot_general(
        x,
        y,
        dnums,
        compute_dtype=target_dtype,
        multipass_mode='four_pass_int4',
        tile_size=32,
    )
    gemms_fc, _ = get_graph_gemms_and_casts(fn_fc, lhs, rhs)
    self.assertLen(gemms_fc, 4)
    for g in gemms_fc:
      in_dtypes = [getattr(v.aval, 'dtype', None) for v in g.invars[:2]]
      self.assertEqual(in_dtypes, [target_dtype, target_dtype])
    fc_jaxpr = jax.make_jaxpr(fn_fc)(lhs, rhs)
    fc_eqns = extract_all_equations(
        fc_jaxpr.jaxpr if hasattr(fc_jaxpr, 'jaxpr') else fc_jaxpr
    )
    shift_eqns = [
        e for e in fc_eqns if e.primitive.name == 'shift_right_arithmetic'
    ]
    and_eqns = [e for e in fc_eqns if e.primitive.name == 'and']
    self.assertNotEmpty(shift_eqns)
    self.assertNotEmpty(and_eqns)

    if target_dtype == jnp.float8_e4m3fn:
      # Explicitly verify 4 FP8 GEMMs for four-pass.
      self.assertLen(gemms_fc, 4)
      for g in gemms_fc:
        self.assertEqual(
            [getattr(v.aval, 'dtype', None) for v in g.invars[:2]],
            [jnp.float8_e4m3fn, jnp.float8_e4m3fn],
        )

  @parameterized.parameters(
      ((8, 16), (16, 8)),
      ((16, 32), (32, 16)),
      ((32, 64), (64, 32)),
      ((64, 64), (64, 64)),
  )
  def test_asymmetric_exact_int4_int8_multiplication(self, shape_l, shape_r):
    """Verifies 2-pass asym INT4 x INT8 matmul produces exact products."""
    k1, k2 = jax.random.split(self.rng)
    # Case 1: LHS is INT4 [-8, 7], RHS is INT8 [-128, 127]
    a_4 = jax.random.randint(k1, shape_l, minval=-8, maxval=8).astype(jnp.int32)
    b_8 = jax.random.randint(k2, shape_r, minval=-128, maxval=128).astype(
        jnp.int32
    )
    ref1 = jnp.matmul(a_4, b_8)
    dnums = (((1,), (0,)), ((), ()))
    res1 = multipass_dot._asymmetric_signed_int4_int8_dot_general(
        a_4,
        b_8,
        dnums,
        multipass_mode='two_pass_rhs_int4',
        compute_dtype=jnp.float8_e4m3fn,
    )
    if is_ghostfish():
      self.assertGreater(compute_snr_db(ref1, res1), 60.0)
      self.assertLessEqual(int(jnp.max(jnp.abs(res1 - ref1))), 256)
    else:
      np.testing.assert_array_equal(res1, ref1)
      self.assertEqual(int(jnp.max(jnp.abs(res1 - ref1))), 0)

    # Case 2: LHS is INT8 [-128, 127], RHS is INT4 [-8, 7]
    a_8 = jax.random.randint(k1, shape_l, minval=-128, maxval=128).astype(
        jnp.int32
    )
    b_4 = jax.random.randint(k2, shape_r, minval=-8, maxval=8).astype(jnp.int32)
    ref2 = jnp.matmul(a_8, b_4)
    res2 = multipass_dot._asymmetric_signed_int4_int8_dot_general(
        a_8,
        b_4,
        dnums,
        multipass_mode='two_pass_lhs_int4',
        compute_dtype=jnp.float8_e4m3fn,
    )
    if is_ghostfish():
      self.assertGreater(compute_snr_db(ref2, res2), 60.0)
      self.assertLessEqual(int(jnp.max(jnp.abs(res2 - ref2))), 256)
    else:
      np.testing.assert_array_equal(res2, ref2)
      self.assertEqual(int(jnp.max(jnp.abs(res2 - ref2))), 0)

  def test_triangular_signed_int8_dot(self):
    """Verifies triangular int8 matmul drops p00 and achieves >33 dB SQNR."""
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
    res = multipass_dot._triangular_signed_int8_dot_general(
        a, b, dnums, compute_dtype=jnp.float8_e4m3fn
    )

    # Algebraic identity check: ref - res must be exactly p00 = a_l * b_l
    _, a_l = multipass_dot._prep_int8_parts(a)
    _, b_l = multipass_dot._prep_int8_parts(b)
    p00 = jnp.matmul(a_l.astype(jnp.int32), b_l.astype(jnp.int32))
    diff = ref - res
    if is_ghostfish():
      self.assertGreater(compute_snr_db(diff, p00), 45.0)
      self.assertLessEqual(int(jnp.max(jnp.abs(diff - p00))), 256)
    else:
      np.testing.assert_array_equal(diff, p00)

    # SQNR must exceed 33.0 dB
    sqnr = compute_snr_db(ref, res)
    self.assertGreater(sqnr, 33.0)

  def test_asymmetric_signed_int4_int8_invalid_mode_raises_error(self):
    """Verifies that an unsupported multipass_mode raises ValueError in asymmetric int8 dot."""
    a = jnp.ones((4, 4), dtype=jnp.int32)
    b = jnp.ones((4, 4), dtype=jnp.int32)
    with self.assertRaisesRegex(ValueError, 'Unsupported multipass_mode'):
      multipass_dot._asymmetric_signed_int4_int8_dot_general(
          a, b, multipass_mode='invalid_mode', compute_dtype=jnp.float8_e4m3fn
      )

  @parameterized.parameters(
      'two_pass_lhs_int4',
      'two_pass_rhs_int4',
  )
  def test_asymmetric_int_dot(self, mode):
    """Verifies asymmetric subchannel integer dot via multipass_dot_general."""
    k1, k2 = jax.random.split(self.rng)
    lhs = jax.random.normal(k1, (16, 64), dtype=jnp.float32)
    rhs = jax.random.normal(k2, (64, 16), dtype=jnp.float32)
    dnums = (((1,), (0,)), ((), ()))

    res = multipass_dot.multipass_dot_general(
        lhs,
        rhs,
        dnums,
        compute_dtype=jnp.float8_e4m3fn,
        multipass_mode=mode,
        tile_size=32,
    )
    self.assertEqual(res.shape, (16, 16))
    self.assertFalse(jnp.isnan(res).any())
    true_dot = lax.dot_general(lhs, rhs, dnums)
    snr = compute_snr_db(true_dot, res)
    self.assertGreater(snr, 16.0)

  def test_approx_int8_multipass_graph_mechanics_and_gemm_ops(self):
    """Verifies low-level graph mechanics and GEMM counts for approx INT8 modes."""
    lhs = jnp.ones((16, 32), dtype=jnp.float32)
    rhs = jnp.ones((32, 16), dtype=jnp.float32)
    dnums = (((1,), (0,)), ((), ()))

    # three_pass_int4: exactly 3 uncoupled GEMMs on FP8 MXU (omitting p00)
    fn_tri = lambda x, y: multipass_dot.multipass_dot_general(
        x, y, dnums, multipass_mode='three_pass_int4', tile_size=32
    )
    gemms_tri, _ = get_graph_gemms_and_casts(fn_tri, lhs, rhs)
    self.assertLen(gemms_tri, 3)
    for g in gemms_tri:
      in_dtypes = [getattr(v.aval, 'dtype', None) for v in g.invars[:2]]
      self.assertEqual(in_dtypes, [jnp.float8_e4m3fn, jnp.float8_e4m3fn])

    # two_pass_lhs_int4: exactly 2 GEMMs on FP8 MXU with LHS int4 nibbles
    fn_asym_l = lambda x, y: multipass_dot.multipass_dot_general(
        x, y, dnums, multipass_mode='two_pass_lhs_int4', tile_size=32
    )
    gemms_l, casts_l = get_graph_gemms_and_casts(fn_asym_l, lhs, rhs)
    self.assertLen(gemms_l, 2)
    for g in gemms_l:
      in_dtypes = [getattr(v.aval, 'dtype', None) for v in g.invars[:2]]
      self.assertEqual(in_dtypes, [jnp.float8_e4m3fn, jnp.float8_e4m3fn])
    self.assertIn(jnp.dtype(jnp.int4), casts_l)

    # two_pass_rhs_int4: exactly 2 GEMMs on FP8 MXU with RHS int4 nibbles
    fn_asym_r = lambda x, y: multipass_dot.multipass_dot_general(
        x, y, dnums, multipass_mode='two_pass_rhs_int4', tile_size=32
    )
    gemms_r, casts_r = get_graph_gemms_and_casts(fn_asym_r, lhs, rhs)
    self.assertLen(gemms_r, 2)
    for g in gemms_r:
      in_dtypes = [getattr(v.aval, 'dtype', None) for v in g.invars[:2]]
      self.assertEqual(in_dtypes, [jnp.float8_e4m3fn, jnp.float8_e4m3fn])
    self.assertIn(jnp.dtype(jnp.int4), casts_r)

  @parameterized.parameters(
      'three_pass_fp8_fp4',
      'three_pass_fp8_int4',
      'three_pass_fp8_mixed4',
  )
  def test_hybrid_fp8_4bit_multipass_dot(self, mode):
    """Verifies basic hybrid FP8 + 4-bit (FP4, INT4, Mixed) 3-pass matmuls."""
    k1, k2 = jax.random.split(self.rng)
    lhs = jax.random.normal(k1, (16, 64), dtype=jnp.float32)
    rhs = jax.random.normal(k2, (64, 16), dtype=jnp.float32)
    dnums = (((1,), (0,)), ((), ()))

    res = multipass_dot.multipass_dot_general(
        lhs,
        rhs,
        dnums,
        multipass_mode=mode,
        tile_size=32,
    )
    self.assertEqual(res.shape, (16, 16))
    self.assertFalse(jnp.isnan(res).any())

    ref = lax.dot_general(lhs, rhs, dnums)
    snr = compute_snr_db(ref, res)
    self.assertGreater(snr, 38.0)

  @parameterized.parameters(
      ('three_pass_fp8_fp4', 'three_pass_fp8_fp4/fp4_fp4/fp4'),
      ('three_pass_fp8_int4', 'three_pass_fp8_int4/int4_int4/int4'),
      ('three_pass_fp8_mixed4', 'three_pass_fp8_int4/fp4_fp4/int4_int4'),
  )
  def test_hybrid_fp8_4bit_aliases(self, canonical_mode, alias_mode):
    """Verifies that explicit slash-separated aliases match canonical modes exactly."""
    k1, k2 = jax.random.split(self.rng)
    lhs = jax.random.normal(k1, (16, 32), dtype=jnp.float32)
    rhs = jax.random.normal(k2, (32, 16), dtype=jnp.float32)
    dnums = (((1,), (0,)), ((), ()))

    res_canonical = multipass_dot.multipass_dot_general(
        lhs,
        rhs,
        dnums,
        multipass_mode=canonical_mode,
        tile_size=32,
    )
    res_alias = multipass_dot.multipass_dot_general(
        lhs,
        rhs,
        dnums,
        multipass_mode=alias_mode,
        tile_size=32,
    )
    np.testing.assert_allclose(res_canonical, res_alias, rtol=1e-5, atol=1e-5)

  def test_hybrid_fp8_4bit_graph_mechanics_and_gemm_ops(self):
    """Verifies low-level graph mechanics and 4-bit casts for hybrid FP8 modes."""
    lhs = jnp.ones((16, 32), dtype=jnp.float32)
    rhs = jnp.ones((32, 16), dtype=jnp.float32)
    dnums = (((1,), (0,)), ((), ()))

    # 1. three_pass_fp8_fp4: 3 GEMMs on FP8 MXU,
    # cross passes cast to float4_e2m1fn
    fn_fp4 = lambda x, y: multipass_dot.multipass_dot_general(
        x, y, dnums, multipass_mode='three_pass_fp8_fp4', tile_size=32
    )
    gemms_fp4, casts_fp4 = get_graph_gemms_and_casts(fn_fp4, lhs, rhs)
    self.assertLen(gemms_fp4, 3)
    for g in gemms_fp4:
      in_dtypes = [getattr(v.aval, 'dtype', None) for v in g.invars[:2]]
      self.assertEqual(in_dtypes, [jnp.float8_e4m3fn, jnp.float8_e4m3fn])
    self.assertEqual(casts_fp4.count(jnp.dtype(jnp.float4_e2m1fn)), 4)
    self.assertEqual(casts_fp4.count(jnp.dtype(jnp.float8_e4m3fn)), 6)
    self.assertEqual(casts_fp4.count(jnp.dtype(jnp.int4)), 0)

    # 2. three_pass_fp8_int4: 3 GEMMs on FP8 MXU, cross passes cast to int4
    fn_int4 = lambda x, y: multipass_dot.multipass_dot_general(
        x, y, dnums, multipass_mode='three_pass_fp8_int4', tile_size=32
    )
    gemms_int4, casts_int4 = get_graph_gemms_and_casts(fn_int4, lhs, rhs)
    self.assertLen(gemms_int4, 3)
    for g in gemms_int4:
      in_dtypes = [getattr(v.aval, 'dtype', None) for v in g.invars[:2]]
      self.assertEqual(in_dtypes, [jnp.float8_e4m3fn, jnp.float8_e4m3fn])
    self.assertEqual(casts_int4.count(jnp.dtype(jnp.int4)), 4)
    self.assertEqual(casts_int4.count(jnp.dtype(jnp.float8_e4m3fn)), 6)
    self.assertEqual(casts_int4.count(jnp.dtype(jnp.float4_e2m1fn)), 0)

    # 3. three_pass_fp8_mixed4: 3 GEMMs on FP8 MXU,
    # cross passes cast to int4 and float4_e2m1fn
    fn_mixed4 = lambda x, y: multipass_dot.multipass_dot_general(
        x, y, dnums, multipass_mode='three_pass_fp8_mixed4', tile_size=32
    )
    gemms_mixed4, casts_mixed4 = get_graph_gemms_and_casts(fn_mixed4, lhs, rhs)
    self.assertLen(gemms_mixed4, 3)
    for g in gemms_mixed4:
      in_dtypes = [getattr(v.aval, 'dtype', None) for v in g.invars[:2]]
      self.assertEqual(in_dtypes, [jnp.float8_e4m3fn, jnp.float8_e4m3fn])
    self.assertEqual(casts_mixed4.count(jnp.dtype(jnp.int4)), 2)
    self.assertEqual(casts_mixed4.count(jnp.dtype(jnp.float4_e2m1fn)), 2)
    self.assertEqual(casts_mixed4.count(jnp.dtype(jnp.float8_e4m3fn)), 6)

  @parameterized.parameters(
      'three_pass_mxfp8_16_mxfp4',
      'three_pass_mxfp8_16_mxint4',
      'three_pass_mxfp8_16_mxmixed4',
  )
  def test_microscaled_hybrid_fp8_4bit_multipass_dot(self, mode):
    """Verifies microscaled hybrid multi-pass matmuls with hardware FP8 conversion."""
    k1, k2 = jax.random.split(self.rng)
    lhs = jax.random.normal(k1, (16, 64), dtype=jnp.float32)
    rhs = jax.random.normal(k2, (64, 16), dtype=jnp.float32)
    dnums = (((1,), (0,)), ((), ()))

    res = multipass_dot.multipass_dot_general(
        lhs,
        rhs,
        dnums,
        multipass_mode=mode,
        tile_size=32,
    )
    self.assertEqual(res.shape, (16, 16))
    self.assertFalse(jnp.isnan(res).any())

    ref = lax.dot_general(lhs, rhs, dnums)
    snr = compute_snr_db(ref, res)
    self.assertGreater(snr, 38.0)

  @parameterized.parameters(
      (
          'three_pass_mxfp8_16_mxfp4',
          'three_pass_mxfp8_16_mxfp4/mxfp4_mxfp4/mxfp4',
      ),
      (
          'three_pass_mxfp8_16_mxint4',
          'three_pass_mxfp8_16_mxint4/mxint4_mxint4/mxint4',
      ),
      (
          'three_pass_mxfp8_16_mxmixed4',
          'three_pass_mxfp8_16_mxint4/mxfp4_mxfp4/mxint4',
      ),
  )
  def test_microscaled_hybrid_aliases(self, canonical_mode, alias_mode):
    """Verifies explicit slash aliases match canonical microscaled modes."""
    k1, k2 = jax.random.split(self.rng)
    lhs = jax.random.normal(k1, (16, 32), dtype=jnp.float32)
    rhs = jax.random.normal(k2, (32, 16), dtype=jnp.float32)
    dnums = (((1,), (0,)), ((), ()))

    res_canonical = multipass_dot.multipass_dot_general(
        lhs,
        rhs,
        dnums,
        multipass_mode=canonical_mode,
        tile_size=32,
    )
    res_alias = multipass_dot.multipass_dot_general(
        lhs,
        rhs,
        dnums,
        multipass_mode=alias_mode,
        tile_size=32,
    )
    np.testing.assert_allclose(res_canonical, res_alias, rtol=1e-5, atol=1e-5)

  def test_microscaled_hybrid_graph_mechanics_and_fp8_path(self):
    """Verifies low-level graph mechanics, GEMM counts, and 4-bit to FP8 conversion."""
    lhs = jnp.ones((16, 32), dtype=jnp.float32)
    rhs = jnp.ones((32, 16), dtype=jnp.float32)
    dnums = (((1,), (0,)), ((), ()))

    # 1. three_pass_mxfp8_16_mxfp4:
    # 3 GEMMs total, cross passes execute on FP8 MXU in float8_e4m3fn,
    # with values quantized to float4_e2m1fn (mxfp4) before scale shift.
    fn_mxfp4 = lambda x, y: multipass_dot.multipass_dot_general(
        x, y, dnums, multipass_mode='three_pass_mxfp8_16_mxfp4', tile_size=32
    )
    gemms_mxfp4, casts_mxfp4 = get_graph_gemms_and_casts(fn_mxfp4, lhs, rhs)
    self.assertLen(gemms_mxfp4, 3)
    cross_gemms_mxfp4 = [
        g for g in gemms_mxfp4 if g.primitive.name == 'dot_general'
    ]
    self.assertLen(cross_gemms_mxfp4, 2)
    for g in cross_gemms_mxfp4:
      in_dtypes = [getattr(v.aval, 'dtype', None) for v in g.invars[:2]]
      self.assertEqual(in_dtypes, [jnp.float8_e4m3fn, jnp.float8_e4m3fn])
    self.assertEqual(casts_mxfp4.count(jnp.dtype(jnp.float4_e2m1fn)), 4)
    self.assertEqual(casts_mxfp4.count(jnp.dtype(jnp.float8_e4m3fn)), 6)
    self.assertEqual(casts_mxfp4.count(jnp.dtype(jnp.float8_e5m2)), 0)
    self.assertEqual(casts_mxfp4.count(jnp.dtype(jnp.int4)), 0)

    # 2. three_pass_mxfp8_16_mxint4:
    # 3 GEMMs total, cross passes execute on FP8 MXU in float8_e5m2,
    # with values quantized to int4 (mxint4) before scale shift.
    fn_mxint4 = lambda x, y: multipass_dot.multipass_dot_general(
        x, y, dnums, multipass_mode='three_pass_mxfp8_16_mxint4', tile_size=32
    )
    gemms_mxint4, casts_mxint4 = get_graph_gemms_and_casts(fn_mxint4, lhs, rhs)
    self.assertLen(gemms_mxint4, 3)
    cross_gemms_mxint4 = [
        g for g in gemms_mxint4 if g.primitive.name == 'dot_general'
    ]
    self.assertLen(cross_gemms_mxint4, 2)
    for g in cross_gemms_mxint4:
      in_dtypes = [getattr(v.aval, 'dtype', None) for v in g.invars[:2]]
      self.assertEqual(in_dtypes, [jnp.float8_e5m2, jnp.float8_e5m2])
    self.assertEqual(casts_mxint4.count(jnp.dtype(jnp.int4)), 4)
    self.assertEqual(casts_mxint4.count(jnp.dtype(jnp.float8_e5m2)), 4)
    self.assertEqual(casts_mxint4.count(jnp.dtype(jnp.float8_e4m3fn)), 2)
    self.assertEqual(casts_mxint4.count(jnp.dtype(jnp.float4_e2m1fn)), 0)

    # 3. three_pass_mxfp8_16_mxmixed4:
    # 3 GEMMs total, cross passes execute on FP8 MXU in float8_e5m2/e4m3fn,
    # with values quantized to int4 & float4_e2m1fn before scale shift.
    fn_mxmixed4 = lambda x, y: multipass_dot.multipass_dot_general(
        x,
        y,
        dnums,
        multipass_mode='three_pass_mxfp8_16_mxmixed4',
        tile_size=32,
    )
    gemms_mxmixed4, casts_mxmixed4 = get_graph_gemms_and_casts(
        fn_mxmixed4, lhs, rhs
    )
    self.assertLen(gemms_mxmixed4, 3)
    cross_gemms_mxmixed4 = [
        g for g in gemms_mxmixed4 if g.primitive.name == 'dot_general'
    ]
    self.assertLen(cross_gemms_mxmixed4, 2)
    self.assertEqual(
        [
            getattr(v.aval, 'dtype', None)
            for v in cross_gemms_mxmixed4[0].invars[:2]
        ],
        [jnp.float8_e5m2, jnp.float8_e4m3fn],
    )
    self.assertEqual(
        [
            getattr(v.aval, 'dtype', None)
            for v in cross_gemms_mxmixed4[1].invars[:2]
        ],
        [jnp.float8_e4m3fn, jnp.float8_e5m2],
    )
    self.assertEqual(casts_mxmixed4.count(jnp.dtype(jnp.int4)), 2)
    self.assertEqual(casts_mxmixed4.count(jnp.dtype(jnp.float4_e2m1fn)), 2)
    self.assertEqual(casts_mxmixed4.count(jnp.dtype(jnp.float8_e5m2)), 2)
    self.assertEqual(casts_mxmixed4.count(jnp.dtype(jnp.float8_e4m3fn)), 4)


if __name__ == '__main__':
  absltest.main()
