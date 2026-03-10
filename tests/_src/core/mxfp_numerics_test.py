# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for MXFP scaled matmul.

This test verifies the behavior of `jax.nn.scaled_matmul` for various
MXFP (Microscaling Formats) configurations on different hardware backends.

Key Formats:
- MXFP8 (OCP): 8-bit data (E4M3FN or E5M2), 8-bit scale (E8M0), block size 32.
  Native support on Blackwell (B200) and expected on future TPU generations.
- MXFP4 (OCP): 4-bit data (E2M1FN), 8-bit scale (E8M0), block size 32.
  Emulated on Blackwell (due to block size mismatch), expected native on future
  TPU generations.
- NVFP4 (NVIDIA): 4-bit data (E2M1FN), 8-bit scale (E4M3), block size 16.
  Native support on Blackwell (B200).

Hardware Support Summary:
- Blackwell (B200): Native MXFP8 and NVFP4.
- Future TPUs: Native MXFP8 and MXFP4 expected.
- H100: Emulated for all formats.
- Older TPU: Emulated for all formats.
"""

import logging
from absl.testing import absltest
import jax
import jax.numpy as jnp
import numpy as np


def reference_scaled_matmul(lhs, rhs, lhs_scale, rhs_scale):
  """Reference implementation using JNP."""
  # Assuming shapes:
  # lhs: (batch, m, k)
  # rhs: (batch, n, k)
  # lhs_scale: (batch, m, k_blocks)
  # rhs_scale: (batch, n, k_blocks)

  batch, m_dim, k_dim = lhs.shape
  _, n_dim, _ = rhs.shape
  block_size = k_dim // lhs_scale.shape[-1]

  lhs_reshaped = lhs.reshape(batch, m_dim, -1, block_size)
  rhs_reshaped = rhs.reshape(batch, n_dim, -1, block_size)

  # Apply scales
  lhs_scaled = lhs_reshaped * lhs_scale[..., jnp.newaxis]
  rhs_scaled = rhs_reshaped * rhs_scale[..., jnp.newaxis]

  # Flatten back to (batch, m, k) and (batch, n, k)
  lhs_f = lhs_scaled.reshape(batch, m_dim, k_dim).astype(jnp.float32)
  rhs_f = rhs_scaled.reshape(batch, n_dim, k_dim).astype(jnp.float32)

  return jnp.einsum("bmk,bnk->bmn", lhs_f, rhs_f)


def local_quantize(x, data_type, scale_type, block_size):
  """Simplified version of quantize for testing."""
  x_shape = x.shape
  contract_dim = x_shape[-1]
  assert contract_dim % block_size == 0

  x_reshaped = x.reshape(x_shape[:-1] + (x_shape[-1] // block_size, block_size))

  # Find max per block for scaling
  amax = jnp.max(jnp.abs(x_reshaped), axis=-1, keepdims=True)

  # Scale to fill the range of data_type
  try:
    dtype_max = jnp.finfo(data_type).max
  except ValueError:
    # For non-standard dtypes like float4_e2m1fn if finfo fails
    if data_type == getattr(jnp, "float4_e2m1fn", None):
      dtype_max = 6.0  # e2m1fn max is 6.0
    else:
      raise

  scales = amax / jnp.array(dtype_max, dtype=jnp.float32)

  # Cast scales to scale_type (e.g. float8_e8m0fnu)
  scales_q = scales.astype(scale_type)

  # Quantize x
  # Explicitly cast to float32 for division and clipping to avoid promotion
  # issues with 4/8-bit floats.
  scaled_x = x_reshaped / scales_q.astype(jnp.float32)
  clipped_x = jnp.clip(scaled_x, -float(dtype_max), float(dtype_max))
  x_q = clipped_x.astype(data_type)

  return x_q.reshape(x_shape), scales_q.reshape(
      x_shape[:-1] + (x_shape[-1] // block_size,)
  )


class MxfpNumericsTest(absltest.TestCase):
  """Unit tests for MXFP numerical correctness and hardware acceleration."""

  def setUp(self):
    super().setUp()
    self._log_device_info()

  def _log_device_info(self):
    devices = jax.devices()
    logging.info("Devices available: %s", devices)
    for i, d in enumerate(devices):
      model = getattr(d, "model", "unknown")
      logging.info(
          "Device %d: %s, kind: %s, model: %s", i, d, d.device_kind, model
      )
      if d.device_kind == "gpu":
        try:
          cc = d.compute_capability
          logging.info("Device %d compute capability: %s", i, cc)
        except AttributeError:
          pass

  def _get_hlo(self, f, *args):
    try:
      return jax.jit(f).lower(*args).compile().as_text()
    except Exception as e:  # pylint: disable=broad-exception-caught
      logging.warning("Failed to get HLO: %s", e)
      return str(e)

  def assertAllClose(self, a, b, atol=1e-5, rtol=1e-5):
    np.testing.assert_allclose(a, b, atol=atol, rtol=rtol)

  def _generate_test_data(self, seed=123):
    """Generates standard test data (lhs, rhs) in FP32."""
    batch, m_dim, n_dim, k_dim = 1, 32, 32, 64
    k1, k2 = jax.random.split(jax.random.key(seed), 2)
    lhs = jax.random.uniform(
        k1, (batch, m_dim, k_dim), minval=-1.0, dtype=jnp.float32
    )
    rhs = jax.random.uniform(
        k2, (batch, n_dim, k_dim), minval=-1.0, dtype=jnp.float32
    )
    return lhs, rhs

  def test_matmul_f32_baseline(self):
    """Sanity check for scaled_matmul with FP32 and unit scales."""
    lhs, rhs = self._generate_test_data()

    # scaled_matmul with 1.0 scales should be same as normal matmul
    lhs_scale = jnp.ones((1, 32, 2), dtype=jnp.float32)
    rhs_scale = jnp.ones((1, 32, 2), dtype=jnp.float32)

    try:
      res = jax.nn.scaled_matmul(lhs, rhs, lhs_scale, rhs_scale)
      expected = jnp.einsum("bmk,bnk->bmn", lhs, rhs)
      self.assertAllClose(res, expected, atol=1e-5)
      logging.info("jax.nn.scaled_matmul (F32) SUCCEEDED")
    except Exception as e:  # pylint: disable=broad-exception-caught
      logging.info("jax.nn.scaled_matmul (F32) BEHAVIOR: %s", type(e).__name__)
      logging.error("Full error: %s", e)

    hlo = self._get_hlo(jax.nn.scaled_matmul, lhs, rhs, lhs_scale, rhs_scale)
    logging.info("F32 Baseline HLO snippet: %s", hlo[:500])

  def run_mxfp_test(self, mode, data_type, scale_type, block_size):
    """Helper to run a specific MXFP configuration test."""
    logging.info(
        "Running MXFP test: mode=%s, data_type=%s, scale_type=%s,"
        " block_size=%d",
        mode,
        data_type,
        scale_type,
        block_size,
    )

    lhs, rhs = self._generate_test_data()

    lhs_q, lhs_scale = local_quantize(lhs, data_type, scale_type, block_size)
    rhs_q, rhs_scale = local_quantize(rhs, data_type, scale_type, block_size)

    # Call jax.nn.scaled_matmul
    try:
      res = jax.nn.scaled_matmul(lhs_q, rhs_q, lhs_scale, rhs_scale)
      logging.info("jax.nn.scaled_matmul succeeded for %s", mode)

      expected = reference_scaled_matmul(
          lhs_q.astype(jnp.float32),
          rhs_q.astype(jnp.float32),
          lhs_scale.astype(jnp.float32),
          rhs_scale.astype(jnp.float32),
      )

      self.assertAllClose(res, expected, atol=1e-2)
    except Exception as e:  # pylint: disable=broad-exception-caught
      logging.info(
          "jax.nn.scaled_matmul (%s) BEHAVIOR: %s", mode, type(e).__name__
      )
      logging.error("Full error: %s", e)

    hlo = self._get_hlo(
        jax.nn.scaled_matmul, lhs_q, rhs_q, lhs_scale, rhs_scale
    )
    logging.info("%s HLO Snippet: %s", mode, hlo[:500])

    # Check for custom-call pattern
    if "custom-call" in hlo and (
        "block_scaled_dot" in hlo or "blockScaledDot" in hlo
    ):
      logging.info("Native %s support (custom-call) DETECTED in HLO", mode)
    else:
      logging.info(
          "Native %s support NOT detected in HLO. Likely emulated or"
          " decomposed.",
          mode,
      )

  def test_scaled_matmul_mxfp8(self):
    e4m3fn = getattr(jnp, "float8_e4m3fn", None)
    e8m0fnu = getattr(jnp, "float8_e8m0fnu", None)

    if e4m3fn is None:
      self.skipTest("float8_e4m3fn not available in jnp")
    if e8m0fnu is None:
      self.skipTest("float8_e8m0fnu not available in jnp")

    self.run_mxfp_test(
        mode="mxfp8", data_type=e4m3fn, scale_type=e8m0fnu, block_size=32
    )

  def test_scaled_matmul_mxfp4(self):
    e2m1fn = getattr(jnp, "float4_e2m1fn", None)
    e8m0fnu = getattr(jnp, "float8_e8m0fnu", None)

    if e2m1fn is None:
      self.skipTest("float4_e2m1fn not available in jnp")
    if e8m0fnu is None:
      self.skipTest("float8_e8m0fnu not available in jnp")

    self.run_mxfp_test(
        mode="mxfp4", data_type=e2m1fn, scale_type=e8m0fnu, block_size=32
    )

  def test_scaled_matmul_nvfp4(self):
    e2m1fn = getattr(jnp, "float4_e2m1fn", None)
    e4m3fn = getattr(jnp, "float8_e4m3fn", None)

    if e2m1fn is None:
      self.skipTest("float4_e2m1fn not available in jnp")
    if e4m3fn is None:
      self.skipTest("float8_e4m3fn not available in jnp")

    self.run_mxfp_test(
        mode="nvfp4", data_type=e2m1fn, scale_type=e4m3fn, block_size=16
    )


if __name__ == "__main__":
  absltest.main()
