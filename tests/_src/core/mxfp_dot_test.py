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
- Blackwell (B200): Native MXFP8 and NVFP4 support.
- H100: Supported via JAX emulation/decomposition (Verified).
- TPUs (v5e, v6e, etc.): Currently raises NotImplementedError in JAX.
- CPU: Currently raises NotImplementedError in JAX.
"""

import functools
from unittest import mock
from absl import logging
from absl.testing import absltest
import jax
import jax.numpy as jnp
import numpy as np
from qwix._src.core import mxfp_dot
from qwix._src.core import qarray


@functools.lru_cache(maxsize=None)
def _scaled_matmul_supported() -> bool:
  """Returns whether `jax.nn.scaled_matmul` can run on the current backend.

  The `scaled_matmul` primitive lacks an MLIR lowering rule on some platforms
  (e.g. CPU and TPU) in released JAX builds, where it raises
  `NotImplementedError` at lowering time. We probe support at runtime and cache
  the result so the affected tests automatically re-enable once a released JAX
  gains a lowering for the current platform.
  """
  lhs = jnp.zeros((1, 1, 32), dtype=jnp.float32)
  rhs = jnp.zeros((1, 1, 32), dtype=jnp.float32)
  scale = jnp.ones((1, 1, 1), dtype=jnp.float32)
  try:
    jax.jit(jax.nn.scaled_matmul)(lhs, rhs, scale, scale).block_until_ready()
    return True
  except NotImplementedError:
    return False


def _skip_if_scaled_matmul_unsupported(test_case: absltest.TestCase) -> None:
  if not _scaled_matmul_supported():
    test_case.skipTest(
        "jax.nn.scaled_matmul has no lowering for platform "
        f"{jax.devices()[0].platform!r} in this JAX build (raises "
        "NotImplementedError). Re-enables automatically when a released JAX "
        "adds support."
    )


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
  dtype_max = jnp.finfo(data_type).max

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
    _skip_if_scaled_matmul_unsupported(self)
    lhs, rhs = self._generate_test_data()

    # scaled_matmul with 1.0 scales should be same as normal matmul
    lhs_scale = jnp.ones((1, 32, 2), dtype=jnp.float32)
    rhs_scale = jnp.ones((1, 32, 2), dtype=jnp.float32)

    res = jax.nn.scaled_matmul(lhs, rhs, lhs_scale, rhs_scale)
    expected = jnp.einsum("bmk,bnk->bmn", lhs, rhs)
    np.testing.assert_allclose(res, expected, rtol=5e-3, atol=5e-3)
    logging.info("jax.nn.scaled_matmul (F32) SUCCEEDED")

    hlo = self._get_hlo(jax.nn.scaled_matmul, lhs, rhs, lhs_scale, rhs_scale)
    logging.info("F32 Baseline HLO snippet: %s", hlo[:500])

  def run_mxfp_test(self, mxfp_format, data_type, scale_type, block_size):
    """Helper to run a specific MXFP configuration test.

    Args:
      mxfp_format: String identifier for the configuration (e.g., 'mxfp8',
        'mxfp4').
      data_type: The JAX data type for the inputs.
      scale_type: The JAX data type for the scales.
      block_size: The block size for microscaling.
    """
    _skip_if_scaled_matmul_unsupported(self)
    logging.info(
        "Running MXFP test: mxfp_format=%s, data_type=%s, scale_type=%s,"
        " block_size=%d",
        mxfp_format,
        data_type,
        scale_type,
        block_size,
    )

    lhs, rhs = self._generate_test_data()

    lhs_q, lhs_scale = local_quantize(lhs, data_type, scale_type, block_size)
    rhs_q, rhs_scale = local_quantize(rhs, data_type, scale_type, block_size)

    res = jax.nn.scaled_matmul(lhs_q, rhs_q, lhs_scale, rhs_scale)
    logging.info("jax.nn.scaled_matmul succeeded for %s", mxfp_format)

    expected = reference_scaled_matmul(
        lhs_q.astype(jnp.float32),
        rhs_q.astype(jnp.float32),
        lhs_scale.astype(jnp.float32),
        rhs_scale.astype(jnp.float32),
    )

    np.testing.assert_allclose(res, expected, rtol=5e-2, atol=5e-2)

    hlo = self._get_hlo(
        jax.nn.scaled_matmul, lhs_q, rhs_q, lhs_scale, rhs_scale
    )
    logging.info("%s HLO Snippet: %s", mxfp_format, hlo[:500])

    # Check for custom-call pattern
    if "custom-call" in hlo and (
        "block_scaled_dot" in hlo or "blockScaledDot" in hlo
    ):
      logging.info(
          "Native %s support (custom-call) DETECTED in HLO", mxfp_format
      )
    else:
      logging.info(
          "Native %s support NOT detected in HLO. Likely emulated or"
          " decomposed.",
          mxfp_format,
      )

  def test_scaled_matmul_mxfp8(self):
    self.run_mxfp_test(
        mxfp_format="mxfp8",
        data_type=jnp.float8_e4m3fn,
        scale_type=jnp.float8_e8m0fnu,
        block_size=32,
    )

  def test_scaled_matmul_mxfp4(self):
    self.run_mxfp_test(
        mxfp_format="mxfp4",
        data_type=jnp.float4_e2m1fn,
        scale_type=jnp.float8_e8m0fnu,
        block_size=32,
    )

  def test_scaled_matmul_nvfp4(self):
    self.run_mxfp_test(
        mxfp_format="nvfp4",
        data_type=jnp.float4_e2m1fn,
        scale_type=jnp.float8_e4m3fn,
        block_size=16,
    )


class MxfpDotTest(absltest.TestCase):
  """Tests for mxfp_dot dispatcher and shape handling."""

  def test_flatten_to_3d(self):
    val = jnp.ones((2, 3, 4, 5))
    scale = jnp.ones((2, 3, 4, 1))
    ca = (3,)
    ba = (0, 1)

    operand = qarray.QArray(val, scale, qtype="mxfp8")
    val_3d, scale_3d = mxfp_dot._flatten_to_3d(operand, ca, ba)
    self.assertEqual(val_3d.shape, (6, 4, 5))
    self.assertEqual(scale_3d.shape, (6, 4, 1))

  def test_flatten_to_3d_with_broadcasting(self):
    val = jnp.ones((2, 4, 5))
    scale = jnp.ones((1, 4, 1))
    ca = (2,)
    ba = (0,)

    operand = qarray.QArray(val, scale, qtype="mxfp8")
    val_3d, scale_3d = mxfp_dot._flatten_to_3d(operand, ca, ba)
    self.assertEqual(val_3d.shape, (2, 4, 5))
    self.assertEqual(scale_3d.shape, (2, 4, 1))

  def test_unflatten_from_3d(self):
    out_3d = jnp.ones((6, 4, 6))
    lhs = qarray.QArray(
        qvalue=jnp.ones((2, 3, 4, 5)),
        scale=jnp.ones((2, 3, 4, 1)),
        qtype="mxfp8",
    )
    rhs = qarray.QArray(
        qvalue=jnp.ones((2, 3, 6, 5)),
        scale=jnp.ones((2, 3, 6, 1)),
        qtype="mxfp8",
    )
    dnums = (((3,), (3,)), ((0, 1), (0, 1)))
    res = mxfp_dot._unflatten_from_3d(out_3d, lhs, rhs, dnums)
    self.assertEqual(res.shape, (2, 3, 4, 6))

  def test_mxfp_dot_general_emulation_fallback(self):
    lhs = qarray.QArray(
        qvalue=jnp.ones((2, 32), jnp.float8_e4m3fn),
        scale=jnp.ones((2, 1)),
        qtype="mxfp8",
    )
    rhs = qarray.QArray(
        qvalue=jnp.ones((2, 32), jnp.float8_e4m3fn),
        scale=jnp.ones((2, 1)),
        qtype="mxfp8",
    )

    platform = jax.devices()[0].platform
    if platform == "cpu":
      res = mxfp_dot.mxfp_dot_general(
          lhs, rhs, dimension_numbers=(((1,), (1,)), ((), ()))
      )
      self.assertIsNone(res)

  @mock.patch.object(jax, "devices")
  def test_one_side_mxfp_fallback(self, mock_devices):
    mock_device = mock.MagicMock()
    mock_device.platform = "gpu"
    mock_devices.return_value = [mock_device]

    mxfp_dot._get_primary_platform.cache_clear()
    try:
      lhs = qarray.QArray(
          qvalue=jnp.ones((2, 32), jnp.float8_e4m3fn),
          scale=jnp.ones((2, 1)),
          qtype="mxfp8",
      )
      rhs = jnp.ones((2, 32), jnp.float32)

      res = mxfp_dot.mxfp_dot_general(
          lhs, rhs, dimension_numbers=(((1,), (1,)), ((), ()))
      )
      self.assertIsNone(res)
    finally:
      mxfp_dot._get_primary_platform.cache_clear()


if __name__ == "__main__":
  absltest.main()
