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
"""Tests for QEP core algorithm."""

import functools

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
from qwix._src.core import qarray
from qwix.contrib import gptq_core
from qwix.contrib import qep_core


def rel_rmse(x: jax.Array, y: jax.Array) -> jax.Array:
  return jnp.sqrt(jnp.mean((x - y) ** 2)) / jnp.sqrt(jnp.mean(y**2))


class QepCoreTest(parameterized.TestCase):

  def test_compute_qep_stats_shapes(self):
    """Tests that QEP stats have correct shapes."""
    x_quantized = jax.random.normal(jax.random.key(0), (32, 100))
    x_float = jax.random.normal(jax.random.key(1), (32, 100))
    stats = qep_core.compute_qep_stats(x_quantized, x_float)

    self.assertEqual(stats['hessian'].shape, (32, 32))
    self.assertEqual(stats['hessian_delta'].shape, (32, 32))
    # Hessian should be symmetric.
    self.assertTrue(
        jnp.allclose(stats['hessian'], stats['hessian'].T, atol=1e-5)
    )

  def test_compute_qep_stats_identical_inputs(self):
    """Tests that hessian_delta is zero when inputs are identical."""
    x_quantized = jax.random.normal(jax.random.key(0), (32, 100))
    stats = qep_core.compute_qep_stats(x_quantized, x_quantized)

    self.assertTrue(jnp.allclose(stats['hessian_delta'], 0.0, atol=1e-6))
    # Hessian should match standard compute_hessian.
    expected_h = gptq_core.compute_hessian(x_quantized)
    self.assertTrue(jnp.allclose(stats['hessian'], expected_h, atol=1e-5))

  def test_weight_correct_identity_with_zero_delta(self):
    """Tests that weight_correct with zero H_delta produces no correction."""
    w = jax.random.normal(jax.random.key(0), (64, 32))
    h = jnp.eye(32)
    h_delta = jnp.zeros((32, 32))
    w_corrected = qep_core.weight_correct(w, h, h_delta, correction_factor=0.5)
    self.assertTrue(jnp.allclose(w, w_corrected, atol=1e-5))

  def test_weight_correct_zero_perccorr(self):
    """Tests that perccorr=0.0 produces no correction."""
    w = jax.random.normal(jax.random.key(0), (64, 32))
    # Make H positive definite.
    h_raw = jax.random.normal(jax.random.key(1), (32, 32))
    h = h_raw @ h_raw.T + jnp.eye(32)
    h_delta = jax.random.normal(jax.random.key(2), (32, 32))
    w_corrected = qep_core.weight_correct(w, h, h_delta, correction_factor=0.0)
    self.assertTrue(jnp.allclose(w, w_corrected, atol=1e-5))

  def test_weight_correct_reduces_output_error(self):
    """Tests that weight correction reduces ||W @ X_float - W_corrected @ X_q||."""
    w = jax.random.normal(jax.random.key(0), (64, 128))
    x_float = jax.random.normal(jax.random.key(1), (128, 256))
    noise = 0.1 * jax.random.normal(jax.random.key(2), x_float.shape)
    x_quantized = x_float + noise

    h = x_quantized @ x_quantized.T
    delta = x_float - x_quantized
    h_delta = delta @ x_quantized.T

    w_corrected = qep_core.weight_correct(
        w, h, h_delta, correction_factor=0.5, dampening_factor=0.01
    )

    # The corrected weight should produce output closer to W @ X_float
    # when multiplied by X_q.
    target = w @ x_float
    error_before = jnp.mean((w @ x_quantized - target) ** 2)
    error_after = jnp.mean((w_corrected @ x_quantized - target) ** 2)
    self.assertLess(error_after, error_before)

  @parameterized.named_parameters(
      dict(testcase_name='g128b128', groupsize=128, blocksize=128),
      dict(testcase_name='g256b128', groupsize=256, blocksize=128),
  )
  def test_qep_quantize_weight_matmul_accuracy(self, groupsize, blocksize):
    """Tests that QEP (weight_correct + GPTQ) improves matmul accuracy.

    This mirrors test_quantize_weight but validates the full QEP pipeline:
    weight correction followed by GPTQ quantization should produce better
    matmul accuracy than GPTQ alone when inputs are noisy (quantized).

    Args:
      groupsize: Group size for tiling.
      blocksize: Block size for quantization.
    """

    w = jax.nn.initializers.lecun_normal()(
        jax.random.key(0), (256, 512), jnp.float32
    )
    how = qarray.HowToQuantize(
        qtype=jnp.int8,
        channelwise_axes=[0],
        tiled_axes={1: groupsize},
    )

    # Simulate float inputs and quantized inputs (with noise).
    x_float = jax.random.t(jax.random.key(1), 5, (512, 1024), jnp.float32)
    noise = 0.05 * jax.random.normal(jax.random.key(2), x_float.shape)
    x_quantized = x_float + noise

    # QEP stats from paired inputs.
    stats = qep_core.compute_qep_stats(x_quantized, x_float)
    h_qep = stats['hessian']
    h_delta = stats['hessian_delta']

    # Standard GPTQ: quantize with Hessian from float inputs.
    h_float = gptq_core.compute_hessian(x_float)
    w_gptq = qarray.dequantize(
        jax.jit(
            functools.partial(
                gptq_core.quantize_weight, how=how, blocksize=blocksize
            )
        )(w, h_float)[0]
    )

    # QEP: weight_correct then quantize with Hessian from quantized inputs.
    w_corrected = qep_core.weight_correct(
        w, h_qep, h_delta, correction_factor=0.5, dampening_factor=0.01
    )
    w_qep = qarray.dequantize(
        jax.jit(
            functools.partial(
                gptq_core.quantize_weight, how=how, blocksize=blocksize
            )
        )(w_corrected, h_qep)[0]
    )

    # RTN baseline (no Hessian optimization at all).
    w_rtn = qarray.dequantize(qarray.quantize(w, how))

    # Target: what the original float model would produce with float inputs.
    target = w @ x_float

    # matmul loss: ||W_q @ X_q - W @ X_float|| (the QEP objective).
    mse_rtn = rel_rmse(w_rtn @ x_quantized, target)
    mse_gptq = rel_rmse(w_gptq @ x_quantized, target)
    mse_qep = rel_rmse(w_qep @ x_quantized, target)

    # QEP should beat RTN on the input-compensated objective.
    self.assertGreater(mse_rtn, mse_qep)
    # QEP should beat or match standard GPTQ on the input-compensated
    # objective, since GPTQ doesn't account for input quantization noise.
    self.assertGreater(mse_gptq, mse_qep)


if __name__ == '__main__':
  absltest.main()
