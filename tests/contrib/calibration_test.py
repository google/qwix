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
"""Tests for common calibration utilities."""

from absl.testing import absltest
from absl.testing import parameterized
import flax.linen as nn
import jax
import jax.numpy as jnp
from qwix._src import model as qwix_model
from qwix._src.core import qarray
from qwix._src.providers import ptq
from qwix.contrib import calibration
from qwix.contrib import gptq
from qwix.contrib import gptq_core


class NormalizeWeightTest(parameterized.TestCase):

  def test_basic_shape(self):
    w = jnp.arange(2 * 3 * 4).reshape(2, 3, 4)
    w2, restore_shape = calibration.normalize_weight(w, 1)
    self.assertEqual(w2.shape, (8, 3))
    w3 = restore_shape(w2)
    self.assertEqual(w3.shape, (2, 3, 4))
    self.assertTrue(jnp.all(w == w3))

  def test_contraction_axis_0(self):
    w = jnp.arange(3 * 5).reshape(3, 5)
    w2, restore_shape = calibration.normalize_weight(w, 0)
    # (5, 3) after moveaxis, then reshape to (5, 3).
    self.assertEqual(w2.shape, (5, 3))
    w3 = restore_shape(w2)
    self.assertEqual(w3.shape, (3, 5))
    self.assertTrue(jnp.all(w == w3))

  def test_contraction_axis_last(self):
    w = jnp.arange(2 * 4 * 6).reshape(2, 4, 6)
    w2, restore_shape = calibration.normalize_weight(w, 2)
    # axis 2 is already last, so (2, 4, 6) -> reshape to (8, 6).
    self.assertEqual(w2.shape, (8, 6))
    w3 = restore_shape(w2)
    self.assertEqual(w3.shape, (2, 4, 6))
    self.assertTrue(jnp.all(w == w3))


class QuantizeParamsWithCalibrationTest(parameterized.TestCase):

  def _setup_model_and_stats(self, rules):
    """Helper to create a model, calibrate with GPTQ, and return all pieces."""

    class DenseModel(nn.Module):

      @nn.compact
      def __call__(self, x):
        x = nn.Dense(128)(x)
        x = nn.gelu(x)
        x = nn.Dense(64)(x)
        return x

    model = DenseModel()
    x = jax.random.normal(jax.random.key(0), (5, 32))
    variables = model.init(jax.random.key(1), x)

    # Calibrate.
    cal_provider = gptq.GptqCalibrationProvider(rules)
    cal_model = qwix_model.quantize_model(model, cal_provider)
    _, new_vars = cal_model.apply(variables, x, mutable='quant_stats')
    variables.update(new_vars)

    # Get abstract quantized params.
    ptq_provider = ptq.PtqProvider(rules)
    ptq_model = qwix_model.quantize_model(model, ptq_provider)
    abs_variables = jax.eval_shape(ptq_model.init, jax.random.key(2), x)

    return model, ptq_model, x, variables, abs_variables

  def test_delegates_to_quantize_fn(self):
    """Tests that quantize_fn is called with a properly constructed context."""

    rules = [gptq.GptqRule(module_path='Dense_0', weight_qtype=jnp.int8)]
    _, ptq_model, x, variables, abs_variables = self._setup_model_and_stats(
        rules
    )

    captured_contexts = []

    def mock_quantize(ctx):
      captured_contexts.append(ctx)
      # Just do PTQ quantization.
      w = qarray.quantize(ctx.weight, ctx.how)
      w = ctx.restore_shape(w)
      return ctx.abs_w.replace(array=w)

    result = calibration.quantize_params_with_calibration(
        variables['params'],
        abs_variables['params'],
        variables['quant_stats'],
        '_gptq',
        mock_quantize,
    )

    # Should have called quantize_fn once (Dense_0's kernel).
    self.assertLen(captured_contexts, 1)
    ctx = captured_contexts[0]
    self.assertEqual(ctx.weight.ndim, 2)
    self.assertIn('hessian', ctx.calibration_stats)
    self.assertEqual(ctx.path[-1], 'kernel')

    # Result should be a valid param tree for the PTQ model.
    y = ptq_model.apply({'params': result}, x)
    self.assertEqual(y.shape, (5, 64))

  def test_ptq_fallback_for_unmatched_params(self):
    """Tests that params without calibration stats get PTQ quantization."""
    # Only quantize Dense_0, so Dense_1 should fall through to PTQ.
    rules = [gptq.GptqRule(module_path='Dense_0', weight_qtype=jnp.int8)]
    _, ptq_model, x, variables, abs_variables = self._setup_model_and_stats(
        rules
    )

    call_count = [0]

    def mock_quantize(prepared):
      call_count[0] += 1
      w = qarray.quantize(prepared.weight, prepared.how)
      w = prepared.restore_shape(w)
      return prepared.abs_w.replace(array=w)

    result = calibration.quantize_params_with_calibration(
        variables['params'],
        abs_variables['params'],
        variables['quant_stats'],
        '_gptq',
        mock_quantize,
    )

    # Only Dense_0/kernel should be handled by quantize_fn.
    self.assertEqual(call_count[0], 1)

    # The full result should still be usable (Dense_1 handled by PTQ fallback).
    y = ptq_model.apply({'params': result}, x)
    self.assertEqual(y.shape, (5, 64))

  def test_matches_gptq_quantize_params(self):
    """Tests that the shared utility produces identical results to gptq."""
    rules = [gptq.GptqRule(module_path='Dense_0', weight_qtype=jnp.int8)]
    _, ptq_model, x, variables, abs_variables = self._setup_model_and_stats(
        rules
    )

    # Use the same logic as gptq.quantize_params._quantize.
    def gptq_quantize(ctx):
      hessian = ctx.calibration_stats['hessian']
      w = gptq_core.quantize_weight(
          ctx.weight,
          hessian,
          ctx.how,
          blocksize=128,
          percdamp=0.01,
      )[0]
      w = ctx.restore_shape(w)
      return ctx.abs_w.replace(array=w)

    shared_result = calibration.quantize_params_with_calibration(
        variables['params'],
        abs_variables['params'],
        variables['quant_stats'],
        '_gptq',
        gptq_quantize,
    )
    direct_result = gptq.quantize_params(
        variables['params'],
        abs_variables['params'],
        variables['quant_stats'],
    )

    # Both should produce the same model output.
    y_shared = ptq_model.apply({'params': shared_result}, x)
    y_direct = ptq_model.apply({'params': direct_result}, x)
    self.assertTrue(jnp.allclose(y_shared, y_direct))


if __name__ == '__main__':
  absltest.main()
