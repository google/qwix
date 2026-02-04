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

from absl.testing import absltest
from absl.testing import parameterized
import flax.linen as nn
import jax
import jax.numpy as jnp
from qwix._src import model as qwix_model
from qwix._src.providers import ptq
from qwix.contrib import awq


class AwqTest(parameterized.TestCase):

  def test_dense_model_linen(self):
    """Test AWQ calibration and quantization on a simple dense model.

    This test verifies that the AWQ pipeline works correctly:
    1. Calibration collects activation statistics
    2. AWQ quantization produces valid parameters with per-channel scales
    3. AwqInferenceProvider applies per-channel compensation during inference
    4. The quantized model runs and produces finite outputs
    """

    class Model(nn.Module):

      @nn.compact
      def __call__(self, x):
        x = nn.Dense(256)(x)
        x = nn.gelu(x)
        x = nn.Dense(128)(x)
        return x

    model = Model()
    x = jax.random.normal(jax.random.key(0), (10, 128))
    # AWQ is sensitive to activation magnitude. To show it beats PTQ, we need
    # activations with varying scales so AWQ can prioritize important channels.
    scale = jnp.ones(128)
    scale = scale.at[0].set(100.0)
    x = x * scale
    variables = model.init(jax.random.key(1), x)

    # Calibration with int4 and tile_size=64 for groupwise quantization.
    rules = [
        awq.AwqRule(module_path='Dense_0', weight_qtype=jnp.int4, tile_size=64)
    ]
    awq_calibration_provider = awq.AwqCalibrationProvider(rules)
    # Note that AwqCalibrationProvider doesn't perform any quantization.
    model_cal = qwix_model.quantize_model(model, awq_calibration_provider)
    _, new_variables = model_cal.apply(variables, x, mutable='quant_stats')
    variables.update(new_variables)
    awq_stats = variables['quant_stats']['Dense_0']['kernel_awq']
    self.assertEqual(awq_stats['count'], 1)
    self.assertEqual(awq_stats['sum_of_act_scale'].shape, (128,))
    fp_y, new_variables = model_cal.apply(variables, x, mutable='quant_stats')
    variables.update(new_variables)
    awq_stats = variables['quant_stats']['Dense_0']['kernel_awq']
    self.assertEqual(awq_stats['count'], 2)

    # Weight quantization with AWQ. We use PtqProvider to get the abstract
    # quantized params tree, but use AwqInferenceProvider for inference.
    ptq_provider = ptq.PtqProvider(rules)
    model_ptq = qwix_model.quantize_model(model, ptq_provider)
    abs_variables = jax.eval_shape(model_ptq.init, jax.random.key(2), x)
    awq_params = awq.quantize_params(
        variables['params'], abs_variables['params'], variables['quant_stats']
    )

    # Verify that AWQ params contain WithAwqScale for the quantized layer.
    self.assertIsInstance(awq_params['Dense_0']['kernel'], awq.WithAwqScale)

    # Use AwqInferenceProvider for inference (applies per-channel compensation).
    awq_inference_provider = awq.AwqInferenceProvider(rules)
    model_awq = qwix_model.quantize_model(model, awq_inference_provider)
    awq_y = model_awq.apply({'params': awq_params}, x)

    # Verify output is valid and has correct shape.
    self.assertEqual(awq_y.shape, fp_y.shape)
    self.assertTrue(jnp.all(jnp.isfinite(awq_y)))

    # Weight transformation using PTQ.
    ptq_params = ptq.quantize_params(
        variables['params'], abs_variables['params']
    )
    model_ptq_inference = qwix_model.quantize_model(model, ptq_provider)
    ptq_y = model_ptq_inference.apply({'params': ptq_params}, x)

    # Verify AWQ outperforms PTQ.
    mae = lambda x, y: jnp.mean(jnp.abs(x - y))
    self.assertLess(mae(fp_y, awq_y), mae(fp_y, ptq_y))

  def test_multiple_calibration_batches(self):
    """Test that AWQ properly averages across calibration batches."""

    class Model(nn.Module):

      @nn.compact
      def __call__(self, x):
        return nn.Dense(64)(x)

    model = Model()
    variables = model.init(jax.random.key(0), jnp.zeros((1, 32)))

    rules = [awq.AwqRule(weight_qtype=jnp.int8)]
    awq_provider = awq.AwqCalibrationProvider(rules)
    model_cal = qwix_model.quantize_model(model, awq_provider)

    # Run 10 calibration batches.
    for i in range(10):
      x = jax.random.normal(jax.random.key(i), (8, 32))
      _, new_variables = model_cal.apply(variables, x, mutable='quant_stats')
      variables.update(new_variables)

    awq_stats = variables['quant_stats']['Dense_0']['kernel_awq']
    self.assertEqual(awq_stats['count'], 10)

  @parameterized.named_parameters(
      dict(testcase_name='int8', qtype=jnp.int8),
      dict(testcase_name='int4', qtype=jnp.int4),
  )
  def test_different_qtypes(self, qtype):
    """Test AWQ with different quantization types."""

    class Model(nn.Module):

      @nn.compact
      def __call__(self, x):
        return nn.Dense(64)(x)

    model = Model()
    x = jax.random.normal(jax.random.key(0), (5, 32))
    variables = model.init(jax.random.key(1), x)

    rules = [awq.AwqRule(weight_qtype=qtype)]
    awq_provider = awq.AwqCalibrationProvider(rules)
    model_cal = qwix_model.quantize_model(model, awq_provider)
    _, new_variables = model_cal.apply(variables, x, mutable='quant_stats')
    variables.update(new_variables)

    ptq_provider = ptq.PtqProvider(rules)
    model_ptq = qwix_model.quantize_model(model, ptq_provider)
    abs_variables = jax.eval_shape(model_ptq.init, jax.random.key(2), x)

    awq_params = awq.quantize_params(
        variables['params'], abs_variables['params'], variables['quant_stats']
    )

    # Verify quantization succeeded.
    self.assertIsNotNone(awq_params)

  def test_multi_layer_model(self):
    """Test AWQ on a model with multiple quantized layers."""

    class Model(nn.Module):

      @nn.compact
      def __call__(self, x):
        x = nn.Dense(128)(x)
        x = nn.relu(x)
        x = nn.Dense(64)(x)
        x = nn.relu(x)
        x = nn.Dense(32)(x)
        return x

    model = Model()
    x = jax.random.normal(jax.random.key(0), (10, 64))
    variables = model.init(jax.random.key(1), x)

    # Quantize all Dense layers with AWQ.
    rules = [awq.AwqRule(weight_qtype=jnp.int8)]
    awq_provider = awq.AwqCalibrationProvider(rules)
    model_cal = qwix_model.quantize_model(model, awq_provider)

    # Run multiple calibration batches.
    for i in range(5):
      x_cal = jax.random.normal(jax.random.key(i + 10), (10, 64))
      _, new_variables = model_cal.apply(
          variables, x_cal, mutable='quant_stats'
      )
      variables.update(new_variables)

    # Verify stats collected for all layers.
    for layer_name in ['Dense_0', 'Dense_1', 'Dense_2']:
      self.assertIn(layer_name, variables['quant_stats'])
      self.assertIn('kernel_awq', variables['quant_stats'][layer_name])

    # Quantize with AWQ.
    ptq_provider = ptq.PtqProvider(rules)
    model_ptq = qwix_model.quantize_model(model, ptq_provider)
    abs_variables = jax.eval_shape(model_ptq.init, jax.random.key(2), x)
    awq_params = awq.quantize_params(
        variables['params'], abs_variables['params'], variables['quant_stats']
    )

    # Verify model runs with AWQ params using AwqInferenceProvider.
    awq_inference_provider = awq.AwqInferenceProvider(rules)
    model_awq = qwix_model.quantize_model(model, awq_inference_provider)
    awq_y = model_awq.apply({'params': awq_params}, x)
    self.assertEqual(awq_y.shape, (10, 32))

  def test_partial_quantization(self):
    """Test AWQ with only some layers quantized."""

    class Model(nn.Module):

      @nn.compact
      def __call__(self, x):
        x = nn.Dense(128, name='dense1')(x)
        x = nn.relu(x)
        x = nn.Dense(64, name='dense2')(x)
        return x

    model = Model()
    x = jax.random.normal(jax.random.key(0), (5, 32))
    variables = model.init(jax.random.key(1), x)

    # Only quantize dense1 with AWQ.
    rules = [awq.AwqRule(module_path='dense1', weight_qtype=jnp.int8)]
    awq_provider = awq.AwqCalibrationProvider(rules)
    model_cal = qwix_model.quantize_model(model, awq_provider)
    _, new_variables = model_cal.apply(variables, x, mutable='quant_stats')
    variables.update(new_variables)

    # Verify only dense1 has AWQ stats.
    self.assertIn('dense1', variables['quant_stats'])
    self.assertIn('kernel_awq', variables['quant_stats']['dense1'])
    # dense2 should not have AWQ stats.
    self.assertNotIn('dense2', variables.get('quant_stats', {}))

    # Quantize with AWQ.
    ptq_provider = ptq.PtqProvider(rules)
    model_ptq = qwix_model.quantize_model(model, ptq_provider)
    abs_variables = jax.eval_shape(model_ptq.init, jax.random.key(2), x)
    awq_params = awq.quantize_params(
        variables['params'], abs_variables['params'], variables['quant_stats']
    )

    # Verify model runs with AwqInferenceProvider.
    awq_inference_provider = awq.AwqInferenceProvider(rules)
    model_awq = qwix_model.quantize_model(model, awq_inference_provider)
    awq_y = model_awq.apply({'params': awq_params}, x)
    self.assertEqual(awq_y.shape, (5, 64))

  def test_awq_outperforms_ptq(self):
    """Test that AWQ provides better accuracy than plain PTQ.

    This test compares AWQ and PTQ on the same model with the same quantization
    settings. AWQ should have lower output error because it uses activation-
    aware scaling to protect salient weight channels.

    Note: The core AWQ algorithm benefits are validated in awq_core_test.py with
    parameterized tests. This integration test verifies the end-to-end pipeline.
    """

    class Model(nn.Module):

      @nn.compact
      def __call__(self, x):
        x = nn.Dense(256)(x)
        x = nn.gelu(x)
        x = nn.Dense(128)(x)
        return x

    model = Model()
    # Use t-distribution for heavy-tailed activations (more salient channels).
    # Use 512 input features to match core test conditions.
    x = jax.random.t(jax.random.key(0), 5, (10, 512), jnp.float32)
    variables = model.init(jax.random.key(1), x)

    # Get floating-point output.
    fp_y = model.apply(variables, x)

    # Quantize only the first layer with int4 and groupsize 128.
    rules = [
        awq.AwqRule(module_path='Dense_0', weight_qtype=jnp.int4, tile_size=128)
    ]
    awq_calibration_provider = awq.AwqCalibrationProvider(rules)
    model_cal = qwix_model.quantize_model(model, awq_calibration_provider)
    _, new_variables = model_cal.apply(variables, x, mutable='quant_stats')
    variables.update(new_variables)

    # Get abstract quantized params tree.
    ptq_provider = ptq.PtqProvider(rules)
    model_ptq = qwix_model.quantize_model(model, ptq_provider)
    abs_variables = jax.eval_shape(model_ptq.init, jax.random.key(2), x)

    # Quantize with AWQ.
    awq_params = awq.quantize_params(
        variables['params'], abs_variables['params'], variables['quant_stats']
    )

    # Quantize with plain PTQ.
    ptq_params = ptq.quantize_params(
        variables['params'], abs_variables['params']
    )

    # Run AWQ inference with per-channel compensation.
    awq_inference_provider = awq.AwqInferenceProvider(rules)
    model_awq = qwix_model.quantize_model(model, awq_inference_provider)
    awq_y = model_awq.apply({'params': awq_params}, x)

    # Run PTQ inference.
    ptq_y = model_ptq.apply({'params': ptq_params}, x)

    # Compare output errors.
    mae = lambda a, b: jnp.mean(jnp.abs(a - b))
    awq_error = mae(fp_y, awq_y)
    ptq_error = mae(fp_y, ptq_y)

    # AWQ should be better than PTQ in terms of output error.
    self.assertLess(
        awq_error,
        ptq_error,
        msg=(
            f'AWQ error ({awq_error:.6f}) should be less than PTQ error'
            f' ({ptq_error:.6f})'
        ),
    )

  def test_einsum_model_linen(self):
    """Tests that an einsum model is quantized correctly with AWQ."""

    class EinsumModel(nn.Module):

      @nn.compact
      def __call__(self, x):
        x = nn.Einsum(shape=(x.shape[-1], 128), einsum_str='bi,io->bo')(x)
        x = nn.gelu(x)
        x = nn.Einsum(shape=(128, 64), einsum_str='bi,io->bo')(x)
        return x

    model = EinsumModel()
    x = jax.random.normal(jax.random.key(0), (5, 32))
    # AWQ is sensitive to activation magnitude. To show it beats PTQ, we need
    # activations with varying scales so AWQ can prioritize important channels.
    # We maintain the same shape but scale channels differently.
    scale = jnp.ones(32)
    scale = scale.at[0].set(100.0)
    x = x * scale
    variables = model.init(jax.random.key(1), x)

    # 1. Calibration
    rules = [awq.AwqRule(module_path='Einsum_0', weight_qtype=jnp.int8)]
    awq_calibration_provider = awq.AwqCalibrationProvider(rules)

    # Use @jax.jit here to ensure disable_jit works under compilation.
    @jax.jit
    def calibrate_step(v, x):
      m = qwix_model.quantize_model(model, awq_calibration_provider)
      return m.apply(v, x, mutable='quant_stats')

    # Running the first calibration batch.
    _, new_variables = calibrate_step(variables, x)
    variables.update(new_variables)
    awq_stats = variables['quant_stats']['Einsum_0']['kernel_awq']
    self.assertEqual(awq_stats['count'], 1)
    self.assertEqual(awq_stats['sum_of_act_scale'].shape, (32,))
    # Running the second calibration batch to test accumulation.
    fp_y, new_variables = calibrate_step(variables, x)
    variables.update(new_variables)
    awq_stats = variables['quant_stats']['Einsum_0']['kernel_awq']
    self.assertEqual(awq_stats['count'], 2)

    # 2. Model preparation for inference.
    # Use the same PtqProvider to get the abstract quantized params tree.
    ptq_provider = ptq.PtqProvider(rules)
    model_ptq = qwix_model.quantize_model(model, ptq_provider)
    abs_variables = jax.eval_shape(model_ptq.init, jax.random.key(2), x)

    # 3. Actual weight transformation using AWQ.
    awq_params = awq.quantize_params(
        variables['params'], abs_variables['params'], variables['quant_stats']
    )

    # Run AWQ inference.
    awq_inference_provider = awq.AwqInferenceProvider(rules)
    model_awq = qwix_model.quantize_model(model, awq_inference_provider)
    awq_y = model_awq.apply({'params': awq_params}, x)

    # Weight transformation using PTQ.
    ptq_params = ptq.quantize_params(
        variables['params'], abs_variables['params']
    )
    model_ptq_inference = qwix_model.quantize_model(model, ptq_provider)
    ptq_y = model_ptq_inference.apply({'params': ptq_params}, x)

    # Verify AWQ outperforms standard PTQ.
    mae = lambda a, b: jnp.mean(jnp.abs(a - b))
    self.assertLess(mae(fp_y, awq_y), mae(fp_y, ptq_y))

  def test_mixed_model_safety(self):
    """Tests that unsupported einsums (like Attention) are safely ignored."""

    class MixedModel(nn.Module):

      @nn.compact
      def __call__(self, x):
        # Valid Operation: Linear Layer (Matrix Mult)
        k = self.param('key', nn.initializers.lecun_normal(), (32, 48))
        k_out = jnp.einsum('bi,io->bo', x, k)

        # Invalid Operation: Attention-style Batch Dot
        q = jax.random.normal(jax.random.key(0), x.shape)
        attn = jnp.einsum('bi,bi->b', q, x)

        # Invalid Operation: Transpose
        transposed = jnp.einsum('bi->ib', x)

        return k_out, attn, transposed

    model = MixedModel()
    x = jax.random.normal(jax.random.key(0), (5, 32))
    variables = model.init(jax.random.key(1), x)

    rules = [awq.AwqRule(module_path='.*', weight_qtype=jnp.int8)]
    awq_calibration_provider = awq.AwqCalibrationProvider(rules)

    @jax.jit
    def calibrate_step(v, x):
      m = qwix_model.quantize_model(model, awq_calibration_provider)
      return m.apply(v, x, mutable='quant_stats')

    _, new_variables = calibrate_step(variables, x)
    variables.update(new_variables)

    # The dictionary should only contain the keys for the valid modules.
    self.assertIn('key_awq', variables['quant_stats'])
    awq_stats = variables['quant_stats']['key_awq']
    self.assertEqual(awq_stats['count'], 1)

    # Only 1 variable should be quantized (key_awq).
    # Note: 'key' stores stats as 'key_awq'.
    self.assertLen(variables['quant_stats'], 1)


if __name__ == '__main__':
  absltest.main()
