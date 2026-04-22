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
from flax import nnx
import flax.linen as nn
import jax
import jax.numpy as jnp
from qwix._src import flax_util
from qwix._src import model as qwix_model
from qwix._src.providers import ptq
from qwix.contrib import smooth_quant as sq


class SqTest(parameterized.TestCase):

  def test_dense_model_linen(self):
    """Test SQ calibration and quantization on a simple dense model.

    This test verifies that the SQ pipeline works correctly:
    1. Calibration collects activation statistics
    2. SQ quantization produces valid parameters with per-channel scales
    3. SqInferenceProvider applies per-channel compensation during inference
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
    # SQ is sensitive to activation magnitude. To show it beats PTQ, we need
    # activations with varying scales so SQ can prioritize important channels.
    scale = jnp.ones(128)
    scale = scale.at[0].set(100.0)
    x = x * scale
    variables = model.init(jax.random.key(1), x)

    # Calibration with int4 and tile_size=64 for groupwise quantization.
    rules = [
        sq.SqRule(
            module_path='Dense_0',
            weight_qtype=jnp.int4,
            act_qtype=jnp.int4,
            tile_size=64,
        )
    ]
    sq_calibration_provider = sq.SqCalibrationProvider(rules)
    # Note that SqCalibrationProvider doesn't perform any quantization.
    model_cal = qwix_model.quantize_model(model, sq_calibration_provider)
    _, new_variables = model_cal.apply(variables, x, mutable='quant_stats')
    variables.update(new_variables)
    sq_stats = variables['quant_stats']['Dense_0']['kernel_sq']
    self.assertEqual(sq_stats['count'], 1)
    self.assertEqual(sq_stats['sum_of_sq_scale'].shape, (128,))
    fp_y, new_variables = model_cal.apply(variables, x, mutable='quant_stats')
    variables.update(new_variables)
    sq_stats = variables['quant_stats']['Dense_0']['kernel_sq']
    self.assertEqual(sq_stats['count'], 2)

    # Weight quantization with SQ. We use PtqProvider to get the abstract
    # quantized params tree, but use SqInferenceProvider for inference.
    ptq_provider = ptq.PtqProvider(rules)
    model_ptq = qwix_model.quantize_model(model, ptq_provider)
    abs_variables = jax.eval_shape(model_ptq.init, jax.random.key(2), x)
    sq_params = sq.quantize_params(
        variables['params'], abs_variables['params'], variables['quant_stats']
    )

    # Verify that SQ params contain WithSqScale for the quantized layer.
    self.assertIsInstance(sq_params['Dense_0']['kernel'], sq.WithSqScale)

    # Use SqInferenceProvider for inference (applies per-channel compensation).
    sq_inference_provider = sq.SqInferenceProvider(rules)
    model_sq = qwix_model.quantize_model(model, sq_inference_provider)
    sq_y = model_sq.apply({'params': sq_params}, x)

    # Verify output is valid and has correct shape.
    self.assertEqual(sq_y.shape, fp_y.shape)
    self.assertTrue(jnp.all(jnp.isfinite(sq_y)))

    # Weight transformation using PTQ.
    ptq_params = ptq.quantize_params(
        variables['params'], abs_variables['params']
    )
    model_ptq_inference = qwix_model.quantize_model(model, ptq_provider)
    ptq_y = model_ptq_inference.apply({'params': ptq_params}, x)

    # Verify SQ outperforms PTQ.
    mae = lambda x, y: jnp.mean(jnp.abs(x - y))
    self.assertLess(mae(fp_y, sq_y), mae(fp_y, ptq_y))

  def test_dense_model_nnx(self):
    """Test SQ calibration and quantization on a simple dense model with nnx."""

    class Model(nnx.Module):

      def __init__(self, *, rngs=nnx.Rngs(0)):
        self.dense1 = nnx.Linear(128, 256, rngs=rngs)
        self.dense2 = nnx.Linear(256, 128, rngs=rngs)

      def __call__(self, x):
        x = self.dense1(x)
        x = nn.gelu(x)
        x = self.dense2(x)
        return x

    model = Model()
    x = jax.random.normal(jax.random.key(0), (10, 128))
    # SQ is sensitive to activation magnitude. To show it beats PTQ, we need
    # activations with varying scales so SQ can prioritize important channels.
    scale = jnp.ones(128)
    scale = scale.at[0].set(100.0)
    x = x * scale

    # Calibration with int4 and tile_size=64 for groupwise quantization.
    rules = [sq.SqRule(weight_qtype=jnp.int4, act_qtype=jnp.int4, tile_size=64)]
    sq_calibration_provider = sq.SqCalibrationProvider(rules)
    # Note that SqCalibrationProvider doesn't perform any quantization.
    model_cal = qwix_model.quantize_model(model, sq_calibration_provider, x)

    # Check that calibration stats are updated correctly.
    _ = model_cal(x)
    sq_stats = model_cal.dense1.kernel_sq
    self.assertEqual(sq_stats['count'], 1)
    self.assertEqual(sq_stats['sum_of_sq_scale'].shape, (128,))

    fp_y = model_cal(x)
    sq_stats = model_cal.dense1.kernel_sq
    self.assertEqual(sq_stats['count'], 2)

    # Weight quantization with SQ. We use PtqProvider to get the abstract
    # quantized params tree, but use SqInferenceProvider for inference.
    ptq_provider = ptq.PtqProvider(rules)
    model_ptq = qwix_model.quantize_model(model, ptq_provider, x)
    state = nnx.to_pure_dict(nnx.state(model_cal, nnx.Param))

    def extract_fn_quant_stats(x):
      if isinstance(x, flax_util.QuantStat):
        return x
      elif isinstance(x, nnx.Variable):
        return x.get_value()
      else:
        return x

    quant_stats = nnx.to_pure_dict(
        nnx.state(model_cal, flax_util.QuantStat),
        extract_fn=extract_fn_quant_stats,
    )

    sq_params = sq.quantize_params(state, model_ptq, quant_stats)

    # Verify that SQ params contain WithSqScale for the quantized layer.
    self.assertIsInstance(sq_params['dense1']['kernel'], sq.WithSqScale)

    # Use SqInferenceProvider for inference (applies per-channel compensation).
    sq_inference_provider = sq.SqInferenceProvider(rules)
    model_sq = qwix_model.quantize_model(model, sq_inference_provider, x)
    sq_y = model_sq(x)

    # Verify output is valid and has correct shape.
    self.assertEqual(sq_y.shape, fp_y.shape)
    self.assertTrue(jnp.all(jnp.isfinite(sq_y)))

    # Weight transformation using PTQ.
    ptq_y = model_ptq(x)

    # Verify SQ outperforms PTQ.
    mae = lambda x, y: jnp.mean(jnp.abs(x - y))
    self.assertLessEqual(mae(fp_y, sq_y), mae(fp_y, ptq_y))

  def test_multiple_calibration_batches(self):
    """Test that SQ properly averages across calibration batches."""

    class Model(nn.Module):

      @nn.compact
      def __call__(self, x):
        return nn.Dense(64)(x)

    model = Model()
    variables = model.init(jax.random.key(0), jnp.zeros((1, 32)))

    rules = [sq.SqRule(weight_qtype=jnp.int8, act_qtype=jnp.int8)]
    sq_provider = sq.SqCalibrationProvider(rules)
    model_cal = qwix_model.quantize_model(model, sq_provider)

    # Run 10 calibration batches.
    for i in range(10):
      x = jax.random.normal(jax.random.key(i), (8, 32))
      _, new_variables = model_cal.apply(variables, x, mutable='quant_stats')
      variables.update(new_variables)

    sq_stats = variables['quant_stats']['Dense_0']['kernel_sq']
    self.assertEqual(sq_stats['count'], 10)

  @parameterized.named_parameters(
      dict(testcase_name='int8', qtype=jnp.int8),
      dict(testcase_name='int4', qtype=jnp.int4),
  )
  def test_different_qtypes(self, qtype):
    """Test SQ with different quantization types."""

    class Model(nn.Module):

      @nn.compact
      def __call__(self, x):
        return nn.Dense(64)(x)

    model = Model()
    x = jax.random.normal(jax.random.key(0), (5, 32))
    variables = model.init(jax.random.key(1), x)

    rules = [sq.SqRule(weight_qtype=qtype, act_qtype=qtype)]
    sq_provider = sq.SqCalibrationProvider(rules)
    model_cal = qwix_model.quantize_model(model, sq_provider)
    _, new_variables = model_cal.apply(variables, x, mutable='quant_stats')
    variables.update(new_variables)

    ptq_provider = ptq.PtqProvider(rules)
    model_ptq = qwix_model.quantize_model(model, ptq_provider)
    abs_variables = jax.eval_shape(model_ptq.init, jax.random.key(2), x)

    sq_params = sq.quantize_params(
        variables['params'], abs_variables['params'], variables['quant_stats']
    )

    # Verify quantization succeeded.
    self.assertIsNotNone(sq_params)

  def test_multi_layer_model(self):
    """Test SQ on a model with multiple quantized layers."""

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

    # Quantize all Dense layers with SQ.
    rules = [sq.SqRule(weight_qtype=jnp.int8, act_qtype=jnp.int8)]
    sq_provider = sq.SqCalibrationProvider(rules)
    model_cal = qwix_model.quantize_model(model, sq_provider)

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
      self.assertIn('kernel_sq', variables['quant_stats'][layer_name])

    # Quantize with SQ.
    ptq_provider = ptq.PtqProvider(rules)
    model_ptq = qwix_model.quantize_model(model, ptq_provider)
    abs_variables = jax.eval_shape(model_ptq.init, jax.random.key(2), x)
    sq_params = sq.quantize_params(
        variables['params'], abs_variables['params'], variables['quant_stats']
    )

    # Verify model runs with SQ params using SqInferenceProvider.
    sq_inference_provider = sq.SqInferenceProvider(rules)
    model_sq = qwix_model.quantize_model(model, sq_inference_provider)
    sq_y = model_sq.apply({'params': sq_params}, x)
    self.assertEqual(sq_y.shape, (10, 32))

  def test_partial_quantization(self):
    """Test SQ with only some layers quantized."""

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

    # Only quantize dense1 with SQ.
    rules = [
        sq.SqRule(
            module_path='dense1', weight_qtype=jnp.int8, act_qtype=jnp.int8
        )
    ]
    sq_provider = sq.SqCalibrationProvider(rules)
    model_cal = qwix_model.quantize_model(model, sq_provider)
    _, new_variables = model_cal.apply(variables, x, mutable='quant_stats')
    variables.update(new_variables)

    # Verify only dense1 has SQ stats.
    self.assertIn('dense1', variables['quant_stats'])
    self.assertIn('kernel_sq', variables['quant_stats']['dense1'])
    # dense2 should not have SQ stats.
    self.assertNotIn('dense2', variables.get('quant_stats', {}))

    # Quantize with SQ.
    ptq_provider = ptq.PtqProvider(rules)
    model_ptq = qwix_model.quantize_model(model, ptq_provider)
    abs_variables = jax.eval_shape(model_ptq.init, jax.random.key(2), x)
    sq_params = sq.quantize_params(
        variables['params'], abs_variables['params'], variables['quant_stats']
    )

    # Verify model runs with SqInferenceProvider.
    sq_inference_provider = sq.SqInferenceProvider(rules)
    model_sq = qwix_model.quantize_model(model, sq_inference_provider)
    sq_y = model_sq.apply({'params': sq_params}, x)
    self.assertEqual(sq_y.shape, (5, 64))


if __name__ == '__main__':
  absltest.main()
