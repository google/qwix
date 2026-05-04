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
from flax import nnx
import flax.linen as nn
import jax
import jax.numpy as jnp
from qwix._src import flax_util
from qwix._src import model as qwix_model
from qwix._src.providers import ptq
from qwix.contrib import gptq


class GptqTest(parameterized.TestCase):

  def test_dense_model_linen(self):
    """Tests that a dense model is quantized correctly."""

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

    # 1. Calibration.
    rules = [gptq.GptqRule(module_path='Dense_0', weight_qtype=jnp.int8)]
    gptq_calibration_provider = gptq.GptqCalibrationProvider(rules)
    # Note that GptqCalibrationProvider doesn't perform any quantization.
    model = qwix_model.quantize_model(model, gptq_calibration_provider)
    # Running the first calibration batch.
    _, new_variables = model.apply(variables, x, mutable='quant_stats')
    variables.update(new_variables)
    gptq_stats = variables['quant_stats']['Dense_0']['kernel_gptq']
    self.assertEqual(gptq_stats['count'], 1)
    self.assertEqual(gptq_stats['sum_of_hessian'].shape, (32, 32))
    # Running the second calibration batch to test accumulation.
    fp_y, new_variables = model.apply(variables, x, mutable='quant_stats')
    variables.update(new_variables)
    gptq_stats = variables['quant_stats']['Dense_0']['kernel_gptq']
    self.assertEqual(gptq_stats['count'], 2)

    # 2. Model preparation for inference.
    # Use the same PtqProvider to get the abstract quantized params tree.
    ptq_provider = ptq.PtqProvider(rules)
    model = qwix_model.quantize_model(model, ptq_provider)
    abs_variables = jax.eval_shape(model.init, jax.random.key(2), x)

    # 3. Actual weight transformation using GPTQ.
    gptq_params = gptq.quantize_params(
        variables['params'], abs_variables['params'], variables['quant_stats']
    )
    gptq_y = model.apply({'params': gptq_params}, x)
    # Weight transformation using PTQ.
    ptq_params = ptq.quantize_params(
        variables['params'], abs_variables['params']
    )
    ptq_y = model.apply({'params': ptq_params}, x)

    # GPTQ should be better than PTQ in terms of the output error.
    mae = lambda x, y: jnp.mean(jnp.abs(x - y))
    self.assertLess(mae(fp_y, gptq_y), mae(fp_y, ptq_y))

  def test_dense_model_nnx(self):
    """Tests that a dense model is quantized correctly with NNX."""

    class Model(nnx.Module):

      def __init__(self, *, rngs=nnx.Rngs(0)):
        self.dense1 = nnx.Linear(32, 128, rngs=rngs)
        self.dense2 = nnx.Linear(128, 64, rngs=rngs)

      def __call__(self, x):
        x = self.dense1(x)
        x = nn.gelu(x)
        x = self.dense2(x)
        return x

    model = Model()
    x = jax.random.normal(jax.random.key(0), (5, 32))

    # 1. Calibration.
    rules = [gptq.GptqRule(module_path='dense1', weight_qtype=jnp.int8)]
    gptq_calibration_provider = gptq.GptqCalibrationProvider(rules)
    model_cal = qwix_model.quantize_model(model, gptq_calibration_provider, x)

    # Running the first calibration batch.
    _ = model_cal(x)
    gptq_stats = model_cal.dense1.kernel_gptq
    self.assertEqual(gptq_stats['count'], 1)
    self.assertEqual(gptq_stats['sum_of_hessian'].shape, (32, 32))

    # Running the second calibration batch to test accumulation.
    fp_y = model_cal(x)
    gptq_stats = model_cal.dense1.kernel_gptq
    self.assertEqual(gptq_stats['count'], 2)

    # 2. Model preparation for inference.
    ptq_provider = ptq.PtqProvider(rules)
    model_ptq = qwix_model.quantize_model(model, ptq_provider, x)
    # Get PTQ baseline output.
    ptq_y = model_ptq(x)

    # 3. Actual weight transformation using GPTQ.
    state = nnx.to_pure_dict(nnx.state(model_cal, nnx.Param))
    quant_stats = nnx.to_pure_dict(nnx.state(model_cal, flax_util.QuantStat))
    gptq_params = gptq.quantize_params(state, model_ptq, quant_stats)
    # Apply GPTQ params to model.
    nnx.update(model_ptq, gptq_params)
    gptq_y = model_ptq(x)

    # GPTQ should be better than PTQ in terms of the output error.
    mae = lambda x, y: jnp.mean(jnp.abs(x - y))
    self.assertLessEqual(mae(fp_y, gptq_y), mae(fp_y, ptq_y))

  def test_einsum_model_linen(self):
    """Tests that an einsum model is quantized correctly."""

    class EinsumModel(nn.Module):

      @nn.compact
      def __call__(self, x):
        x = nn.Einsum(shape=(x.shape[-1], 128), einsum_str='bi,io->bo')(x)
        x = nn.gelu(x)
        x = nn.Einsum(shape=(128, 64), einsum_str='bi,io->bo')(x)
        return x

    model = EinsumModel()
    x = jax.random.normal(jax.random.key(0), (5, 32))
    variables = model.init(jax.random.key(1), x)

    # 1. Calibration
    rules = [gptq.GptqRule(module_path='Einsum_0', weight_qtype=jnp.int8)]
    gptq_calibration_provider = gptq.GptqCalibrationProvider(rules)

    # Use @jax.jit here to ensure disable_jit works under compilation.
    @jax.jit
    def calibrate_step(v, x):
      # Note that GptqCalibrationProvider doesn't perform any quantization.
      m = qwix_model.quantize_model(model, gptq_calibration_provider)
      return m.apply(v, x, mutable='quant_stats')

    # Running the first calibration batch.
    _, new_variables = calibrate_step(variables, x)
    variables.update(new_variables)
    gptq_stats = variables['quant_stats']['Einsum_0']['kernel_gptq']
    self.assertEqual(gptq_stats['count'], 1)
    self.assertEqual(gptq_stats['sum_of_hessian'].shape, (32, 32))
    # Running the second calibration batch to test accumulation.
    fp_y, new_variables = calibrate_step(variables, x)
    variables.update(new_variables)
    gptq_stats = variables['quant_stats']['Einsum_0']['kernel_gptq']
    self.assertEqual(gptq_stats['count'], 2)

    # 2. Model preparation for inference.
    # Use the same PtqProvider to get the abstract quantized params tree.
    ptq_provider = ptq.PtqProvider(rules)
    model = qwix_model.quantize_model(model, ptq_provider)
    abs_variables = jax.eval_shape(model.init, jax.random.key(2), x)

    # 3. Actual weight transformation using GPTQ.
    gptq_params = gptq.quantize_params(
        variables['params'], abs_variables['params'], variables['quant_stats']
    )
    gptq_y = model.apply({'params': gptq_params}, x)
    # Weight transformation using PTQ.
    ptq_params = ptq.quantize_params(
        variables['params'], abs_variables['params']
    )
    ptq_y = model.apply({'params': ptq_params}, x)

    # GPTQ should generally match float precision better than blind PTQ
    mae = lambda a, b: jnp.mean(jnp.abs(a - b))
    self.assertLess(mae(fp_y, gptq_y), mae(fp_y, ptq_y))

  def test_einsum_model_nnx(self):
    """Tests that an einsum model is quantized correctly with NNX."""

    class EinsumModel(nnx.Module):

      def __init__(self, *, rngs=nnx.Rngs(0)):
        self.einsum1 = nnx.Einsum('bi,io->bo', (32, 128), rngs=rngs)
        self.einsum2 = nnx.Einsum('bi,io->bo', (128, 64), rngs=rngs)

      def __call__(self, x):
        x = self.einsum1(x)
        x = nn.gelu(x)
        x = self.einsum2(x)
        return x

    model = EinsumModel()
    x = jax.random.normal(jax.random.key(0), (5, 32))

    # 1. Calibration
    rules = [gptq.GptqRule(module_path='einsum1', weight_qtype=jnp.int8)]
    gptq_calibration_provider = gptq.GptqCalibrationProvider(rules)
    model_cal = qwix_model.quantize_model(model, gptq_calibration_provider, x)

    # Running the first calibration batch.
    _ = model_cal(x)
    gptq_stats = model_cal.einsum1.kernel_gptq
    self.assertEqual(gptq_stats['count'], 1)
    self.assertEqual(gptq_stats['sum_of_hessian'].shape, (32, 32))

    # Running the second calibration batch to test accumulation.
    fp_y = model_cal(x)
    gptq_stats = model_cal.einsum1.kernel_gptq
    self.assertEqual(gptq_stats['count'], 2)

    # 2. Model preparation for inference.
    ptq_provider = ptq.PtqProvider(rules)
    model_ptq = qwix_model.quantize_model(model, ptq_provider, x)
    # Get PTQ baseline output.
    ptq_y = model_ptq(x)

    # 3. Actual weight transformation using GPTQ.
    state = nnx.to_pure_dict(nnx.state(model_cal, nnx.Param))
    quant_stats = nnx.to_pure_dict(nnx.state(model_cal, flax_util.QuantStat))
    gptq_params = gptq.quantize_params(state, model_ptq, quant_stats)
    # Apply GPTQ params to model.
    nnx.update(model_ptq, gptq_params)
    gptq_y = model_ptq(x)

    # GPTQ should generally match float precision better than blind PTQ
    mae = lambda a, b: jnp.mean(jnp.abs(a - b))
    self.assertLessEqual(mae(fp_y, gptq_y), mae(fp_y, ptq_y))

  def test_mixed_model_linen(self):
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

    rules = [gptq.GptqRule(module_path='.*', weight_qtype=jnp.int8)]
    gptq_calibration_provider = gptq.GptqCalibrationProvider(rules)

    @jax.jit
    def calibrate_step(v, x):
      m = qwix_model.quantize_model(model, gptq_calibration_provider)
      return m.apply(v, x, mutable='quant_stats')

    _, new_variables = calibrate_step(variables, x)
    variables.update(new_variables)

    gptq_stats = variables['quant_stats']['key_gptq']
    self.assertEqual(gptq_stats['count'], 1)
    self.assertEqual(gptq_stats['sum_of_hessian'].shape, (32, 32))
    # The dictionary should only contain the keys for the valid modules.
    self.assertLen(variables['quant_stats'], 1)

  def test_mixed_model_nnx(self):
    """Tests that unsupported einsums are safely ignored in NNX."""

    class MixedModel(nnx.Module):

      def __init__(self, *, rngs=nnx.Rngs(0)):
        self.key = nnx.Param(
            jax.random.normal(rngs.params(), (32, 48))
        )  # Valid weight

      def __call__(self, x):
        # Valid Operation: Linear Layer (Matrix Mult)
        k_out = jnp.einsum('bi,io->bo', x, self.key)

        # Invalid Operation: Attention-style Batch Dot
        q = jax.random.normal(jax.random.key(0), x.shape)
        attn = jnp.einsum('bi,bi->b', q, x)

        # Invalid Operation: Transpose
        transposed = jnp.einsum('bi->ib', x)

        return k_out, attn, transposed

    model = MixedModel()
    x = jax.random.normal(jax.random.key(0), (5, 32))

    rules = [gptq.GptqRule(module_path='.*', weight_qtype=jnp.int8)]
    gptq_calibration_provider = gptq.GptqCalibrationProvider(rules)
    model_cal = qwix_model.quantize_model(model, gptq_calibration_provider, x)

    _ = model_cal(x)

    # Verify that only the valid operation got calibration stats.
    self.assertTrue(hasattr(model_cal, 'key_gptq'))
    gptq_stats = model_cal.key_gptq
    self.assertEqual(gptq_stats['count'], 1)
    self.assertEqual(gptq_stats['sum_of_hessian'].shape, (32, 32))

    # The dictionary should only contain the keys for the valid modules.
    quant_stats = nnx.state(model_cal, flax_util.QuantStat)
    self.assertLen(nnx.to_pure_dict(quant_stats), 1)


if __name__ == '__main__':
  absltest.main()
