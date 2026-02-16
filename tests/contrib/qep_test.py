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
from qwix._src import averaging
from qwix._src import model as qwix_model
from qwix._src.core import qarray
from qwix._src.providers import ptq
from qwix.contrib import gptq
from qwix.contrib import qep


class QepTest(parameterized.TestCase):

  def _make_dense_model(self):
    """Creates a simple dense model for testing."""

    class DenseModel(nn.Module):

      @nn.compact
      def __call__(self, x):
        x = nn.Dense(128)(x)
        x = nn.gelu(x)
        x = nn.Dense(64)(x)
        return x

    return DenseModel()

  def _dequantize_params(self, ptq_params):
    """Dequantize PTQ params to float with quantization artifacts."""
    return jax.tree.map(
        lambda p: (
            qarray.dequantize(p.array) if isinstance(p, ptq.WithAux) else p
        ),
        ptq_params,
        is_leaf=lambda p: isinstance(p, ptq.WithAux),
    )

  def test_qep_dense_model_calibration(self):
    """Tests QEP calibration collects correct stats structure."""
    model = self._make_dense_model()
    x = jax.random.normal(jax.random.key(0), (5, 32))
    variables = model.init(jax.random.key(1), x)

    # Get dequantized PTQ params to simulate quantized activations.
    rules = [qep.QepRule(module_path='Dense_0', weight_qtype=jnp.int8)]
    ptq_provider = ptq.PtqProvider(rules)
    ptq_model = qwix_model.quantize_model(model, ptq_provider)
    abs_variables = jax.eval_shape(ptq_model.init, jax.random.key(2), x)
    ptq_params = ptq.quantize_params(
        variables['params'], abs_variables['params']
    )
    deq_params = self._dequantize_params(ptq_params)

    # QEP Calibration.
    qep_provider = qep.QepCalibrationProvider(rules)
    cal_model = qwix_model.quantize_model(model, qep_provider)

    # First batch.
    new_vars = qep_provider.calibrate_batch(
        cal_model, variables, {'params': deq_params}, x
    )
    cal_variables = {**variables, **new_vars}

    # Check stats structure.
    qep_stats = cal_variables['quant_stats']['Dense_0']['kernel_qep']
    self.assertEqual(qep_stats['count'], 1)
    self.assertEqual(qep_stats['sum_of_hessian'].shape, (32, 32))
    self.assertEqual(qep_stats['sum_of_hessian_delta'].shape, (32, 32))

    # Second batch (test accumulation).
    new_vars = qep_provider.calibrate_batch(
        cal_model, cal_variables, {**cal_variables, 'params': deq_params}, x
    )
    cal_variables.update(new_vars)
    qep_stats = cal_variables['quant_stats']['Dense_0']['kernel_qep']
    self.assertEqual(qep_stats['count'], 2)

  def test_qep_better_than_ptq(self):
    """Tests that QEP output is closer to float than PTQ."""
    model = self._make_dense_model()
    x = jax.random.normal(jax.random.key(0), (5, 32))
    variables = model.init(jax.random.key(1), x)

    # Float output for reference.
    fp_y = model.apply(variables, x)

    rules = [qep.QepRule(module_path='Dense_0', weight_qtype=jnp.int8)]
    ptq_provider = ptq.PtqProvider(rules)
    ptq_model = qwix_model.quantize_model(model, ptq_provider)
    abs_variables = jax.eval_shape(ptq_model.init, jax.random.key(2), x)
    ptq_params = ptq.quantize_params(
        variables['params'], abs_variables['params']
    )
    deq_params = self._dequantize_params(ptq_params)

    # QEP calibration.
    qep_provider = qep.QepCalibrationProvider(rules)
    cal_model = qwix_model.quantize_model(model, qep_provider)
    new_vars = qep_provider.calibrate_batch(
        cal_model, variables, {'params': deq_params}, x
    )
    cal_variables = {**variables, **new_vars}

    # Quantize with QEP.
    qep_params = qep.quantize_params(
        variables['params'],
        abs_variables['params'],
        cal_variables['quant_stats'],
        correction_factor=0.5,
        dampening_factor=1.0,
    )
    qep_y = ptq_model.apply({'params': qep_params}, x)

    # PTQ output for comparison.
    ptq_y = ptq_model.apply({'params': ptq_params}, x)

    # QEP should be better than PTQ in terms of the output error.
    mae = lambda a, b: jnp.mean(jnp.abs(a - b))
    self.assertLess(mae(fp_y, qep_y), mae(fp_y, ptq_y))

  def test_qep_vs_gptq_accuracy(self):
    """Tests that QEP matches or improves upon standard GPTQ accuracy."""
    model = self._make_dense_model()
    x = jax.random.normal(jax.random.key(0), (5, 32))
    variables = model.init(jax.random.key(1), x)

    # Float output for reference.
    fp_y = model.apply(variables, x)

    rules = [qep.QepRule(module_path='Dense_0', weight_qtype=jnp.int8)]
    ptq_provider = ptq.PtqProvider(rules)
    ptq_model = qwix_model.quantize_model(model, ptq_provider)
    abs_variables = jax.eval_shape(ptq_model.init, jax.random.key(2), x)
    ptq_params = ptq.quantize_params(
        variables['params'], abs_variables['params']
    )
    deq_params = self._dequantize_params(ptq_params)

    # QEP calibration.
    qep_provider = qep.QepCalibrationProvider(rules)
    cal_model = qwix_model.quantize_model(model, qep_provider)
    new_vars = qep_provider.calibrate_batch(
        cal_model, variables, {'params': deq_params}, x
    )
    cal_variables = {**variables, **new_vars}

    # Quantize with QEP.
    qep_params = qep.quantize_params(
        variables['params'],
        abs_variables['params'],
        cal_variables['quant_stats'],
        correction_factor=0.5,
        dampening_factor=1.0,
    )
    qep_y = ptq_model.apply({'params': qep_params}, x)

    # Standard GPTQ for comparison.
    gptq_rules = [gptq.GptqRule(module_path='Dense_0', weight_qtype=jnp.int8)]
    gptq_cal_provider = gptq.GptqCalibrationProvider(gptq_rules)
    gptq_cal_model = qwix_model.quantize_model(model, gptq_cal_provider)
    _, gptq_vars = gptq_cal_model.apply(variables, x, mutable='quant_stats')
    gptq_variables = {**variables, **gptq_vars}
    gptq_params = gptq.quantize_params(
        variables['params'],
        abs_variables['params'],
        gptq_variables['quant_stats'],
    )
    gptq_y = ptq_model.apply({'params': gptq_params}, x)

    mae = lambda a, b: jnp.mean(jnp.abs(a - b))
    qep_error = mae(fp_y, qep_y)
    gptq_error = mae(fp_y, gptq_y)

    # QEP should be at least as good as standard GPTQ.
    self.assertLessEqual(qep_error, gptq_error)

  def test_qep_all_layers_quantized(self):
    """Tests QEP accuracy when all layers are quantized."""
    model = self._make_dense_model()
    x = jax.random.normal(jax.random.key(0), (5, 32))
    variables = model.init(jax.random.key(1), x)
    fp_y = model.apply(variables, x)

    # Quantize ALL Dense layers.
    rules = [qep.QepRule(module_path='Dense_.*', weight_qtype=jnp.int8)]
    ptq_provider = ptq.PtqProvider(rules)
    ptq_model = qwix_model.quantize_model(model, ptq_provider)
    abs_variables = jax.eval_shape(ptq_model.init, jax.random.key(2), x)
    ptq_params = ptq.quantize_params(
        variables['params'], abs_variables['params']
    )
    deq_params = self._dequantize_params(ptq_params)

    # QEP calibration over all layers.
    qep_provider = qep.QepCalibrationProvider(rules)
    cal_model = qwix_model.quantize_model(model, qep_provider)
    new_vars = qep_provider.calibrate_batch(
        cal_model, variables, {'params': deq_params}, x
    )
    cal_variables = {**variables, **new_vars}

    # Quantize with QEP.
    qep_params = qep.quantize_params(
        variables['params'],
        abs_variables['params'],
        cal_variables['quant_stats'],
        correction_factor=0.5,
        dampening_factor=1.0,
    )
    qep_y = ptq_model.apply({'params': qep_params}, x)

    # PTQ output.
    ptq_y = ptq_model.apply({'params': ptq_params}, x)

    # Standard GPTQ for comparison.
    gptq_rules = [gptq.GptqRule(module_path='Dense_.*', weight_qtype=jnp.int8)]
    gptq_cal_provider = gptq.GptqCalibrationProvider(gptq_rules)
    gptq_cal_model = qwix_model.quantize_model(model, gptq_cal_provider)
    _, gptq_vars = gptq_cal_model.apply(variables, x, mutable='quant_stats')
    gptq_variables = {**variables, **gptq_vars}
    gptq_params = gptq.quantize_params(
        variables['params'],
        abs_variables['params'],
        gptq_variables['quant_stats'],
    )
    gptq_y = ptq_model.apply({'params': gptq_params}, x)

    mae = lambda a, b: jnp.mean(jnp.abs(a - b))

    # QEP should be better than PTQ.
    self.assertLess(mae(fp_y, qep_y), mae(fp_y, ptq_y))

    # GPTQ should also be better than PTQ (baseline sanity check).
    self.assertLess(mae(fp_y, gptq_y), mae(fp_y, ptq_y))

  def test_qep_no_matching_layers_raises(self):
    """Tests that calibrate_batch raises when no layers match."""
    model = self._make_dense_model()
    x = jax.random.normal(jax.random.key(0), (5, 32))
    variables = model.init(jax.random.key(1), x)

    # Use a module_path that matches nothing.
    rules = [qep.QepRule(module_path='NonExistent', weight_qtype=jnp.int8)]
    qep_provider = qep.QepCalibrationProvider(rules)
    cal_model = qwix_model.quantize_model(model, qep_provider)

    with self.assertRaises(ValueError):
      qep_provider.calibrate_batch(cal_model, variables, variables, x)

  def test_quantize_params_without_hessian_delta_raises(self):
    """Tests error when QEP stats lack hessian_delta."""
    model = self._make_dense_model()
    x = jax.random.normal(jax.random.key(0), (5, 32))
    variables = model.init(jax.random.key(1), x)

    rules = [qep.QepRule(module_path='Dense_0', weight_qtype=jnp.int8)]
    ptq_provider = ptq.PtqProvider(rules)
    ptq_model = qwix_model.quantize_model(model, ptq_provider)
    abs_variables = jax.eval_shape(ptq_model.init, jax.random.key(2), x)

    fake_hessian = jnp.eye(32)
    fake_stats = averaging.SimpleMovingAverage().init({'hessian': fake_hessian})
    fake_stats = averaging.SimpleMovingAverage().update(
        fake_stats, {'hessian': fake_hessian}
    )
    quant_stats = {'Dense_0': {'kernel_qep': fake_stats}}

    with self.assertRaises(ValueError):
      qep.quantize_params(
          variables['params'],
          abs_variables['params'],
          quant_stats,
      )

  def test_qep_dot_general_without_calibrate_batch_raises(self):
    """Tests that dot_general raises when called outside calibrate_batch."""
    model = self._make_dense_model()
    x = jax.random.normal(jax.random.key(0), (5, 32))
    variables = model.init(jax.random.key(1), x)

    rules = [qep.QepRule(module_path='Dense_0', weight_qtype=jnp.int8)]
    qep_provider = qep.QepCalibrationProvider(rules)
    cal_model = qwix_model.quantize_model(model, qep_provider)

    with self.assertRaisesRegex(ValueError, 'Must use calibrate_batch'):
      # This calls dot_general which checks self._mode
      cal_model.apply(variables, x, mutable='quant_stats')


if __name__ == '__main__':
  absltest.main()
