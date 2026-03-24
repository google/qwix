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
from qwix._src.providers import ptq
from qwix.contrib import gptq
from qwix.contrib import qep
from qwix.contrib import qep_core


def _mae(a, b):
  return jnp.mean(jnp.abs(a - b))


class QepLinenTest(parameterized.TestCase):

  def _make_dense_model(self):
    class DenseModel(nn.Module):

      @nn.compact
      def __call__(self, x, return_hidden=False):
        x = nn.Dense(128, name='Dense_0')(x)
        x = nn.gelu(x)
        hidden = x
        x = nn.Dense(64, name='Dense_1')(x)
        if return_hidden:
          return hidden, x
        return x

    return DenseModel()

  def _make_branch_model(self):
    class BranchModel(nn.Module):

      @nn.compact
      def __call__(self, x):
        a = nn.Dense(16, name='DenseA')(x)
        b = nn.Dense(16, name='DenseB')(x)
        x = jax.nn.relu(a + b)
        x = nn.Dense(8, name='DenseC')(x)
        return x

    return BranchModel()

  def _make_ptq_model(self, model, rules):
    ptq_provider = ptq.PtqProvider(rules)
    ptq_model = qwix_model.quantize_model(model, ptq_provider)
    return ptq_model

  def _get_abs_quantized(self, ptq_model, x):
    return jax.eval_shape(ptq_model.init, jax.random.key(0), x)['params']

  def _manual_two_stage_reference(self, model, variables, x):
    stage0_rules = [qep.QepRule(module_path='Dense_0', weight_qtype=jnp.int8)]
    stage0_result = qep.quantize(model, [x], stage0_rules, variables=variables)
    hidden_q, _ = stage0_result.model.apply(
        {'params': stage0_result.params}, x, return_hidden=True
    )
    hidden_fp, _ = model.apply(variables, x, return_hidden=True)

    stage1_rules = [qep.QepRule(module_path='Dense_1', weight_qtype=jnp.int8)]
    ptq_model = self._make_ptq_model(model, stage1_rules)
    abs_quantized = self._get_abs_quantized(ptq_model, x)
    stage1_stats = qep_core.compute_qep_stats(hidden_q.T, hidden_fp.T)
    aggregator = averaging.SimpleMovingAverage()
    quant_stat = aggregator.update(aggregator.init(stage1_stats), stage1_stats)
    stage1_params = qep.quantize_params(
        variables['params'],
        abs_quantized,
        {'Dense_1': {'kernel_qep': quant_stat}},
    )
    stage1_params['Dense_0'] = stage0_result.params['Dense_0']
    full_ptq_model = self._make_ptq_model(
        model, [qep.QepRule(module_path='Dense_.*', weight_qtype=jnp.int8)]
    )
    return full_ptq_model.apply({'params': stage1_params}, x)

  def test_single_layer_qep_beats_ptq_and_matches_gptq(self):
    model = self._make_dense_model()
    x = jax.random.normal(jax.random.key(0), (8, 32))
    variables = model.init(jax.random.key(1), x)
    fp_y = model.apply(variables, x)

    rules = [qep.QepRule(module_path='Dense_0', weight_qtype=jnp.int8)]
    result = qep.quantize(model, [x], rules, variables=variables)
    qep_y = result.model.apply({'params': result.params}, x)

    ptq_model = self._make_ptq_model(model, rules)
    abs_quantized = self._get_abs_quantized(ptq_model, x)
    ptq_params = ptq.quantize_params(variables['params'], abs_quantized)
    ptq_y = ptq_model.apply({'params': ptq_params}, x)

    gptq_provider = gptq.GptqCalibrationProvider(
        [gptq.GptqRule(module_path='Dense_0', weight_qtype=jnp.int8)]
    )
    gptq_model = qwix_model.quantize_model(model, gptq_provider)
    _, gptq_vars = gptq_model.apply(variables, x, mutable='quant_stats')
    gptq_params = gptq.quantize_params(
        variables['params'], abs_quantized, gptq_vars['quant_stats']
    )
    gptq_y = ptq_model.apply({'params': gptq_params}, x)

    self.assertLess(_mae(fp_y, qep_y), _mae(fp_y, ptq_y))
    self.assertLessEqual(_mae(fp_y, qep_y), _mae(fp_y, gptq_y) * 1.1)

  def test_exact_stagewise_matches_manual_two_stage_reference(self):
    model = self._make_dense_model()
    x = jax.random.normal(jax.random.key(2), (8, 32))
    variables = model.init(jax.random.key(3), x)

    rules = [qep.QepRule(module_path='Dense_.*', weight_qtype=jnp.int8)]
    result = qep.quantize(model, [x], rules, variables=variables)
    exact_y = result.model.apply({'params': result.params}, x)
    ref_y = self._manual_two_stage_reference(model, variables, x)

    self.assertLess(_mae(exact_y, ref_y), 1e-6)

  def test_infers_shared_input_branch_stage(self):
    model = self._make_branch_model()
    x = jax.random.normal(jax.random.key(4), (8, 12))
    variables = model.init(jax.random.key(5), x)
    rules = [qep.QepRule(module_path='.*', weight_qtype=jnp.int8)]

    result = qep.quantize(model, [x], rules, variables=variables)

    self.assertLen(result.stages, 2)
    self.assertEqual(set(result.stages[0].module_paths), {'DenseA', 'DenseB'})
    self.assertEqual(result.stages[1].module_paths, ('DenseC',))

  def test_no_matching_layers_raises(self):
    model = self._make_dense_model()
    x = jax.random.normal(jax.random.key(6), (4, 32))
    variables = model.init(jax.random.key(7), x)

    with self.assertRaises(ValueError):
      qep.quantize(
          model,
          [x],
          [qep.QepRule(module_path='NonExistent', weight_qtype=jnp.int8)],
          variables=variables,
      )

  def test_non_reiterable_input_raises(self):
    model = self._make_dense_model()
    x = jax.random.normal(jax.random.key(8), (4, 32))
    variables = model.init(jax.random.key(9), x)
    rules = [qep.QepRule(module_path='Dense_0', weight_qtype=jnp.int8)]

    with self.assertRaises(ValueError):
      qep.quantize(model, iter([x]), rules, variables=variables)

  def test_quantize_params_without_correction_does_not_require_hessian_delta(
      self,
  ):
    model = self._make_dense_model()
    x = jax.random.normal(jax.random.key(10), (4, 32))
    variables = model.init(jax.random.key(11), x)
    rules = [qep.QepRule(module_path='Dense_0', weight_qtype=jnp.int8)]
    ptq_model = self._make_ptq_model(model, rules)
    abs_quantized = self._get_abs_quantized(ptq_model, x)

    fake_hessian = jnp.eye(32)
    aggregator = averaging.SimpleMovingAverage()
    quant_stat = aggregator.update(
        aggregator.init({'hessian': fake_hessian}),
        {'hessian': fake_hessian},
    )
    params = qep.quantize_params(
        variables['params'],
        abs_quantized,
        {'Dense_0': {'kernel_qep': quant_stat}},
        apply_correction=False,
    )

    y = ptq_model.apply({'params': params}, x)
    self.assertEqual(y.shape, (4, 64))


if __name__ == '__main__':
  absltest.main()
