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
from qwix.contrib import gptq


class GptqTest(parameterized.TestCase):

  def test_dense_model_linen(self):
    class Model(nn.Module):

      @nn.compact
      def __call__(self, x):
        x = nn.Dense(128)(x)
        x = nn.gelu(x)
        x = nn.Dense(64)(x)
        return x

    model = Model()
    x = jax.random.normal(jax.random.key(0), (5, 32))
    variables = model.init(jax.random.key(1), x)

    # Calibration.
    rules = [gptq.GptqRule(module_path='Dense_0', weight_qtype=jnp.int8)]
    gptq_calibration_provider = gptq.GptqCalibrationProvider(rules)
    # Note that GptqCalibrationProvider doesn't perform any quantization.
    model = qwix_model.quantize_model(model, gptq_calibration_provider)
    _, new_variables = model.apply(variables, x, mutable='quant_stats')
    variables.update(new_variables)
    gptq_stats = variables['quant_stats']['Dense_0']['kernel_gptq']
    self.assertEqual(gptq_stats['count'], 1)
    self.assertEqual(gptq_stats['sum_of_hessian'].shape, (32, 32))
    fp_y, new_variables = model.apply(variables, x, mutable='quant_stats')
    variables.update(new_variables)
    gptq_stats = variables['quant_stats']['Dense_0']['kernel_gptq']
    self.assertEqual(gptq_stats['count'], 2)

    # Weight quantization with GPTQ. Note that we use the same PtqProvider to
    # get the abstract quantized params tree.
    ptq_provider = ptq.PtqProvider(rules)
    model = qwix_model.quantize_model(model, ptq_provider)
    abs_variables = jax.eval_shape(model.init, jax.random.key(2), x)
    gptq_params = gptq.quantize_params(
        variables['params'], abs_variables['params'], variables['quant_stats']
    )

    # Load it to PTQ model.
    gptq_y = model.apply({'params': gptq_params}, x)

    # Compare with plain PTQ.
    ptq_params = ptq.quantize_params(
        variables['params'], abs_variables['params']
    )
    ptq_y = model.apply({'params': ptq_params}, x)

    # GPTQ should be better than PTQ in terms of the output error.
    mae = lambda x, y: jnp.mean(jnp.abs(x - y))
    self.assertLess(mae(fp_y, gptq_y), mae(fp_y, ptq_y))


if __name__ == '__main__':
  absltest.main()
