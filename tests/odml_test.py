# Copyright 2024 Google LLC
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
"""Test the ODML providers for op coverage."""


from absl.testing import absltest
from absl.testing import parameterized
import flax
from flax import linen as nn
from flax import nnx
import jax
from jax import numpy as jnp
from qwix import flax_util
from qwix import model as qwix_model
from qwix import odml
from qwix import qconfig


class OdmlTest(parameterized.TestCase):

  def test_linen(self):
    class LinenModel(nn.Module):

      @nn.compact
      def __call__(self, x):
        x = nn.Dense(features=64)(x)
        x = nn.relu(x)
        x = nn.Dense(features=16)(x)
        return x

    model = LinenModel()
    rules = [
        qconfig.QuantizationRule(
            module_path='.*',
            weight_qtype=jnp.int8,
            act_qtype=jnp.int8,
        ),
    ]
    qat_provider = odml.OdmlQatProvider(rules)
    qat_model = qwix_model.quantize_model(model, qat_provider)
    model_input = jnp.ones((1, 16), dtype=jnp.float32)
    qat_vars = qat_model.init(jax.random.key(0), model_input)
    qat_res, new_vars = qat_model.apply(qat_vars, model_input, mutable=True)
    qat_vars.update(new_vars)

    self.assertEqual(
        {
            '/'.join(k[:-1])
            for k in flax.traverse_util.flatten_dict(qat_vars['quant_stats'])
        },
        {
            'Dense_0/dot_general0_lhs',
            'Dense_1/dot_general0_lhs',
            'final_output0',
        },
    )

    conversion_provider = odml.OdmlConversionProvider(
        rules,
        qat_vars['params'],
        qat_vars['quant_stats'],
    )
    conversion_model = qwix_model.quantize_model(model, conversion_provider)
    conversion_res = conversion_model.apply(qat_vars, model_input)
    self.assertTrue(jnp.allclose(qat_res, conversion_res))

  def test_nnx(self):
    class NnxModel(nnx.Module):

      def __init__(self, rngs: nnx.Rngs):
        self.linear1 = nnx.Linear(16, 64, rngs=rngs)
        self.linear2 = nnx.Linear(64, 16, rngs=rngs)

      def __call__(self, x):
        x = self.linear1(x)
        x = nnx.relu(x)
        x = self.linear2(x)
        return x

    model = NnxModel(nnx.Rngs(0))
    rules = [
        qconfig.QuantizationRule(
            module_path='.*',
            weight_qtype=jnp.int8,
            act_qtype=jnp.int8,
        ),
    ]
    qat_provider = odml.OdmlQatProvider(rules)
    model_input = jnp.ones((1, 16), dtype=jnp.float32)
    qat_model = qwix_model.quantize_model(model, qat_provider, model_input)
    qat_res = qat_model(model_input)

    quant_stats = nnx.to_pure_dict(nnx.state(qat_model, flax_util.QuantStat))
    self.assertEqual(
        {
            '/'.join(k[:-1])
            for k in flax.traverse_util.flatten_dict(quant_stats)
        },
        {
            'linear2/dot_general0_lhs',
            'linear1/dot_general0_lhs',
            'final_output0',
        },
    )

    conversion_provider = odml.OdmlConversionProvider(
        rules,
        nnx.to_pure_dict(nnx.state(qat_model, nnx.Param)),
        quant_stats,
    )
    conversion_model = qwix_model.quantize_model(
        model, conversion_provider, model_input
    )
    conversion_res = conversion_model(model_input)
    self.assertTrue(jnp.allclose(qat_res, conversion_res))


if __name__ == '__main__':
  absltest.main()
