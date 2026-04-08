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
from flax import nnx
from jax import numpy as jnp
from qwix._src import model as qwix_model
from qwix._src import qconfig
from qwix._src.core import qarray
from qwix._src.providers import ptq


class QconfigTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    dim: int = 16

    class MyModel(nnx.Module):

      def __init__(self, rngs: nnx.Rngs):
        self.lin1 = nnx.Linear(dim, dim, rngs=rngs)
        self.lin2 = nnx.Linear(dim, dim, rngs=rngs)
        self.layers = nnx.List(
            [nnx.Linear(dim, dim, rngs=rngs) for _ in range(2)]
        )

      def __call__(self, x):
        return self.lin1(x) + self.lin2(x) + sum(l(x) for l in self.layers)

    self.model = MyModel(rngs=nnx.Rngs(0))
    self.x = jnp.ones((1, dim))

  def test_all_rules_used(self):
    rules = [
        qconfig.QuantizationRule(
            weight_qtype="float8_e4m3fn",
            act_qtype="float8_e4m3fn",
            act_static_scale=False,
        ),
    ]
    provider = ptq.PtqProvider(rules)
    quant_model = qwix_model.quantize_model(self.model, provider, self.x)

    # Check unused rules.
    self.assertEmpty(provider.get_unused_rules())

    # Check that all layers are quantized.
    self.assertIsInstance(quant_model.lin1.kernel.array, qarray.QArray)
    self.assertIsInstance(quant_model.lin2.kernel.array, qarray.QArray)
    self.assertIsInstance(quant_model.layers[0].kernel.array, qarray.QArray)
    self.assertIsInstance(quant_model.layers[1].kernel.array, qarray.QArray)

  def test_some_rules_unused(self):
    rules = [
        qconfig.QuantizationRule(
            module_path=r"layers/\d+",
            weight_qtype="float8_e4m3fn",
            act_qtype="float8_e4m3fn",
            act_static_scale=False,
        ),
        qconfig.QuantizationRule(
            module_path=r"LIN\d+",  # Typo in module path.
            weight_qtype="float8_e4m3fn",
            act_qtype="float8_e4m3fn",
            act_static_scale=False,
        ),
    ]
    provider = ptq.PtqProvider(rules)
    quant_model = qwix_model.quantize_model(self.model, provider, self.x)
    unused_rules = provider.get_unused_rules()

    # Check unused rules.
    self.assertLen(unused_rules, 1)
    self.assertEqual(unused_rules[0].module_path, rules[1].module_path)

    # Check that lin1 and lin2 are not quantized.
    self.assertFalse(hasattr(quant_model.lin1.kernel, "array"))
    self.assertFalse(hasattr(quant_model.lin2.kernel, "array"))

    # Check that layers are quantized.
    self.assertIsInstance(quant_model.layers[0].kernel.array, qarray.QArray)
    self.assertIsInstance(quant_model.layers[1].kernel.array, qarray.QArray)

  def test_get_unused_rules_before_quantize_model(self):
    rules = [
        qconfig.QuantizationRule(
            module_path=r"layers/\d+",
            weight_qtype="float8_e4m3fn",
            act_qtype="float8_e4m3fn",
            act_static_scale=False,
        ),
    ]
    provider = ptq.PtqProvider(rules)
    with self.assertRaisesRegex(
        ValueError,
        "Quantization is not completed yet. Please call `quantize_model`"
        " before calling `get_unused_rules`.",
    ):
      provider.get_unused_rules()

    qwix_model.quantize_model(self.model, provider, self.x)
    self.assertEmpty(provider.get_unused_rules())


if __name__ == "__main__":
  absltest.main()
