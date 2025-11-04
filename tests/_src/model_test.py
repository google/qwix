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

from collections.abc import Callable, Mapping
from typing import Any

from absl.testing import absltest
from flax import linen as nn
from jax import numpy as jnp
from qwix._src import model
from qwix._src import qconfig


class NnModel(nn.Module):

  def __call__(self, x):
    return self.sin(x)

  def sin(self, x):
    return jnp.sin(x)


class CustomProvider(qconfig.QuantizationProvider):

  def get_intercept_map(self) -> Mapping[str, Callable[..., Any]]:
    return {
        "jax.numpy.sin": lambda x: x + 10,
        "jax.numpy.cos": lambda x: x + 20,
    }

  def process_model_output(self, method_name: str, model_output: Any) -> Any:
    return model_output + 100


class ModelTest(absltest.TestCase):

  def test_quantize_linen_model(self):
    quantized = model.quantize_linen_model(
        NnModel(), CustomProvider([]), methods=["sin", "__call__"]
    )
    self.assertEqual(quantized.sin(0), 0)
    self.assertEqual(quantized.apply({}, 0), 110)
    self.assertEqual(quantized.apply({}, 0, method="sin"), 110)


if __name__ == "__main__":
  absltest.main()
