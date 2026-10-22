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
import pickle
from typing import Any

from absl.testing import absltest
from flax import linen as nn
from flax import nnx
from jax import numpy as jnp
from qwix._src import model
from qwix._src import qconfig


class NnModel(nn.Module):

  def __call__(self, x):
    return self.sin(x)

  def sin(self, x):
    return jnp.sin(x)


class CustomProvider(qconfig.QuantizationProvider):

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    # Cache the intercept map so the lambdas share the same ID for
    # stable output_transform deduplication across nested intercepts.
    self._intercept_map = {
        "jax.numpy.sin": lambda x: x + 10,
        "jax.numpy.cos": lambda x: x + 20,
    }

  def get_intercept_map(self) -> Mapping[str, Callable[..., Any]]:
    return self._intercept_map

  def process_model_output(self, method_name: str, model_output: Any) -> Any:
    self._initial_run_complete = True
    return model_output + 100


class ModelTest(absltest.TestCase):

  def test_quantize_linen_model(self):
    quantized = model.quantize_linen_model(
        NnModel(), CustomProvider([]), methods=["sin", "__call__"]
    )
    self.assertEqual(quantized.sin(0), 0)
    self.assertEqual(quantized.apply({}, 0), 110)
    self.assertEqual(quantized.apply({}, 0, method="sin"), 110)


class NnxModel(nnx.Module):

  def __init__(self, offset: float = 0.0):
    self.offset = offset

  def __call__(self, x):
    return jnp.sin(x) + self.offset


def _sin_plus_10(x):
  return x + 10


class OffsetProvider(qconfig.QuantizationProvider):
  """Adds `bonus` to the output, to tell providers apart. Picklable."""

  def __init__(self, bonus: float):
    super().__init__([])
    self.bonus = bonus

  def get_intercept_map(self) -> Mapping[str, Callable[..., Any]]:
    return {"jax.numpy.sin": _sin_plus_10}

  def process_model_output(self, method_name: str, model_output: Any) -> Any:
    self._initial_run_complete = True
    return model_output + 100 + self.bonus


class NnxModelTest(absltest.TestCase):

  def test_quantize_nnx_model(self):
    quantized = model.quantize_nnx_model(NnxModel(), CustomProvider([]), 0.0)
    self.assertEqual(quantized(0.0), 110)
    # The original is untouched.
    self.assertEqual(NnxModel()(0.0), 0)

  def test_one_class_per_model_class_with_per_instance_providers(self):
    a = model.quantize_nnx_model(NnxModel(), OffsetProvider(1), 0.0)
    b = model.quantize_nnx_model(NnxModel(), OffsetProvider(2), 0.0)
    self.assertIs(type(a), type(b))
    self.assertIs(type(a), model.quantized_nnx_type(NnxModel))
    self.assertIs(type(a), NnxModel._qwix_quantized___call__)  # pylint: disable=protected-access
    self.assertEqual(a(0.0), 111)
    self.assertEqual(b(0.0), 112)
    # Re-quantizing a quantized model keeps the class and swaps the provider.
    c = model.quantize_nnx_model(a, OffsetProvider(3), 0.0)
    self.assertIs(type(c), type(a))
    self.assertEqual(c(0.0), 113)
    self.assertEqual(a(0.0), 111)

  def test_subclass_gets_its_own_quantized_type(self):
    class Sub(NnxModel):
      pass

    self.assertIsNot(
        model.quantized_nnx_type(Sub), model.quantized_nnx_type(NnxModel)
    )
    self.assertIs(model.quantized_nnx_type(Sub)._unquantized_type, Sub)  # pylint: disable=protected-access

  def test_split_merge_and_clone_keep_quantization(self):
    quantized = model.quantize_nnx_model(NnxModel(), OffsetProvider(1), 0.0)
    merged = nnx.merge(*nnx.split(quantized))
    self.assertIs(type(merged), type(quantized))
    self.assertEqual(merged(0.0), 111)
    self.assertEqual(nnx.clone(quantized)(0.0), 111)

  def test_graphdef_pickles_by_reference(self):
    quantized = model.quantize_nnx_model(NnxModel(), OffsetProvider(1), 0.0)
    graphdef, state = nnx.split(quantized)
    loaded = nnx.merge(pickle.loads(pickle.dumps(graphdef)), state)  # pylint: disable=g-unsafe-pickle-load
    self.assertIs(type(loaded), type(quantized))
    self.assertEqual(loaded(0.0), 111)

  def test_quantized_type_of_another_method(self):
    class TwoMethods(nnx.Module):

      def __call__(self, x):
        return x

      def sin(self, x):
        return jnp.sin(x)

    quantized = model.quantize_nnx_model(
        TwoMethods(), CustomProvider([]), 0.0, call_method="sin"
    )
    self.assertEqual(quantized.sin(0.0), 110)
    self.assertEqual(quantized(0.0), 0.0)
    self.assertIs(type(quantized), TwoMethods._qwix_quantized_sin)  # pylint: disable=protected-access
    self.assertIsNot(type(quantized), model.quantized_nnx_type(TwoMethods))


if __name__ == "__main__":
  absltest.main()
