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
import copy
import pickle
from typing import Any

from absl.testing import absltest
from absl.testing import parameterized

from flax import linen as nn
from flax import nnx
from jax import numpy as jnp
from qwix._src import model
from qwix._src import qconfig
from qwix._src.providers import odml
from qwix._src.providers import qt


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

  def __call__(self, x):
    return jnp.sin(x)


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


class NnxPickleTest(absltest.TestCase):

  def test_graphdef_pickle_rebuilds_class(self):
    quantized = model.quantize_nnx_model(NnxModel(), OffsetProvider(1), 0.0)
    graphdef, state = nnx.split(quantized)
    loaded = nnx.merge(pickle.loads(pickle.dumps(graphdef)), state)  # pylint: disable=g-unsafe-pickle-load
    # The class is rebuilt from the pickle rather than looked up by name, so
    # this works in a process that never quantized the model.
    self.assertIsNot(type(loaded), type(quantized))
    self.assertIs(type(loaded)._unquantized_type, NnxModel)  # pylint: disable=protected-access
    self.assertEqual(loaded(0.0), 111)

  def test_deepcopy_keeps_class_and_graphdef(self):
    quantized = model.quantize_nnx_model(NnxModel(), OffsetProvider(1), 0.0)
    copied = copy.deepcopy(quantized)
    self.assertIs(type(copied), type(quantized))
    self.assertEqual(nnx.graphdef(copied), nnx.graphdef(quantized))
    self.assertEqual(copied(0.0), 111)

  def test_pickle_format(self):
    # Existing pickles store this call; see `_reduce_quantized_nnx_type`.
    provider = OffsetProvider(1)
    quantized = model.quantize_nnx_model(NnxModel(), provider, 0.0)
    fn, args = model._reduce_quantized_nnx_type(type(quantized))  # pylint: disable=protected-access
    self.assertEqual(
        (fn.__module__, fn.__qualname__),
        ("qwix._src.model", "_make_quantized_nnx_type"),
    )
    self.assertEqual(args, (NnxModel, provider, "__call__"))


class TwoLinears(nnx.Module):

  def __init__(self):
    self.linear1 = nnx.Linear(4, 8, rngs=nnx.Rngs(0))
    self.linear2 = nnx.Linear(8, 4, rngs=nnx.Rngs(1))

  def __call__(self, x):
    return self.linear2(self.linear1(x))


_INT8_RULES = [
    qconfig.QuantizationRule(module_path=".*", weight_qtype=jnp.int8)
]


class BuiltInProviderTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ("qt", lambda: qt.QtProvider(_INT8_RULES)),
      ("odml_qat", lambda: odml.OdmlQatProvider(_INT8_RULES)),
  )
  def test_pickle(self, make_provider):
    x = jnp.ones((2, 4))
    quantized = model.quantize_model(TwoLinears(), make_provider(), x)
    graphdef, state = nnx.split(quantized)
    loaded = nnx.merge(pickle.loads(pickle.dumps(graphdef)), state)  # pylint: disable=g-unsafe-pickle-load
    self.assertTrue(jnp.array_equal(loaded(x), quantized(x)))


if __name__ == "__main__":
  absltest.main()
