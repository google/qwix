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
from flax import linen as nn
from flax import nnx
import jax
from jax import numpy as jnp
from jax.nn import initializers
import numpy as np
from qwix._src import flax_util


class FlaxUtilTest(parameterized.TestCase):

  def test_get_current_module(self):
    test = self

    class Bar(nnx.Module):

      def __init__(self, rngs: nnx.Rngs):
        self.rngs = rngs
        self.linear1 = nnx.Linear(2, 3, rngs=rngs)
        self.linear2 = nnx.Linear(3, 4, rngs=rngs)

      def __call__(self, x):
        test.assertEqual(flax_util.get_current_module(), self)
        x = self.linear1(x)
        x = self.linear2(x)
        return x

    class Foo(nnx.Module):

      def __init__(self, rngs: nnx.Rngs):
        self.rngs = rngs
        self.bar = Bar(rngs=rngs)
        self.linear1 = nnx.Linear(4, 5, rngs=rngs)

      def __call__(self, x):
        test.assertEqual(flax_util.get_current_module(), self)
        x = self.bar(x)
        x = self.linear1(x)
        return x

    foo = Foo(nnx.Rngs(0))
    foo(jnp.ones((1, 2)))

  def test_get_or_create_variable_linen(self):
    class Foo(nn.Module):

      def setup(self):
        self.weight = self.param(
            "weight", nn.initializers.ones, (4, 5), jnp.float32
        )

      @nn.compact
      def __call__(self, x):
        flax_util.get_or_create_variable(
            "quant_stats", "test_var", lambda: jnp.ones((4, 5), jnp.float32)
        )
        return x @ self.weight

    foo = Foo()

    variables = foo.init(jax.random.PRNGKey(0), jnp.ones((1, 4)))
    self.assertLen(jax.tree.flatten(variables), 2)
    np.testing.assert_array_equal(
        variables["params"]["weight"], jnp.ones((4, 5), jnp.float32)
    )
    np.testing.assert_array_equal(
        variables["quant_stats"]["test_var"], jnp.ones((4, 5), jnp.float32)
    )

  def test_get_or_create_variable_nnx(self):
    class Foo(nnx.Module):

      def __init__(self):
        self.weight = nnx.Param(jnp.ones((4, 5), jnp.float32))

      def __call__(self, x):
        flax_util.get_or_create_variable(
            "quant_stats", "test_var", lambda: jnp.ones((4, 5), jnp.float32)
        )
        return x @ self.weight

    foo = Foo()

    foo(jnp.ones((1, 4)))
    variables = nnx.state(foo)
    self.assertLen(jax.tree.flatten(variables), 2)
    np.testing.assert_array_equal(
        variables["weight"].value, jnp.ones((4, 5), jnp.float32)
    )
    self.assertEqual(variables["weight"].type, nnx.Param)
    np.testing.assert_array_equal(
        variables["test_var"].value, jnp.ones((4, 5), jnp.float32)
    )
    self.assertEqual(variables["test_var"].type, flax_util.QuantStat)

  def test_create_param_nnx(self):
    class Foo(nnx.Module):

      def __init__(self):
        self.weight = nnx.Param(jnp.ones((4, 5), jnp.float32))

      def __call__(self, x):
        flax_util.get_or_create_param(
            "weight", lambda: jnp.zeros((4, 5), jnp.float32)
        )  # should not change weight
        flax_util.get_or_create_param(
            "lora_a",
            lambda: initializers.he_uniform()(
                jax.random.key(0), (4, 1), jnp.float32
            ),
            nnx.LoRAParam,
        )
        flax_util.get_or_create_param(
            "lora_b",
            lambda: jnp.zeros((1, 5), jnp.float32),
            nnx.LoRAParam,
        )
        return x @ self.weight + x @ self.lora_a + self.lora_b

    foo = Foo()

    foo(jnp.ones((1, 4)))
    variables = nnx.variables(foo)
    self.assertLen(variables.flat_state(), 3)
    np.testing.assert_array_equal(
        variables["weight"].value, jnp.ones((4, 5), jnp.float32)
    )

    self.assertIsInstance(variables["lora_a"], nnx.LoRAParam)
    self.assertIsInstance(variables["lora_b"], nnx.LoRAParam)

  def test_unbox(self):
    mesh = jax.make_mesh((1, 1), ("a", "b"))
    with jax.set_mesh(mesh):
      array = jnp.ones((4, 8))
      boxed = {
          "unboxed": array,
          "nn": nn.Partitioned(array, names=("a", "b")),
          "nnx": nnx.Param(array, sharding=("a", "b")),
      }
      unboxed = flax_util.unbox(boxed)
    np.testing.assert_array_equal(unboxed["unboxed"], array)
    np.testing.assert_array_equal(unboxed["nn"], array)
    np.testing.assert_array_equal(unboxed["nnx"], array)

  def test_update_boxed(self):
    mesh = jax.make_mesh((1, 1), ("a", "b"))
    unboxed = jnp.ones((4, 8))
    value = jnp.zeros((4, 8))
    self.assertIs(flax_util.update_boxed(unboxed), unboxed)
    self.assertIs(flax_util.update_boxed(unboxed, value=value), value)

    with jax.set_mesh(mesh):
      boxed = nn.Partitioned(jnp.ones((4, 8)), names=("a", "b"))
      updated = flax_util.update_boxed(
          boxed, value=jnp.ones((2, 2, 8)), split=[0]
      )
      self.assertIsInstance(updated, nn.Partitioned)
      self.assertEqual(updated.value.shape, (2, 2, 8))
      self.assertEqual(updated.names, ("a", None, "b"))

      boxed = nnx.Param(jnp.ones((2, 2, 8)), sharding=("a", None, "b"))
      updated = flax_util.update_boxed(boxed, value=jnp.ones((4, 8)), merge=[0])
      self.assertIsInstance(updated, nnx.Param)
      self.assertEqual(updated.value.shape, (4, 8))
      self.assertEqual(updated.sharding_names, ("a", "b"))

      boxed = nnx.Param(jnp.ones((2, 2, 8)), sharding=("a", None, "b"))
      updated = flax_util.update_boxed(boxed, transpose=[2, 0, None])
      self.assertIsInstance(updated, nnx.Param)
      self.assertEqual(updated.sharding_names, ("b", "a", None))


if __name__ == "__main__":
  absltest.main()
