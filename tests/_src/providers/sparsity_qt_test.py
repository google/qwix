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
"""Tests for sparsity_qt module, for update mask and apply sparsity."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
from qwix._src.core import sparsity
from qwix._src.providers import sparsity_qt


class SparsityQtTest(parameterized.TestCase):

  def test_no_sparsity(self):
    module = sparsity_qt.SparsityModule()
    inputs = jnp.arange(10, dtype=jnp.float32)
    weight = jnp.arange(10, dtype=jnp.float32)
    out_inputs, out_weight = module.apply({}, inputs, weight)
    self.assertTrue(jnp.array_equal(out_inputs, inputs))
    self.assertTrue(jnp.array_equal(out_weight, weight))

  def test_activation_sparsity(self):
    rule = sparsity.SparsityRule(
        activation_sparsity_n=1, activation_sparsity_m=2
    )
    module = sparsity_qt.SparsityModule(sparsity_rule=rule)
    inputs = jnp.array([1.0, 2.0, 3.0, 4.0])
    weight = jnp.array([1.0, 1.0, 1.0, 1.0])
    out_inputs, out_weight = module.apply({}, inputs, weight)
    self.assertTrue(
        jnp.array_equal(out_inputs, jnp.array([0.0, 2.0, 0.0, 4.0]))
    )
    self.assertTrue(jnp.array_equal(out_weight, weight))

  def test_weight_sparsity(self):
    rule = sparsity.SparsityRule(
        weight_sparsity_n=1,
        weight_sparsity_m=2,
        weight_sparsity_start_step=0,
        weight_sparsity_update_step=1,
    )
    module = sparsity_qt.SparsityModule(sparsity_rule=rule)
    inputs = jnp.array([1.0, 1.0, 1.0, 1.0])
    weight = jnp.array([1.0, 2.0, 3.0, 4.0])

    variables = module.init(jax.random.key(0), inputs, weight)
    self.assertEqual(variables["compression"]["step"], 0)

    (out_inputs, out_weight), new_vars = module.apply(
        variables, inputs, weight, mutable=["compression"]
    )
    self.assertEqual(new_vars["compression"]["step"], 1)
    expected_weight = jnp.array([0.0, 2.0, 0.0, 4.0])
    self.assertTrue(jnp.array_equal(out_inputs, inputs))
    self.assertTrue(jnp.array_equal(out_weight, expected_weight))

  def test_eval_mode(self):
    rule = sparsity.SparsityRule(
        weight_sparsity_n=1,
        weight_sparsity_m=2,
        eval_mode=True,
    )
    module = sparsity_qt.SparsityModule(sparsity_rule=rule)
    inputs = jnp.array([1.0, 1.0, 1.0, 1.0])
    weight = jnp.array([1.0, 2.0, 3.0, 4.0])

    variables = module.init(jax.random.key(0), inputs, weight)
    # Mask initialized to all ones in evaluation/init
    (out_inputs, out_weight), new_vars = module.apply(
        variables, inputs, weight, mutable=["compression"]
    )
    # In eval_mode, mask isn't updated and step isn't incremented.
    # It just applies the existing mask, which is currently all ones.
    self.assertTrue(jnp.array_equal(out_weight, weight))
    self.assertTrue(jnp.array_equal(out_inputs, inputs))
    self.assertEqual(new_vars["compression"]["step"], 0)


if __name__ == "__main__":
  absltest.main()
