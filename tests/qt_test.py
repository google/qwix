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

from absl.testing import absltest
from flax import linen as nn
from flax import nnx
import jax
from jax import numpy as jnp
from qwix import flax_util
from qwix import model as qwix_model
from qwix import qconfig
from qwix import qt

jax.config.update("jax_threefry_partitionable", False)


class QtTest(absltest.TestCase):

  def _make_array(self, shape, seed=42):
    return jax.random.normal(jax.random.key(seed), shape, jnp.bfloat16)

  def test_dot_general_grad(self):
    qt_provider = qt.QtProvider([])
    rule = qconfig.QuantizationRule(
        module_path=".*",
        weight_qtype=jnp.int8,
        act_qtype=jnp.int8,
        act_calibration_method="absmax",
    )
    qt_provider._get_current_rule_and_op_id = lambda _: (rule, None)

    lhs = self._make_array((2, 4)) * 8
    rhs = self._make_array((4, 2)) * 8
    dimension_numbers = ([0, 1], [1, 0]), ([], [])
    fp_value, fp_grad = jax.value_and_grad(jax.lax.dot_general)(
        lhs, rhs, dimension_numbers
    )
    q_value, q_grad = jax.value_and_grad(qt_provider.dot_general)(
        lhs, rhs, dimension_numbers
    )
    self.assertEqual(fp_value.dtype, q_value.dtype)
    self.assertEqual(fp_value.shape, q_value.shape)
    # The gradients are not exactly the same, because each input is using the
    # quantized value of the other side to compute the gradient.
    self.assertFalse(jnp.array_equal(fp_grad, q_grad), f"{fp_grad=} {q_grad=}")
    rel_mae = jnp.abs(fp_value - q_value).mean() / jnp.abs(fp_value).mean()
    self.assertLess(rel_mae, 0.01)

  def test_srq_jit_grad(self):
    """Test that the grad of SRQ can be taken inside a jitted function."""
    dense = nn.Dense(features=10, param_dtype=jnp.bfloat16)
    qt_provider = qt.QtProvider([
        qconfig.QuantizationRule(
            module_path=".*",
            weight_qtype=jnp.int8,
            act_qtype=jnp.int8,
            act_static_scale=True,
        ),
    ])
    qt_dense = qwix_model.quantize_model(dense, qt_provider)
    model_input = jnp.ones((10, 12), dtype=jnp.bfloat16)
    variables = qt_dense.init(jax.random.key(0), model_input)
    self.assertEqual(
        variables["quant_stats"]["dot_general0_lhs"]["sum_of_absmax"].shape,
        (1, 1),
    )
    self.assertEqual(variables["quant_stats"]["dot_general0_lhs"]["count"], 0)

    @jax.jit
    def jit_apply(variables):
      def loss_fn(params):
        out, new_vars = qt_dense.apply(
            {"params": params, "quant_stats": variables["quant_stats"]},
            model_input,
            mutable="quant_stats",
        )
        return jnp.sum(jnp.abs(out)), new_vars["quant_stats"]

      return jax.value_and_grad(loss_fn, has_aux=True)(variables["params"])

    # Some users prefer frozen dict for variables.
    (loss, quant_stats), unused_grad = jit_apply(nn.FrozenDict(variables))
    self.assertEqual(loss.dtype, jnp.bfloat16)
    # They should have the same structure.
    jax.tree.map(lambda *_: ..., quant_stats, variables["quant_stats"])
    self.assertEqual(quant_stats["dot_general0_lhs"]["count"], 1)

  def test_srq_jit_grad_nnx(self):
    """Test SRQ on NNX module."""
    linear = nnx.Linear(12, 10, rngs=nnx.Rngs(0))
    qt_provider = qt.QtProvider([
        qconfig.QuantizationRule(
            module_path=".*",
            weight_qtype=jnp.int8,
            act_qtype=jnp.int8,
            act_static_scale=True,
        ),
    ])

    model_input = jnp.ones((10, 12))
    qt_linear = qwix_model.quantize_model(linear, qt_provider, model_input)
    quant_stats = nnx.variables(qt_linear, flax_util.QuantStat)

    # quant_stats should be initialized but empty.
    self.assertLen(quant_stats, 1)
    self.assertEqual(quant_stats["dot_general0_lhs"].value["count"], 0)

    @nnx.jit
    def jit_apply(model, x):
      def loss_fn(model, x):
        out = model(x)
        return jnp.sum(jnp.abs(out))

      return nnx.grad(loss_fn)(model, x)

    jit_apply(qt_linear, model_input)
    quant_stats = nnx.variables(qt_linear, flax_util.QuantStat)

    self.assertNotEmpty(quant_stats)
    self.assertLen(quant_stats, 1)
    self.assertIn("dot_general0_lhs", quant_stats)
    self.assertEqual(
        quant_stats["dot_general0_lhs"].value["sum_of_absmax"].shape, (1, 1)
    )
    self.assertEqual(quant_stats["dot_general0_lhs"].value["count"], 1)

    # Disables quant_stats update and check that quant_stats are not updated.
    act_sum_of_absmax = quant_stats["dot_general0_lhs"].value["sum_of_absmax"]
    qt_linear.set_attributes(
        disable_quant_stats_update=True, raise_if_not_found=False
    )
    jit_apply(qt_linear, model_input)
    quant_stats = nnx.variables(qt_linear, flax_util.QuantStat)

    self.assertEqual(
        quant_stats["dot_general0_lhs"]["sum_of_absmax"], act_sum_of_absmax
    )
    self.assertEqual(quant_stats["dot_general0_lhs"]["count"], 1)

    # Enables quant_stats update and check that quant_stats are updated again.
    qt_linear.set_attributes(
        disable_quant_stats_update=False, raise_if_not_found=False
    )
    jit_apply(qt_linear, model_input)
    quant_stats = nnx.variables(qt_linear, flax_util.QuantStat)

    self.assertNotEqual(
        quant_stats["dot_general0_lhs"]["sum_of_absmax"], act_sum_of_absmax
    )
    self.assertEqual(quant_stats["dot_general0_lhs"]["count"], 2)


if __name__ == "__main__":
  absltest.main()
