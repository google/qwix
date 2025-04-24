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
import os

from absl.testing import absltest
from absl.testing import parameterized
from flax import linen as nn
from flax import nnx
import jax
from jax import numpy as jnp
from jax import sharding as shd
import numpy as np
from qwix import lora
from qwix import ptq

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"


class LoraTest(parameterized.TestCase):

  @parameterized.parameters(
      dict(
          einsum_str="BTD,DNH->BTNH",
          lhs_shape=(16, 4, 12),
          rhs_shape=(12, 10, 8),
          output_shape=(16, 4, 10, 8),
          lora_rank=3,
          expected_lora_einsum_str="BTD,Dr,rNH->BTNH",
      ),
      dict(
          einsum_str="bjir,bnmjr->bimn",
          lhs_shape=(16, 4, 12, 5),
          rhs_shape=(16, 10, 8, 4, 5),
          output_shape=(16, 12, 8, 10),
          lora_rank=3,
          expected_lora_einsum_str="bjir,bjrA,Anm->bimn",
      ),
      dict(
          einsum_str=" ABD, DNH -> ABNH",
          lhs_shape=(5, 16, 12),
          rhs_shape=(12, 10, 8),
          output_shape=(5, 16, 10, 8),
          lora_rank=3,
          expected_lora_einsum_str="ABD,Dr,rNH->ABNH",
      ),
  )
  def test_parse_einsum_str_for_lora(
      self,
      einsum_str,
      lhs_shape,
      rhs_shape,
      output_shape,
      lora_rank,
      expected_lora_einsum_str,
  ):
    a_shape, b_shape, lora_einsum_str, _, _ = lora._parse_einsum_str_for_lora(
        lhs_shape, rhs_shape, einsum_str, lora_rank
    )
    self.assertEqual(lora_einsum_str, expected_lora_einsum_str)
    self.assertEqual(a_shape[-1], lora_rank)
    self.assertEqual(b_shape[0], lora_rank)

    lhs = jnp.ones(lhs_shape)
    lora_a = jnp.ones(a_shape)
    lora_b = jnp.ones(b_shape)
    res = jnp.einsum(lora_einsum_str, lhs, lora_a, lora_b)

    self.assertEqual(res.shape, output_shape)

  @parameterized.parameters(
      dict(
          einsum_str="BTD,DNH->BTNH->BTNH",
          lhs_shape=(16, 4, 12),
          rhs_shape=(12, 10, 8),
      ),
      dict(
          einsum_str="BTD->BTNH",
          lhs_shape=(16, 4, 12),
          rhs_shape=(12, 10, 8),
      ),
  )
  def test_parse_einsum_str_for_lora_invalid_einsum_str(
      self, einsum_str, lhs_shape, rhs_shape
  ):
    with self.assertRaises(ValueError):
      lora._parse_einsum_str_for_lora(
          lhs_shape=lhs_shape,
          rhs_shape=rhs_shape,
          einsum_str=einsum_str,
          lora_rank=3,
      )

  @parameterized.parameters(0.0, 1.0)
  def test_lora_jit_grad_dot_general(self, dropout_rate):
    """Test LoRA on nnx.Linear module."""
    linear = nnx.Linear(12, 10, rngs=nnx.Rngs(0))
    lora_provider = lora.LoraProvider([
        lora.LoraRule(
            module_path=".*",
            rank=3,
            alpha=0.5,
            dropout=dropout_rate,
            lora_b_initializer=nnx.initializers.ones,
        ),
    ])

    model_input = jnp.ones((10, 12))
    lora_linear = lora.apply_lora_to_model(linear, lora_provider, model_input)

    @nnx.jit
    def jit_apply(model, x):
      def loss_fn(model, x):
        out = model(x)
        return jnp.sum(jnp.abs(out))

      return nnx.grad(loss_fn)(model, x)

    jit_apply(lora_linear, model_input)
    variables = nnx.variables(lora_linear, nnx.LoRAParam)
    self.assertLen(variables, 2)
    self.assertIn("kernel_lora_a", variables)
    self.assertIn("kernel_lora_b", variables)

    expected_model_output = linear(model_input)
    if dropout_rate == 0.0:
      expected_model_output += (
          model_input
          @ variables["kernel_lora_a"].value
          @ variables["kernel_lora_b"].value
          * 0.5
      )

    model_output = lora_linear(model_input)
    np.testing.assert_array_equal(model_output, expected_model_output)

  @parameterized.parameters(0.0, 1.0)
  def test_lora_jit_grad_einsum(self, dropout_rate):
    """Test LoRA on nnx.Einsum module."""
    einsum = nnx.Einsum("btd,dnh->btnh", (12, 8, 10), (8, 10), rngs=nnx.Rngs(0))
    lora_provider = lora.LoraProvider([
        lora.LoraRule(
            module_path=".*",
            rank=3,
            alpha=1.0,
            dropout=dropout_rate,
            lora_b_initializer=nnx.initializers.ones,
        ),
    ])
    model_input = jnp.ones((16, 4, 12))
    lora_einsum = lora.apply_lora_to_model(einsum, lora_provider, model_input)
    self.assertEqual(lora_einsum.kernel_lora_einsum_str, "btd,dr,rnh->btnh")

    @nnx.jit
    def jit_apply(model, x):
      def loss_fn(model, x):
        out = model(x)
        return jnp.sum(jnp.abs(out))

      return nnx.grad(loss_fn)(model, x)

    jit_apply(lora_einsum, model_input)

    variables = nnx.variables(lora_einsum, nnx.LoRAParam)
    self.assertLen(variables, 2)
    self.assertIn("kernel_lora_a", variables)
    self.assertIn("kernel_lora_b", variables)

    expected_model_output = einsum(model_input)
    if dropout_rate == 0.0:
      expected_model_output += jnp.einsum(
          "btd,dr,rnh->btnh",
          model_input,
          variables["kernel_lora_a"].value,
          variables["kernel_lora_b"].value,
      )

    model_output = lora_einsum(model_input)
    np.testing.assert_array_equal(model_output, expected_model_output)

  @parameterized.parameters("nf4", "int4")
  def test_qlora_jit_grad_einsum(self, weight_qtype):
    """Test QLoRA on nnx.Einsum module."""
    einsum = nnx.Einsum("btd,dnh->btnh", (12, 8, 10), (8, 10), rngs=nnx.Rngs(0))
    qlora_provider = lora.LoraProvider([
        lora.LoraRule(
            module_path=".*",
            rank=3,
            alpha=1.0,
            dropout=0.0,
            weight_qtype=weight_qtype,
            tile_size=4,
            lora_b_initializer=nnx.initializers.ones,
        ),
    ])
    model_input = jnp.ones((16, 4, 12))
    qlora_einsum = lora.apply_lora_to_model(einsum, qlora_provider, model_input)

    nnx.display(qlora_einsum)
    self.assertEqual(qlora_einsum.kernel_lora_einsum_str, "btd,dr,rnh->btnh")

    @nnx.jit
    def jit_apply(model, x):
      def loss_fn(model, x):
        out = model(x)
        return jnp.sum(jnp.abs(out))

      return nnx.grad(loss_fn, argnums=nnx.DiffState(0, nnx.LoRAParam))(
          model, x
      )

    jit_apply(qlora_einsum, model_input)

    variables = nnx.variables(qlora_einsum, nnx.LoRAParam)
    self.assertLen(variables, 2)
    self.assertIn("kernel_lora_a", variables)
    self.assertIn("kernel_lora_b", variables)

    # Runs the model again to verify numeric correctness.
    model_output = qlora_einsum(model_input)

    lora_model_output = einsum(model_input) + jnp.einsum(
        "btd,dr,rnh->btnh",
        model_input,
        variables["kernel_lora_a"].value,
        variables["kernel_lora_b"].value,
    )

    rel_mae = (
        jnp.abs(model_output - lora_model_output).mean()
        / jnp.abs(lora_model_output).mean()
    )
    self.assertLess(rel_mae, 0.1)

  def test_lora_einsum_jit_with_sharding(self):
    """Test LoRA on nnx.Einsum module with sharding."""

    @nnx.jit
    def create_sharded_model():
      einsum = nnx.Einsum(
          "btd,dnh->btnh",
          (12, 8, 10),
          (8, 10),
          rngs=nnx.Rngs(0),
          kernel_init=nnx.with_partitioning(
              nnx.initializers.lecun_normal(), ("fsdp", "tp", None)
          ),
          bias_init=nnx.with_partitioning(nnx.initializers.zeros, ("tp", None)),
      )

      lora_provider = lora.LoraProvider([
          lora.LoraRule(
              module_path=".*",
              rank=3,
              alpha=6.0,
              dropout=0.0,
          ),
      ])
      model_input = jnp.ones((16, 4, 12))
      lora_einsum = lora.apply_lora_to_model(einsum, lora_provider, model_input)
      state = nnx.state(lora_einsum)
      pspecs = nnx.get_partition_spec(state)
      sharded_state = jax.lax.with_sharding_constraint(state, pspecs)
      nnx.update(lora_einsum, sharded_state)
      return lora_einsum

    mesh = jax.make_mesh((2, 2), ("fsdp", "tp"))
    with mesh:
      sharded_lora_einsum = create_sharded_model()

    self.assertEqual(sharded_lora_einsum.kernel.sharding, ("fsdp", "tp", None))
    self.assertEqual(sharded_lora_einsum.kernel_lora_a.sharding, ("fsdp", None))
    self.assertEqual(
        sharded_lora_einsum.kernel_lora_b.sharding, (None, "tp", None)
    )
    self.assertEqual(
        sharded_lora_einsum.kernel_lora_a.value.sharding.spec,
        shd.PartitionSpec("fsdp"),
    )
    self.assertEqual(
        sharded_lora_einsum.kernel_lora_b.value.sharding.spec,
        shd.PartitionSpec(None, "tp"),
    )

    input_sharding = shd.NamedSharding(mesh, shd.PartitionSpec("fsdp"))
    model_input = jax.device_put(jnp.ones((16, 4, 12)), input_sharding)
    with mesh:
      model_output = sharded_lora_einsum(model_input)

    self.assertEqual(
        model_output.sharding.spec,
        shd.PartitionSpec("fsdp", None, "tp"),
    )

  def test_qlora_einsum_eval_shape_with_sharding(self):
    """Test QLoRA on nnx.Einsum module param with sharding."""

    def lora_einsum(*model_args, **model_kwargs):
      einsum = nnx.Einsum(
          "btd,dnh->btnh",
          (12, 8, 10),
          (8, 10),
          rngs=nnx.Rngs(0),
          kernel_init=nnx.with_partitioning(
              nnx.initializers.lecun_normal(), ("fsdp", "tp", None)
          ),
          bias_init=nnx.with_partitioning(nnx.initializers.zeros, ("tp", None)),
      )
      lora_provider = lora.LoraProvider([
          lora.LoraRule(
              module_path=".*",
              rank=3,
              alpha=6.0,
              dropout=0.0,
              weight_qtype="nf4",
              tile_size=4,
          ),
      ])
      return lora.apply_lora_to_model(
          einsum, lora_provider, *model_args, **model_kwargs
      )

    abs_lora_einsum = nnx.eval_shape(lora_einsum, jnp.ones((16, 4, 12)))

    abs_lora_state, abs_state = nnx.state(
        abs_lora_einsum, nnx.LoRAParam, nnx.Param
    )

    self.assertEqual(abs_lora_state.kernel_lora_a.sharding, ("fsdp", None))
    self.assertEqual(
        abs_state.kernel.array.qvalue.sharding, ("fsdp", "tp", None)
    )
    self.assertEqual(
        abs_state.kernel.array.scale.sharding, ("fsdp", "tp", None)
    )

  @parameterized.named_parameters(
      dict(
          testcase_name="without_quantization",
          weight_qtype=None,
      ),
      dict(
          testcase_name="with_quantization",
          weight_qtype="int8",
      ),
  )
  def test_lora_with_nn(self, weight_qtype):
    """Test LoRA on nn.Dense module."""
    dense = nn.Dense(
        32, kernel_init=nn.with_partitioning(nn.zeros_init(), ("a", "b"))
    )
    lora_provider = lora.LoraProvider(
        lora.LoraRule(
            module_path=".*",
            rank=3,
            alpha=6.0,
            dropout=0.0,
            weight_qtype=weight_qtype,
        )
    )
    lora_dense = lora.apply_lora_to_model(dense, lora_provider)
    model_input = jnp.ones((10, 16))
    lora_variables = lora_dense.init(jax.random.PRNGKey(0), model_input)
    kernel = lora_variables["params"]["kernel"]
    lora_a = lora_variables["params"]["kernel_lora_a"]
    lora_b = lora_variables["params"]["kernel_lora_b"]
    if weight_qtype is None:
      self.assertIsInstance(kernel, nn.Partitioned)
      self.assertEqual(kernel.unbox().shape, (16, 32))
      self.assertEqual(kernel.names, ("a", "b"))
    else:
      self.assertIsInstance(kernel, ptq.WithAux)
      self.assertIsInstance(kernel.array.qvalue, nn.Partitioned)
      self.assertEqual(kernel.array.qvalue.unbox().shape, (16, 32))
      self.assertEqual(kernel.array.qvalue.names, ("a", "b"))

    self.assertIsInstance(lora_a, nn.Partitioned)
    self.assertEqual(lora_a.unbox().shape, (16, 3))
    self.assertEqual(lora_a.names, ("a", None))
    self.assertIsInstance(lora_b, nn.Partitioned)
    self.assertEqual(lora_b.unbox().shape, (3, 32))
    self.assertEqual(lora_b.names, (None, "b"))


if __name__ == "__main__":
  absltest.main()
