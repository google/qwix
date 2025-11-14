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
import functools
import os

from absl.testing import absltest
from absl.testing import parameterized
from flax import linen as nn
from flax import nnx
import jax
from jax import numpy as jnp
from jax import sharding as shd
from qwix._src.providers import lora
from qwix._src.providers import ptq

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"


class LoraTest(parameterized.TestCase):

  @parameterized.parameters(
      dict(
          einsum_str="BTD,DNH->BTNH",
          lhs_shape=(16, 4, 12),
          rhs_shape=(12, 10, 8),
          lora_rank=3,
          expected_lora_einsum_str="BTD,Dr,rNH->BTNH",
          expected_a_sharding_transpose=(0, None),
          expected_b_sharding_transpose=(None, 1, 2),
          expected_output_shape=(16, 4, 10, 8),
      ),
      dict(
          einsum_str="bjir,bnmjr->bimn",
          lhs_shape=(16, 4, 12, 5),
          rhs_shape=(16, 10, 8, 4, 5),
          lora_rank=3,
          expected_lora_einsum_str="bjir,bjrA,Anm->bimn",
          expected_a_sharding_transpose=(0, 3, 4, None),
          expected_b_sharding_transpose=(None, 1, 2),
          expected_output_shape=(16, 12, 8, 10),
      ),
      dict(
          einsum_str=" ABD, DNH -> ABNH",
          lhs_shape=(5, 16, 12),
          rhs_shape=(12, 10, 8),
          lora_rank=3,
          expected_lora_einsum_str="ABD,Dr,rNH->ABNH",
          expected_a_sharding_transpose=(0, None),
          expected_b_sharding_transpose=(None, 1, 2),
          expected_output_shape=(5, 16, 10, 8),
      ),
  )
  def test_parse_einsum_str_for_lora(
      self,
      einsum_str,
      lhs_shape,
      rhs_shape,
      lora_rank,
      expected_lora_einsum_str,
      expected_a_sharding_transpose,
      expected_b_sharding_transpose,
      expected_output_shape,
  ):
    (
        a_shape,
        b_shape,
        lora_einsum_str,
        a_sharding_transpose,
        b_sharding_transpose,
    ) = lora._parse_einsum_str_for_lora(
        lhs_shape, rhs_shape, einsum_str, lora_rank
    )
    self.assertEqual(lora_einsum_str, expected_lora_einsum_str)
    self.assertEqual(a_shape[-1], lora_rank)
    self.assertEqual(b_shape[0], lora_rank)
    self.assertEqual(a_sharding_transpose, expected_a_sharding_transpose)
    self.assertEqual(b_sharding_transpose, expected_b_sharding_transpose)

    lhs = jax.ShapeDtypeStruct(lhs_shape, jnp.float32)
    lora_a = jax.ShapeDtypeStruct(a_shape, jnp.float32)
    lora_b = jax.ShapeDtypeStruct(b_shape, jnp.float32)
    res = jax.eval_shape(
        functools.partial(jnp.einsum, lora_einsum_str), lhs, lora_a, lora_b
    )
    self.assertEqual(res.shape, expected_output_shape)

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
  def test_lora_dot_general_nnx(self, dropout_rate):
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
    lora_linear.set_attributes(qwix_rngs=nnx.Rngs(dropout=0))

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
          / 3
      )

    model_output = lora_linear(model_input)
    self.assertTrue(jnp.allclose(model_output, expected_model_output))

  @parameterized.parameters(0.0, 1.0)
  def test_lora_einsum_nnx(self, dropout_rate):
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
    lora_einsum.set_attributes(qwix_rngs=nnx.Rngs(dropout=0))
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
      expected_model_output += (
          jnp.einsum(
              "btd,dr,rnh->btnh",
              model_input,
              variables["kernel_lora_a"].value,
              variables["kernel_lora_b"].value,
          )
          / 3
      )

    model_output = lora_einsum(model_input)
    self.assertTrue(jnp.allclose(model_output, expected_model_output))

  @parameterized.product(
      weight_qtype=[None, "nf4", "int4"],
      apply_sharding_to_base_model=[True, False],
  )
  def test_lora_einsum_nnx_sharded(
      self, weight_qtype, apply_sharding_to_base_model
  ):
    """Test QLoRA on nnx.Einsum module param with sharding."""
    mesh = jax.make_mesh(
        (2, 2),
        ("fsdp", "tp"),
        axis_types=(jax.sharding.AxisType.Auto,) * len(("fsdp", "tp")),
    )
    with jax.set_mesh(mesh):
      einsum = nnx.Einsum(
          "btd,dnh->btnh",
          (16, 8, 10),
          (8, 10),
          rngs=nnx.Rngs(0),
          kernel_init=nnx.with_partitioning(
              nnx.initializers.lecun_normal(), ("fsdp", "tp", None)
          ),
          bias_init=nnx.with_partitioning(nnx.initializers.zeros, ("tp", None)),
      )
    if apply_sharding_to_base_model:
      # Apply sharding on the base model.
      self._shard_nnx_model(einsum, mesh)
      self.assertEqual(einsum.kernel.sharding_names, ("fsdp", "tp", None))
      self.assertEqual(einsum.kernel.value.sharding.spec, ("fsdp", "tp", None))
      self.assertEqual(einsum.bias.sharding_names, ("tp", None))
      self.assertEqual(einsum.bias.value.sharding.spec, ("tp", None))

    lora_provider = lora.LoraProvider([
        lora.LoraRule(
            rank=3,
            alpha=6.0,
            weight_qtype=weight_qtype,
            tile_size=4,
        ),
    ])
    input_sharding = shd.NamedSharding(
        mesh, shd.PartitionSpec("fsdp", None, None)
    )
    model_input = jax.device_put(jnp.ones((16, 4, 16)), input_sharding)
    with jax.set_mesh(mesh):
      lora_einsum = lora.apply_lora_to_model(einsum, lora_provider, model_input)
    self.assertEqual(lora_einsum.kernel_lora_einsum_str, "btd,dr,rnh->btnh")

    if not apply_sharding_to_base_model:
      # Apply sharding on the LoRA model. Both should work.
      self._shard_nnx_model(lora_einsum, mesh)

    lora_state, base_state = nnx.state(lora_einsum, nnx.LoRAParam, nnx.Param)

    self.assertEqual(lora_state.kernel_lora_a.value.shape, (16, 3))
    self.assertEqual(lora_state.kernel_lora_a.sharding_names, ("fsdp", None))
    self.assertEqual(lora_state.kernel_lora_b.value.shape, (3, 8, 10))
    self.assertEqual(
        lora_state.kernel_lora_b.sharding_names, (None, "tp", None)
    )
    if weight_qtype is None:  # unquantized
      self.assertEqual(base_state.kernel.sharding_names, ("fsdp", "tp", None))
      self.assertEqual(
          base_state.kernel.value.sharding.spec, ("fsdp", "tp", None)
      )
    else:  # quantized weights
      self.assertEqual(base_state.kernel.array.qvalue.value.shape, (16, 8, 10))
      self.assertEqual(
          base_state.kernel.array.qvalue.sharding_names, ("fsdp", "tp", None)
      )
      self.assertEqual(base_state.kernel.array.scale.value.shape, (4, 8, 10))
      self.assertEqual(
          base_state.kernel.array.scale.sharding_names, ("fsdp", "tp", None)
      )

    # Check that the actual sharding matches the sharding on the metadata,
    # regardless of whether apply_sharding_to_base_model is True or False.
    for variable in nnx.iter_graph(lora_einsum):
      if isinstance(variable, nnx.Variable):
        self.assertEqual(variable.sharding, variable.value.sharding.spec)

    model_output = lora_einsum(model_input)
    self.assertTrue(model_output.shape, (16, 4, 8, 10))
    self.assertTrue(model_output.sharding.spec, ("fsdp", None, "tp", None))

  @parameterized.named_parameters(
      dict(
          testcase_name="no_quant",
          weight_qtype=None,
      ),
      dict(
          testcase_name="int8",
          weight_qtype="int8",
      ),
  )
  def test_lora_dot_general_nn(self, weight_qtype):
    """Test LoRA on nn.Dense module."""
    dense = nn.Dense(
        32, kernel_init=nn.with_partitioning(nn.zeros_init(), ("a", "b"))
    )
    lora_provider = lora.LoraProvider(
        module_path=".*",
        rank=3,
        alpha=6.0,
        dropout=0.0,
        weight_qtype=weight_qtype,
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

  def test_lora_conv_nn(self):
    """Test LoRA on nn.Conv module."""
    conv = nn.Conv(
        features=32,
        kernel_size=(3, 3),
        kernel_init=nn.with_partitioning(
            nn.zeros_init(), (None, None, "in", "out")
        ),
    )
    lora_provider = lora.LoraProvider(rank=3, alpha=1.0)
    lora_conv = lora.apply_lora_to_model(conv, lora_provider)
    model_input = jnp.ones((1, 8, 8, 16))
    lora_variables = lora_conv.init(jax.random.PRNGKey(0), model_input)
    lora_a = lora_variables["params"]["kernel_lora_a"]
    lora_b = lora_variables["params"]["kernel_lora_b"]
    self.assertIsInstance(lora_a, nn.Partitioned)
    self.assertEqual(lora_a.unbox().shape, (3, 3, 16, 3))
    self.assertEqual(lora_a.names, (None, None, "in", None))
    self.assertIsInstance(lora_b, nn.Partitioned)
    self.assertEqual(lora_b.unbox().shape, (3, 32))
    self.assertEqual(lora_b.names, (None, "out"))

  @parameterized.named_parameters(
      dict(
          testcase_name="no_quant",
          qtype=None,
      ),
      dict(
          testcase_name="int8",
          qtype="int8",  # Qwix PTQ requires act_qtype == weight_qtype.
      ),
  )
  def test_lora_conv_nnx(self, qtype):
    """Test LoRA on nnx.Conv module."""
    mesh = jax.make_mesh(
        (2, 2),
        ("in", "out"),
        axis_types=(jax.sharding.AxisType.Auto,) * len(("in", "out")),
    )
    with jax.set_mesh(mesh):
      conv = nnx.Conv(
          in_features=16,
          out_features=32,
          kernel_size=(3, 3),
          kernel_init=nnx.with_partitioning(
              nnx.initializers.zeros, (None, None, "in", "out")
          ),
          rngs=nnx.Rngs(0),
      )
      # Shard the module on a 2x2 mesh.
      self._shard_nnx_model(
          conv,
          jax.make_mesh(
              (2, 2),
              ("in", "out"),
              axis_types=(jax.sharding.AxisType.Auto,) * len(("in", "out")),
          ),
      )
    # Check the sharding of both the metadata and the actual jax.Array.
    self.assertEqual(conv.kernel.sharding_names, (None, None, "in", "out"))
    self.assertEqual(conv.kernel.value.sharding.spec, (None, None, "in", "out"))

    lora_provider = lora.LoraProvider(
        weight_qtype=qtype,
        act_qtype=qtype,
        rank=3,
        alpha=1.0,
    )
    model_input = jnp.ones((1, 8, 8, 16))
    with jax.set_mesh(mesh):
      lora_conv = lora.apply_lora_to_model(conv, lora_provider, model_input)
    lora_a = lora_conv.kernel_lora_a
    lora_b = lora_conv.kernel_lora_b
    self.assertIsInstance(lora_a, nnx.LoRAParam)
    self.assertEqual(lora_a.shape, (3, 3, 16, 3))
    self.assertEqual(lora_a.sharding_names, (None, None, "in", None))
    self.assertEqual(lora_a.value.sharding.spec, (None, None, "in", None))
    self.assertIsInstance(lora_b, nnx.LoRAParam)
    self.assertEqual(lora_b.shape, (3, 32))
    self.assertEqual(lora_b.sharding_names, (None, "out"))
    self.assertEqual(lora_b.value.sharding.spec, (None, "out"))

  def _shard_nnx_model(self, model: nnx.Module, mesh: jax.sharding.Mesh):
    """Shards the model in-place with the given mesh."""
    unsharded_state = nnx.state(model)
    sharding = nnx.get_named_sharding(unsharded_state, mesh)
    sharded_state = jax.device_put(unsharded_state, sharding)
    nnx.update(model, sharded_state)

  def test_nnx_remat(self):
    """Test nnx.remat with LoRA."""

    class Model(nnx.Module):

      def __init__(self, *, rngs: nnx.Rngs):
        self.linear = nnx.Linear(16, 32, rngs=rngs)

      def __call__(self, x):
        return nnx.remat(self.linear.__call__.__func__)(self.linear, x)

    model = Model(rngs=nnx.Rngs(0))
    lora_provider = lora.LoraProvider(rank=3, alpha=1.0)
    x = jnp.ones((10, 16))
    lora_model = lora.apply_lora_to_model(model, lora_provider, x)
    self.assertIsInstance(lora_model.linear.kernel_lora_a, nnx.LoRAParam)


if __name__ == "__main__":
  absltest.main()
