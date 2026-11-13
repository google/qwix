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
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from flax import linen as nn
from flax import nnx
import jax
from jax import numpy as jnp
from jax import sharding as shd
from qwix._src.providers import boxed_param
from qwix._src.providers import lora

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

  def test_lora_with_jnp_matmul_nnx(self):
    """jnp.matmul passes reduction_order to the intercepted dot_general."""

    class LinearThenMatmul(nnx.Module):

      def __init__(self, rngs: nnx.Rngs):
        self.linear = nnx.Linear(12, 10, rngs=rngs)
        self.basis = jnp.arange(40, dtype=jnp.float32).reshape(10, 4) / 40

      def __call__(self, x):
        # Like Gemma4's audio mel projection: `x @ constant` via jnp.matmul.
        return self.linear(x) @ self.basis

    module = LinearThenMatmul(nnx.Rngs(0))
    lora_provider = lora.LoraProvider([
        lora.LoraRule(
            module_path=".*",
            rank=3,
            alpha=0.5,
            lora_b_initializer=nnx.initializers.ones,
        ),
    ])
    model_input = jnp.ones((10, 12))
    base_linear_output = module.linear(model_input)
    lora_module = lora.apply_lora_to_model(module, lora_provider, model_input)
    variables = nnx.variables(lora_module.linear, nnx.LoRAParam)
    expected_model_output = (
        base_linear_output
        + model_input
        @ variables["kernel_lora_a"].value
        @ variables["kernel_lora_b"].value
        * 0.5
        / 3
    ) @ module.basis
    self.assertTrue(
        jnp.allclose(lora_module(model_input), expected_model_output)
    )

  def test_lora_dot_general_passes_options_through_on_non_weight(self):
    """Options unknown to Qwix, e.g. from a newer JAX, reach JAX unchanged."""

    class Module(nn.Module):

      @nn.compact
      def __call__(self, x, y, **options):
        return jax.lax.dot_general(x, y, (((1,), (0,)), ((), ())), **options)

    mod = Module()
    x = jnp.ones((2, 3), dtype=jnp.float32)
    y = jnp.ones((3, 4), dtype=jnp.float32)
    # JAX itself rejects the made-up option, with or without Qwix.
    with self.assertRaises(TypeError) as jax_error:
      mod.apply({}, x, y, new_option="x")
    lora_mod = lora.apply_lora_to_model(
        mod, lora.LoraProvider(rank=2, alpha=1.0)
    )
    with self.assertRaises(TypeError) as lora_error:
      lora_mod.apply({}, x, y, new_option="x")
    self.assertEqual(str(lora_error.exception), str(jax_error.exception))

  def test_lora_dot_general_options_on_weight(self):
    class Module(nn.Module):
      provider: lora.LoraProvider

      @nn.compact
      def __call__(self, x, **options):
        w = self.param("w", nn.initializers.ones, (3, 4))
        dnums = (((1,), (0,)), ((), ()))
        return self.provider.dot_general(x, w, dnums, **options)

    mod = Module(lora.LoraProvider(rank=2, alpha=1.0))
    x = jnp.ones((2, 3), dtype=jnp.float32)
    jax_dot_general = jax.lax.dot_general

    def dot_general(*args, new_option=None, reduction_order=None, **kwargs):
      del new_option, reduction_order  # The JAX under test may not have them.
      return jax_dot_general(*args, **kwargs)

    with mock.patch.object(
        jax.lax, "dot_general", side_effect=dot_general
    ) as mock_dot_general:
      y, _ = mod.init_with_output(jax.random.key(0), x, new_option="x")
      self.assertEqual(y.shape, (2, 4))
      # Only the base contraction passes the option on to JAX: the two
      # contractions of the LoRA delta ignore it.
      self.assertEqual(
          ["new_option" in c.kwargs for c in mock_dot_general.call_args_list],
          [True, False, False],
      )
      with self.assertRaisesRegex(NotImplementedError, "reduction_order"):
        mod.init(jax.random.key(0), x, reduction_order="linear")

  def test_lora_einsum_passes_options_through_on_non_weight(self):
    """Options unknown to Qwix, e.g. from a newer JAX, reach JAX unchanged."""

    class Module(nn.Module):

      @nn.compact
      def __call__(self, x, y, **options):
        return jnp.einsum("ij,jk->ik", x, y, **options)

    mod = Module()
    x = jnp.ones((2, 3), dtype=jnp.float32)
    y = jnp.ones((3, 4), dtype=jnp.float32)
    with self.assertRaises(TypeError) as jax_error:
      mod.apply({}, x, y, new_option="x")
    lora_mod = lora.apply_lora_to_model(
        mod, lora.LoraProvider(rank=2, alpha=1.0)
    )
    with self.assertRaises(TypeError) as lora_error:
      lora_mod.apply({}, x, y, new_option="x")
    self.assertEqual(str(lora_error.exception), str(jax_error.exception))

  def test_lora_einsum_options_on_weight(self):
    class Module(nnx.Module):

      def __init__(self):
        self.w = nnx.Param(jnp.ones((3, 4)))

      def __call__(self, x, **options):
        return jnp.einsum("ij,jk->ik", x, self.w, **options)

    x = jnp.ones((2, 3), dtype=jnp.float32)
    lora_module = lora.apply_lora_to_model(
        Module(), lora.LoraProvider(rank=2, alpha=1.0), x
    )
    jax_einsum = jnp.einsum

    def einsum(*args, new_option=None, reduction_order=None, **kwargs):
      del new_option, reduction_order
      return jax_einsum(*args, **kwargs)

    with mock.patch.object(jnp, "einsum", side_effect=einsum) as mock_einsum:
      y = lora_module(x, new_option="x")
      self.assertEqual(y.shape, (2, 4))
      # Only the base contraction passes the option on to JAX; the LoRA delta
      # ignores it.
      self.assertEqual(
          ["new_option" in c.kwargs for c in mock_einsum.call_args_list],
          [True, False],
      )
      with self.assertRaisesRegex(NotImplementedError, "reduction_order"):
        lora_module(x, reduction_order="linear")

  def test_lora_conv_general_dilated_passes_options_through_on_non_weight(self):
    """Options unknown to Qwix, e.g. from a newer JAX, reach JAX unchanged."""

    class Module(nn.Module):

      @nn.compact
      def __call__(self, x, y, **options):
        return jax.lax.conv_general_dilated(
            x,
            y,
            (1, 1),
            "SAME",
            dimension_numbers=("NHWC", "HWIO", "NHWC"),
            **options,
        )

    mod = Module()
    x = jnp.ones((1, 4, 4, 2), dtype=jnp.float32)
    y = jnp.ones((2, 2, 2, 3), dtype=jnp.float32)
    with self.assertRaises(TypeError) as jax_error:
      mod.apply({}, x, y, new_option="x")
    lora_mod = lora.apply_lora_to_model(
        mod, lora.LoraProvider(rank=2, alpha=1.0)
    )
    with self.assertRaises(TypeError) as lora_error:
      lora_mod.apply({}, x, y, new_option="x")
    self.assertEqual(str(lora_error.exception), str(jax_error.exception))

  def test_lora_conv_general_dilated_options_on_weight(self):
    class Module(nn.Module):
      provider: lora.LoraProvider

      @nn.compact
      def __call__(self, x, **options):
        w = self.param("w", nn.initializers.ones, (2, 2, 2, 3))
        return self.provider.conv_general_dilated(
            x,
            w,
            window_strides=(1, 1),
            padding="SAME",
            dimension_numbers=("NHWC", "HWIO", "NHWC"),
            **options,
        )

    mod = Module(lora.LoraProvider(rank=2, alpha=1.0))
    x = jnp.ones((1, 4, 4, 2), dtype=jnp.float32)
    jax_conv = jax.lax.conv_general_dilated

    def conv_general_dilated(
        *args, new_option=None, reduction_order=None, **kwargs
    ):
      del new_option, reduction_order
      return jax_conv(*args, **kwargs)

    with mock.patch.object(
        jax.lax, "conv_general_dilated", side_effect=conv_general_dilated
    ) as mock_conv:
      y, _ = mod.init_with_output(jax.random.key(0), x, new_option="x")
      self.assertEqual(y.shape, (1, 4, 4, 3))
      self.assertEqual(
          ["new_option" in c.kwargs for c in mock_conv.call_args_list],
          [True, False],
      )
      with self.assertRaisesRegex(NotImplementedError, "reduction_order"):
        mod.init(jax.random.key(0), x, reduction_order="linear")

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
      self.assertEqual(
          einsum.kernel.value.sharding.spec,
          jax.sharding.PartitionSpec("fsdp", "tp", None),
      )
      self.assertEqual(einsum.bias.sharding_names, ("tp", None))
      self.assertEqual(
          einsum.bias.value.sharding.spec,
          jax.sharding.PartitionSpec("tp", None),
      )

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
          base_state.kernel.value.sharding.spec,
          jax.sharding.PartitionSpec("fsdp", "tp", None),
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
        self.assertEqual(
            jax.sharding.PartitionSpec(*variable.sharding),
            variable.value.sharding.spec,
        )

    model_output = lora_einsum(model_input)
    self.assertTrue(model_output.shape, (16, 4, 8, 10))
    self.assertEqual(
        model_output.sharding.spec,
        jax.sharding.PartitionSpec("fsdp", None, "tp"),
    )

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
      self.assertIsInstance(kernel, boxed_param.WithAux)
      self.assertIsInstance(kernel.array.qvalue, nn.Partitioned)
      self.assertEqual(kernel.array.qvalue.unbox().shape, (16, 32))
      self.assertEqual(kernel.array.qvalue.names, ("a", "b"))

    self.assertIsInstance(lora_a, nn.Partitioned)
    self.assertEqual(lora_a.unbox().shape, (16, 3))
    self.assertEqual(lora_a.names, ("a", None))
    self.assertIsInstance(lora_b, nn.Partitioned)
    self.assertEqual(lora_b.unbox().shape, (3, 32))
    self.assertEqual(lora_b.names, (None, "b"))

  def test_lora_dot_general_batch(self):
    """Test LoRA on a dot_general operation with batch dimensions."""
    batch, in_dim, contract, out_dim = 2, 4, 8, 16
    rank = 3
    alpha = 1.0

    class BatchLinear(nnx.Module):

      def __init__(self, rngs):
        # kernel shape: (batch, contract, out)
        self.kernel = nnx.Param(
            jax.random.normal(rngs.params(), (batch, contract, out_dim))
        )

      def __call__(self, x):
        # x: (batch, in, contract)
        # dim_nums: ((lhs_contract, rhs_contract), (lhs_batch, rhs_batch))
        return jax.lax.dot_general(
            x,
            self.kernel.value,
            (((2,), (1,)), ((0,), (0,))),
        )

    model = BatchLinear(nnx.Rngs(0))
    lora_provider = lora.LoraProvider([
        lora.LoraRule(
            module_path=".*",
            rank=rank,
            alpha=alpha,
            lora_b_initializer=nnx.initializers.ones,
        ),
    ])

    lhs = jnp.ones((batch, in_dim, contract))
    lora_model = lora.apply_lora_to_model(model, lora_provider, lhs)

    # Check if LoRA parameters were created and have expected shapes
    variables = nnx.variables(lora_model, nnx.LoRAParam)
    self.assertIn("kernel_lora_a", variables)
    self.assertIn("kernel_lora_b", variables)

    lora_a = variables["kernel_lora_a"].value
    lora_b = variables["kernel_lora_b"].value

    self.assertEqual(lora_a.shape, (batch, contract, rank))
    self.assertEqual(lora_b.shape, (batch, rank, out_dim))

    # Verify forward pass
    output = lora_model(lhs)
    self.assertEqual(output.shape, (batch, in_dim, out_dim))

    # Expected output
    res = jax.lax.dot_general(
        lhs, model.kernel.value, (((2,), (1,)), ((0,), (0,)))
    )
    delta_a = jax.lax.dot_general(lhs, lora_a, (((2,), (1,)), ((0,), (0,))))
    delta = jax.lax.dot_general(delta_a, lora_b, (((2,), (1,)), ((0,), (0,))))
    expected = res + delta * (alpha / rank)

    self.assertTrue(jnp.allclose(output, expected))

  def test_lora_dot_general_multi_out_dim(self):
    """Test LoRA on a dot_general operation with multiple output dimensions."""
    batch, in_dim, contract, out_dim1, out_dim2 = 2, 4, 8, 4, 4
    rank = 3
    alpha = 1.0

    class MultiOutLinear(nnx.Module):

      def __init__(self, rngs):
        # kernel shape: (batch, contract, out_dim1, out_dim2)
        self.kernel = nnx.Param(
            jax.random.normal(
                rngs.params(), (batch, contract, out_dim1, out_dim2)
            )
        )

      def __call__(self, x):
        # x: (batch, in, contract)
        # dim_nums: ((lhs_contract, rhs_contract), (lhs_batch, rhs_batch))
        return jax.lax.dot_general(
            x, self.kernel.value, (((2,), (1,)), ((0,), (0,)))
        )

    model = MultiOutLinear(nnx.Rngs(0))
    lora_provider = lora.LoraProvider(rank=rank, alpha=alpha)
    lhs = jnp.ones((batch, in_dim, contract))
    lora_model = lora.apply_lora_to_model(model, lora_provider, lhs)

    # Check if LoRA parameters were created and have expected shapes
    variables = nnx.variables(lora_model, nnx.LoRAParam)
    self.assertIn("kernel_lora_a", variables)
    self.assertIn("kernel_lora_b", variables)

    lora_a = variables["kernel_lora_a"].value
    lora_b = variables["kernel_lora_b"].value

    self.assertEqual(lora_a.shape, (batch, contract, rank))
    self.assertEqual(lora_b.shape, (batch, rank, out_dim1 * out_dim2))

    # Verify forward pass
    output = lora_model(lhs)
    self.assertEqual(output.shape, (batch, in_dim, out_dim1, out_dim2))

    # Expected output
    res = model(lhs)
    delta_a = jax.lax.dot_general(lhs, lora_a, (((2,), (1,)), ((0,), (0,))))
    lora_b_reshaped = jnp.reshape(lora_b, (batch, rank, out_dim1, out_dim2))
    delta = jax.lax.dot_general(
        delta_a, lora_b_reshaped, (((2,), (1,)), ((0,), (0,)))
    )
    expected = res + delta * (alpha / rank)

    self.assertTrue(jnp.allclose(output, expected, atol=1e-5))

  def test_lora_dot_general_no_weight(self):
    """Tests that LoRA skips operations where rhs is not a weight."""

    class NoWeightModel(nnx.Module):

      def __call__(self, x):
        # Multiply by a constant array, not a parameter
        return jax.lax.dot_general(
            x, jnp.ones((8, 8)), (((1,), (0,)), ((), ()))
        )

    model = NoWeightModel()
    lora_provider = lora.LoraProvider(rank=2, alpha=1.0)
    # This should work without error and without creating LoRA params
    lora_model = lora.apply_lora_to_model(
        model, lora_provider, jnp.ones((1, 8))
    )
    self.assertEmpty(nnx.variables(lora_model, nnx.LoRAParam))
    output = lora_model(jnp.ones((1, 8)))
    self.assertEqual(output.shape, (1, 8))

  def test_lora_dot_general_no_remain(self):
    """Tests LoRA where there are no remaining dimensions (full contraction)."""

    class FullContractModel(nnx.Module):

      def __init__(self, rngs):
        self.kernel = nnx.Param(jax.random.normal(rngs.params(), (8,)))

      def __call__(self, x):
        return jax.lax.dot_general(
            x, self.kernel.value, (((0,), (0,)), ((), ()))
        )

    model = FullContractModel(nnx.Rngs(0))
    lora_provider = lora.LoraProvider(rank=2, alpha=1.0)
    lora_model = lora.apply_lora_to_model(model, lora_provider, jnp.ones((8,)))
    output = lora_model(jnp.ones((8,)))
    self.assertEqual(output.shape, ())

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
    self.assertEqual(
        conv.kernel.value.sharding.spec,
        jax.sharding.PartitionSpec(None, None, "in", "out"),
    )

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
    self.assertEqual(
        lora_a.value.sharding.spec,
        jax.sharding.PartitionSpec(None, None, "in", None),
    )
    self.assertIsInstance(lora_b, nnx.LoRAParam)
    self.assertEqual(lora_b.shape, (3, 32))
    self.assertEqual(lora_b.sharding_names, (None, "out"))
    self.assertEqual(
        lora_b.value.sharding.spec, jax.sharding.PartitionSpec(None, "out")
    )

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
