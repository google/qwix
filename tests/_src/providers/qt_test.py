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

import itertools
import os
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from flax import linen as nn
from flax import nnx
import jax
from jax import numpy as jnp
from qwix._src import model as qwix_model
from qwix._src import qconfig
from qwix._src.core import sparsity
from qwix._src.providers import qt
from qwix._src.utils import flax_util

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"


class QtTest(parameterized.TestCase):

  def test_bwd_reuse_noise(self):
    """Tests that noise is reused in bwd pass for lhs/rhs."""

    class TestModule(nn.Module):
      provider: qt.QtProvider

      def test_config(self, lhs, rhs):
        return self.provider._create_dot_general_qt_config(
            self.provider._rules[0], "dot_general0", lhs, rhs
        )

    lhs = jnp.ones((2, 3), dtype=jnp.bfloat16)
    rhs = jnp.ones((2, 3), dtype=jnp.bfloat16)

    provider_reuse = qt.QtProvider([
        qt.QtRule(
            bwd_qtype=jnp.int8,
            bwd_stochastic_rounding="low_bit_uniform",
        ),
    ])
    module_reuse = TestModule(provider_reuse)
    rngs = {"stochastic_rounding": jax.random.key(0)}
    config_reuse = module_reuse.apply(
        {}, lhs, rhs, rngs=rngs, method=TestModule.test_config
    )

    self.assertIsNotNone(config_reuse.dlhs_stochastic_rounding_noise_fn)
    self.assertIsNotNone(config_reuse.drhs_stochastic_rounding_noise_fn)
    self.assertIs(
        config_reuse.dlhs_stochastic_rounding_noise_fn,
        config_reuse.drhs_stochastic_rounding_noise_fn,
    )

    shape = (8, 32)
    noise_lhs_reuse = config_reuse.dlhs_stochastic_rounding_noise_fn(shape)
    noise_rhs_reuse = config_reuse.drhs_stochastic_rounding_noise_fn(shape)
    self.assertTrue(jnp.array_equal(noise_lhs_reuse, noise_rhs_reuse))

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
    """Test creating and train an SRQ NNX model inside jit."""

    def create_srq_nnx_model(model_input):
      linear = nnx.Linear(12, 10, rngs=nnx.Rngs(0), param_dtype=jnp.bfloat16)
      qt_provider = qt.QtProvider([
          qconfig.QuantizationRule(
              weight_qtype=jnp.int8,
              act_qtype=jnp.int8,
              act_static_scale=True,
          ),
      ])
      return qwix_model.quantize_model(linear, qt_provider, model_input)

    model_input = jnp.ones((9, 12), dtype=jnp.float32)
    qt_linear = nnx.jit(create_srq_nnx_model)(model_input)
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

  def test_mxfp4_requires_tile_size(self):
    """QT does not default the tile size of microscaled formats (yet)."""
    rule = qt.QtRule(module_path=".*", weight_qtype="mxfp4")
    linear = nnx.Linear(64, 16, rngs=nnx.Rngs(0))
    with self.assertRaisesRegex(ValueError, "requires `tiled_axes`"):
      qwix_model.quantize_model(
          linear, qt.QtProvider([rule]), jnp.ones((2, 64))
      )

  def test_dot_general_with_sparsity(self):
    """Tests that dot_general applies sparsity to rhs."""
    lhs = jnp.ones((1, 4), dtype=jnp.bfloat16)
    rhs = jnp.ones((4, 1), dtype=jnp.bfloat16)

    rule = qt.QtRule(
        weight_qtype=jnp.int8,
        additional_qt_config={
            "sparsity_rule": sparsity.SparsityRule(
                weight_sparsity_n=2,
                weight_sparsity_m=4,
            )
        },
    )
    provider = qt.QtProvider([rule])

    class TestModule(nn.Module):
      provider: qt.QtProvider

      def __call__(self, lhs, rhs):
        dimension_numbers = (((1,), (0,)), ((), ()))
        return self.provider.dot_general(lhs, rhs, dimension_numbers)

    module = TestModule(provider)
    out = module.apply({}, lhs, rhs)

    # 2:4 sparsity means 2 elements non-zero in each block of 4.
    # Since rhs is all ones, sparsifying it will make 2 elements zero.
    # So the dot product of ones(1,4) and sparsified ones(4,1) should be 2.0.
    self.assertEqual(out, jnp.array([[2.0]], dtype=jnp.bfloat16))

  def test_nnx_multi_head_attention_qt_bwd(self):
    rule = qt.QtRule(
        module_path=".*",
        weight_qtype="mxfp4",
        act_qtype="mxfp4",
        bwd_qtype="mxfp4",
        tile_size=32,
        bwd_weight_grad_tile_size=32,
        disable_channelwise_axes=False,
        act_static_scale=False,
        additional_qt_config={
            "dlhs_grad_qtype": "mxfp4",
            "dlhs_tile_size": 32,
            "drhs_grad_qtype": "mxfp4",
            "drhs_tile_size": 32,
        },
    )
    model_input = jnp.ones((2, 32, 64), dtype=jnp.float32)
    mha = nnx.MultiHeadAttention(
        num_heads=4,
        in_features=64,
        qkv_features=128,
        out_features=64,
        rngs=nnx.Rngs(0),
        param_dtype=jnp.float32,
    )
    qt_mha = qwix_model.quantize_model(
        mha,
        qt.QtProvider([rule]),
        model_input,
        decode=False,
    )

    @nnx.jit
    def train_step(model, x):
      def loss_fn(model, x):
        out = model(x, decode=False)
        return jnp.sum(out)

      return nnx.grad(loss_fn)(model, x)

    grads = train_step(qt_mha, model_input)
    self.assertIsNotNone(grads)

  @parameterized.parameters(itertools.product([True, False], repeat=3))
  def test_nnx_linear_qt_2d_bwd(
      self,
      tsb: bool,
      bwd_weight_grad_tsb: bool,
      dlhs_tsb: bool,
  ):
    """Tests that QtProvider is compatible with 2D tile sizes."""
    # tsb is short for tile_size_bool

    # If True, use a 2D tile size, otherwise use a scalar tile size.
    def _get_tile_size(use_multi_axis: bool):
      if use_multi_axis:
        return {0: 2, 1: 2}
      return 2

    rule = qt.QtRule(
        module_path=".*",
        weight_qtype="int8",
        act_qtype="int8",
        bwd_qtype="int8",
        tile_size=_get_tile_size(tsb),
        bwd_weight_grad_tile_size=_get_tile_size(bwd_weight_grad_tsb),
        disable_channelwise_axes=False,
        act_static_scale=False,
        additional_qt_config={
            "dlhs_grad_qtype": "int8",
            "dlhs_tile_size": _get_tile_size(dlhs_tsb),
            "drhs_grad_qtype": "int8",
        },
    )
    model_input = jnp.ones((8, 16), dtype=jnp.float32)
    linear = nnx.Linear(16, 32, rngs=nnx.Rngs(0))
    qt_linear = qwix_model.quantize_model(
        linear,
        qt.QtProvider([rule]),
        model_input,
    )

    # Training function
    @nnx.jit
    def train_step(model, x):
      def loss_fn(model, x):
        out = model(x)
        return jnp.sum(out)

      return nnx.grad(loss_fn)(model, x)

    grads = train_step(qt_linear, model_input)
    self.assertIsNotNone(grads)

  def test_explicit_sharding_dot_general_and_einsum(self):
    """Tests QtProvider.dot_general and einsum with explicit out_sharding."""
    mesh = jax.make_mesh(
        (2,),
        ("x",),
        axis_types=(jax.sharding.AxisType.Explicit,),
        devices=jax.devices()[:2],
    )
    pspec = jax.sharding.PartitionSpec
    rule = qt.QtRule(
        module_path=".*",
        weight_qtype="int8",
        act_qtype="int8",
        bwd_qtype="int8",
    )
    provider = qt.QtProvider([rule])

    class DotModule(nn.Module):

      @nn.compact
      def __call__(self, l, r):
        return jax.lax.dot_general(
            l, r, (((1,), (0,)), ((), ())), out_sharding=pspec("x", None)
        )

    class EinsumModule(nn.Module):

      @nn.compact
      def __call__(self, l, r):
        return jnp.einsum("ab,bc->ac", l, r, out_sharding=pspec("x", None))

    qt_dot = qwix_model.quantize_model(DotModule(), provider)
    qt_einsum = qwix_model.quantize_model(EinsumModule(), provider)

    with jax.set_mesh(mesh):
      lhs = jax.device_put(
          jnp.ones((4, 8), dtype=jnp.float32), pspec("x", None)
      )
      rhs = jax.device_put(
          jnp.ones((8, 4), dtype=jnp.float32), pspec(None, None)
      )

      def fn_dot(l, r):
        out = qt_dot.apply({}, l, r)
        self.assertEqual(jax.typeof(out).sharding.spec, pspec("x", None))
        return jnp.sum(out)

      dlhs, drhs = jax.grad(fn_dot, argnums=(0, 1))(lhs, rhs)
      self.assertEqual(dlhs.sharding.spec, pspec("x", None))
      self.assertEqual(drhs.sharding.spec, pspec(None, None))

      def fn_einsum(l, r):
        out = qt_einsum.apply({}, l, r)
        self.assertEqual(jax.typeof(out).sharding.spec, pspec("x", None))
        return jnp.sum(out)

      dlhs_ein, drhs_ein = jax.grad(fn_einsum, argnums=(0, 1))(lhs, rhs)
      self.assertEqual(dlhs_ein.sharding.spec, pspec("x", None))
      self.assertEqual(drhs_ein.sharding.spec, pspec(None, None))

      rhs_reduced = jax.device_put(
          jnp.ones((8, 4), dtype=jnp.float32), pspec(None, None, reduced={"x"})
      )
      dlhs_red, drhs_red = jax.grad(fn_dot, argnums=(0, 1))(lhs, rhs_reduced)
      self.assertEqual(dlhs_red.sharding.spec, pspec("x", None))
      self.assertEqual(
          drhs_red.sharding.spec, pspec(None, None, unreduced={"x"})
      )

  def test_unquantized_dot_general_passes_no_unset_options(self):
    """JAX versions that predate an option must not get it from Qwix."""

    class Module(nn.Module):
      provider: qt.QtProvider

      @nn.compact
      def __call__(self, x, y):
        return self.provider.dot_general(x, y, (((1,), (0,)), ((), ())))

    x = jnp.ones((2, 3), dtype=jnp.float32)
    y = jnp.ones((3, 4), dtype=jnp.float32)
    with mock.patch.object(
        jax.lax, "dot_general", wraps=jax.lax.dot_general
    ) as jax_dot_general:
      Module(qt.QtProvider([])).apply({}, x, y)
    self.assertEqual(
        jax_dot_general.call_args.kwargs,
        dict(precision=None, preferred_element_type=None, out_sharding=None),
    )

  def test_unquantized_dot_general_passes_options_through(self):
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
    qt_mod = qwix_model.quantize_model(mod, qt.QtProvider([]))
    with self.assertRaises(TypeError) as qt_error:
      qt_mod.apply({}, x, y, new_option="x")
    self.assertEqual(str(qt_error.exception), str(jax_error.exception))

  def test_quantized_dot_general_options(self):
    class Module(nn.Module):

      @nn.compact
      def __call__(self, x, y, **options):
        return jax.lax.dot_general(x, y, (((1,), (0,)), ((), ())), **options)

    mod = qwix_model.quantize_model(
        Module(), qt.QtProvider([qt.QtRule(weight_qtype=jnp.int8)])
    )
    x = jnp.ones((2, 3), dtype=jnp.float32)
    y = jnp.ones((3, 4), dtype=jnp.float32)
    # Options that Qwix does not know are ignored, like precision.
    self.assertEqual(mod.apply({}, x, y, new_option="x").shape, (2, 4))
    with self.assertRaisesRegex(NotImplementedError, "reduction_order"):
      mod.apply({}, x, y, reduction_order="linear")

  @parameterized.named_parameters(
      dict(testcase_name="dot", op=jnp.dot),
      dict(testcase_name="matmul", op=jnp.matmul),
  )
  def test_quantized_jnp_dot_and_matmul(self, op):
    class OpModule(nn.Module):

      @nn.compact
      def __call__(self, x, y):
        return op(x, y)

    qt_mod = qwix_model.quantize_model(
        OpModule(), qt.QtProvider([qt.QtRule(weight_qtype=jnp.int8)])
    )
    x = jnp.ones((2, 3), dtype=jnp.float32)
    y = jnp.ones((3, 4), dtype=jnp.float32)
    res = qt_mod.apply({}, x, y)
    self.assertEqual(res.shape, (2, 4))

  @parameterized.product(
      op=["dot_general", "einsum"],
      bwd_qtype=[None, jnp.int8],
  )
  def test_preferred_element_type_and_bwd(self, op, bwd_qtype):
    class Mod(nn.Module):

      @nn.compact
      def __call__(self, x):
        w = self.param("w", nn.initializers.normal(), (8, 4), jnp.bfloat16)
        if op == "dot_general":
          return jax.lax.dot_general(
              x,
              w,
              (((1,), (0,)), ((), ())),
              preferred_element_type=jnp.float32,
          )
        return jnp.einsum("ab,bc->ac", x, w, preferred_element_type=jnp.float32)

    rule = qt.QtRule(
        weight_qtype=jnp.float8_e4m3fn,
        act_qtype=jnp.float8_e4m3fn,
        bwd_qtype=bwd_qtype,
    )
    qt_mod = qwix_model.quantize_model(Mod(), qt.QtProvider([rule]))
    x = jax.random.normal(jax.random.key(0), (2, 8), dtype=jnp.bfloat16)
    variables = qt_mod.init(jax.random.key(1), x)
    out = qt_mod.apply(variables, x)
    self.assertEqual(out.dtype, jnp.float32)
    self.assertFalse(
        jnp.array_equal(out, out.astype(jnp.bfloat16).astype(jnp.float32))
    )
    dx, dw = jax.grad(
        lambda x, p: qt_mod.apply({"params": p}, x).sum(), argnums=(0, 1)
    )(x, variables["params"])
    self.assertEqual(dx.dtype, jnp.bfloat16)
    self.assertEqual(dw["w"].dtype, jnp.bfloat16)

  @parameterized.parameters("dot_general", "einsum")
  def test_additional_qt_config_preferred_element_type(self, op):
    class Mod(nn.Module):
      call_preferred_dtype: jax.typing.DTypeLike | None = None

      @nn.compact
      def __call__(self, x):
        w = self.param("w", nn.initializers.normal(), (8, 4), jnp.bfloat16)
        if op == "dot_general":
          return jax.lax.dot_general(
              x,
              w,
              (((1,), (0,)), ((), ())),
              preferred_element_type=self.call_preferred_dtype,
          )
        return jnp.einsum(
            "ab,bc->ac",
            x,
            w,
            preferred_element_type=self.call_preferred_dtype,
        )

    rule = qt.QtRule(
        weight_qtype=jnp.float8_e4m3fn,
        act_qtype=jnp.float8_e4m3fn,
        additional_qt_config={"preferred_element_type": jnp.float32},
    )
    x = jax.random.normal(jax.random.key(0), (2, 8), dtype=jnp.bfloat16)

    qt_mod = qwix_model.quantize_model(Mod(), qt.QtProvider([rule]))
    variables = qt_mod.init(jax.random.key(1), x)
    out = qt_mod.apply(variables, x)
    self.assertEqual(out.dtype, jnp.float32)
    self.assertFalse(
        jnp.array_equal(out, out.astype(jnp.bfloat16).astype(jnp.float32))
    )

    qt_override = qwix_model.quantize_model(
        Mod(call_preferred_dtype=jnp.bfloat16), qt.QtProvider([rule])
    )
    out_override = qt_override.apply(variables, x)
    self.assertEqual(out_override.dtype, jnp.bfloat16)

  def test_weight_sparsity_not_implemented(self):
    rule = qconfig.QuantizationRule(
        weight_qtype=jnp.float8_e4m3fn,
        weight_sparsity=sparsity.StructuredSparsity(1, 4),
    )
    with self.assertRaisesRegex(
        NotImplementedError, "QtProvider does not support weight_sparsity"
    ):
      qt.QtProvider([rule])


if __name__ == "__main__":
  absltest.main()
