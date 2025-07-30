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
import functools
import os

from absl.testing import absltest
from absl.testing import parameterized
from flax import linen as nn
from flax import nnx
import jax
from jax import numpy as jnp
from jax.experimental import pallas as pl
from qwix._src import flax_util
from qwix._src import model as qwix_model
from qwix._src import qconfig
from qwix._src.providers import ptq
from qwix._src.providers import qt

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"


class PtqTest(parameterized.TestCase):

  def test_nn_ptq(self):
    dense = nn.Dense(
        features=5,
        kernel_init=nn.with_partitioning(
            nn.initializers.lecun_normal(), ("contraction", "remaining")
        ),
    )
    q_rules = [
        qconfig.QuantizationRule(
            module_path=".*", weight_qtype=jnp.int8, tile_size=4
        ),
    ]
    ptq_dense = qwix_model.quantize_model(dense, ptq.PtqProvider(q_rules))
    model_input = jnp.ones((10, 12))
    # PTQ model can be initialized, both actually and in eval_shape.
    ptq_params = ptq_dense.init(jax.random.key(0), model_input)["params"]
    ptq_abs_params = jax.eval_shape(
        ptq_dense.init, jax.random.key(0), model_input
    )["params"]
    qw = ptq_abs_params["kernel"]

    # Test PTQ param structure.
    self.assertIsInstance(qw, ptq.WithAux)
    qw = qw.array
    self.assertIsInstance(qw.qvalue, nn.Partitioned)
    self.assertIsInstance(qw.scale, nn.Partitioned)
    self.assertEqual(qw.qvalue.value.dtype, jnp.int8)
    self.assertEqual(qw.qvalue.value.shape, (12, 5))
    self.assertEqual(qw.qvalue.names, ("contraction", "remaining"))
    self.assertEqual(qw.scale.value.shape, (3, 5))
    self.assertEqual(qw.scale.names, ("contraction", "remaining"))

    # Test param quantization.
    orig_params = dense.init(jax.random.key(0), model_input)["params"]
    orig_params = nn.unbox(orig_params)
    quantized_params = ptq.quantize_params(orig_params, ptq_abs_params)
    # They should have the same structure.
    jax.tree.map(lambda *_: ..., quantized_params, ptq_abs_params)
    # The result should be the same as ptq_params, because ptq_dense.init
    # also performs the correct quantization for the params.
    jax.tree.map_with_path(
        lambda kp, x, y: self.assertTrue(jnp.allclose(x, y), kp),
        quantized_params,
        nn.unbox(ptq_params),
    )
    # Ensure that the model can be called.
    ptq_dense.apply({"params": quantized_params}, model_input)

    # nn model shouldn't allow implicit quantization.
    with self.assertRaises(ValueError):
      ptq_dense.apply({"params": orig_params}, model_input)

  @parameterized.parameters("absmax", "minmax", "rms,7")
  def test_nn_srq(self, act_calibration_method):
    dense = nn.Dense(features=5)
    q_rules = [
        qconfig.QuantizationRule(
            module_path=".*",
            weight_qtype=jnp.int8,
            act_qtype=jnp.int8,
            act_static_scale=True,
            act_calibration_method=act_calibration_method,
        ),
    ]

    # QT to generate quant_stats.
    qt_dense = qwix_model.quantize_model(dense, qt.QtProvider(q_rules))
    model_input = jnp.ones((10, 12))
    qt_variables = qt_dense.init(jax.random.key(0), model_input)
    quant_stat = qt_variables["quant_stats"]["dot_general0_lhs"]
    self.assertEqual(quant_stat["count"].shape, ())
    self.assertEqual(quant_stat["count"], 0)
    if act_calibration_method == "minmax":
      self.assertEqual(quant_stat["sum_of_min"].shape, (1, 1))
      self.assertEqual(quant_stat["sum_of_max"].shape, (1, 1))
    else:
      self.assertEqual(quant_stat["sum_of_absmax"].shape, (1, 1))

    # Run the model to initialize the quant_stats.
    _, new_vars = qt_dense.apply(
        qt_variables, model_input, mutable="quant_stats"
    )
    quant_stats = new_vars["quant_stats"]
    self.assertEqual(quant_stats["dot_general0_lhs"]["count"], 1)

    # PTQ.
    ptq_dense = qwix_model.quantize_model(dense, ptq.PtqProvider(q_rules))
    ptq_abs_params = jax.eval_shape(
        ptq_dense.init, jax.random.key(0), model_input
    )["params"]
    self.assertIn("dot_general0_lhs_scale", ptq_abs_params)
    self.assertIsInstance(ptq_abs_params["dot_general0_lhs_scale"], ptq.WithAux)
    self.assertEqual(
        ptq_abs_params["dot_general0_lhs_scale"].array.shape, (1, 1)
    )
    if act_calibration_method == "minmax":
      self.assertIn("dot_general0_lhs_zero_point", ptq_abs_params)
      self.assertEqual(
          ptq_abs_params["dot_general0_lhs_zero_point"].shape, (1, 1)
      )

    quantized_params = ptq.quantize_params(
        qt_variables["params"], ptq_abs_params, quant_stats
    )
    # They should have the same structure.
    jax.tree.map(lambda *_: ..., quantized_params, ptq_abs_params)
    # Ensure that the model can be called.
    ptq_dense.apply({"params": quantized_params}, model_input)

  def test_nnx_ptq(self):
    q_rules = [
        qconfig.QuantizationRule(
            module_path=".*", weight_qtype=jnp.int8, tile_size=4
        ),
    ]

    model_input = jnp.ones((10, 12))
    fp_linear = nnx.Linear(
        in_features=12,
        out_features=5,
        rngs=nnx.Rngs(0),
        kernel_init=nnx.with_partitioning(
            nnx.initializers.lecun_normal(), ("contraction", "remaining")
        ),
    )
    # Weight quantization method 1: use quantize_model to convert both the
    # model and params, i.e., implicit quantization.
    ptq_linear = qwix_model.quantize_model(
        fp_linear,
        ptq.PtqProvider(q_rules),
        model_input,
    )
    # Test PTQ param structure.
    qw = ptq_linear.kernel
    self.assertIsInstance(qw, ptq.WithAux)
    self.assertEqual(qw.weight_name, "kernel")
    qw = qw.array
    self.assertEqual(qw.qvalue.dtype, jnp.int8)
    self.assertEqual(qw.qvalue.shape, (12, 5))
    self.assertEqual(qw.qvalue.sharding, ("contraction", "remaining"))
    self.assertEqual(qw.scale.shape, (3, 5))
    self.assertEqual(qw.scale.sharding, ("contraction", "remaining"))

    # Weight quantization method 2: call quantize_model in eval_shape and
    # quantize_params.
    abs_ptq_linear = nnx.eval_shape(
        lambda: qwix_model.quantize_model(
            fp_linear,
            ptq.PtqProvider(q_rules),
            model_input,
        ),
    )
    # Test manual quantize_params.
    orig_params = nnx.state(fp_linear, nnx.Param)
    orig_params = nnx.to_pure_dict(orig_params)
    quantized_params = ptq.quantize_params(orig_params, abs_ptq_linear)
    # quantized_params can be updated to abs_ptq_linear.
    nnx.update(abs_ptq_linear, quantized_params)
    # Ensure that the model can be called.
    abs_ptq_linear(model_input)

    # The two methods should produce the same result.
    jax.tree.map_with_path(
        lambda kp, x, y: self.assertTrue(jnp.allclose(x, y), f"{kp} {x} {y}"),
        nnx.state(abs_ptq_linear),
        nnx.state(ptq_linear),
    )

  def test_nnx_einsum_sharding_ptq(self):
    mesh = jax.make_mesh((2, 2), ("fsdp", "tp"))
    q_rules = [
        qconfig.QuantizationRule(
            module_path=".*", weight_qtype=jnp.int8, tile_size=4
        ),
    ]

    model_input = jnp.ones((10, 1, 16))
    fp_einsum = nnx.Einsum(
        "btd,dnh->btnh",
        (16, 8, 10),
        (8, 10),
        rngs=nnx.Rngs(0),
        kernel_init=nnx.with_partitioning(
            nnx.initializers.lecun_normal(), ("fsdp", "tp", None)
        ),
        bias_init=nnx.with_partitioning(nnx.initializers.zeros, ("tp", None)),
    )

    # Shard the fp_einsum model in-place.
    unsharded_state = nnx.state(fp_einsum)
    sharding = nnx.get_named_sharding(unsharded_state, mesh)
    sharded_state = jax.device_put(unsharded_state, sharding)
    nnx.update(fp_einsum, sharded_state)
    self.assertEqual(fp_einsum.kernel.sharding, ("fsdp", "tp", None))
    self.assertEqual(fp_einsum.kernel.value.sharding.spec, ("fsdp", "tp", None))
    self.assertEqual(fp_einsum.bias.sharding, ("tp", None))
    self.assertEqual(fp_einsum.bias.value.sharding.spec, ("tp", None))

    # PTQ method 1: use quantize_model to convert both the model and params.
    ptq_einsum = qwix_model.quantize_model(
        fp_einsum,
        ptq.PtqProvider(q_rules),
        model_input,
    )

    def get_canonical_pspec(x: jax.Array):
      """The sharding.spec may be shorter than the ndim."""
      return x.sharding.spec + (None,) * (x.ndim - len(x.sharding.spec))

    # Test PTQ param structure.
    qw = ptq_einsum.kernel
    self.assertIsInstance(qw, ptq.WithAux)
    self.assertEqual(qw.weight_name, "kernel")
    qw = qw.array
    self.assertEqual(qw.qvalue.dtype, jnp.int8)
    self.assertEqual(qw.qvalue.shape, (16, 8, 10))
    self.assertEqual(qw.qvalue.sharding, ("fsdp", "tp", None))
    self.assertEqual(get_canonical_pspec(qw.qvalue.value), qw.qvalue.sharding)
    self.assertEqual(qw.scale.shape, (4, 8, 10))
    self.assertEqual(qw.scale.sharding, ("fsdp", "tp", None))
    self.assertEqual(get_canonical_pspec(qw.scale.value), qw.scale.sharding)

    # PTQ method 2: call quantize_model in eval_shape and quantize_params.
    abs_ptq_einsum = nnx.eval_shape(
        lambda: qwix_model.quantize_model(
            fp_einsum,
            ptq.PtqProvider(q_rules),
            model_input,
        ),
    )
    # Test manual quantize_params.
    orig_params = nnx.state(fp_einsum, nnx.Param)
    orig_params = nnx.to_pure_dict(orig_params)
    quantized_params = ptq.quantize_params(orig_params, abs_ptq_einsum)
    # quantized_params can be updated to abs_ptq_einsum.
    nnx.update(abs_ptq_einsum, quantized_params)
    # Ensure that the model can be called.
    abs_ptq_einsum(model_input)

    # The two methods should produce the same sharding.
    jax.tree.map_with_path(
        lambda kp, x, y: self.assertEqual(
            x.sharding, y.sharding, f"{kp} {x.sharding} {y.sharding}"
        ),
        nnx.to_pure_dict(nnx.state(abs_ptq_einsum)),
        nnx.to_pure_dict(nnx.state(ptq_einsum)),
    )

  @parameterized.parameters("absmax", "minmax")
  def test_nnx_srq(self, act_calibration_method):
    q_rules = [
        qconfig.QuantizationRule(
            module_path=".*",
            weight_qtype=jnp.int8,
            act_qtype=jnp.int8,
            act_static_scale=True,
            act_calibration_method=act_calibration_method,
        ),
    ]

    # QT to generate quant_stats.
    model_input = jnp.ones((10, 12))
    qt_linear = qwix_model.quantize_model(
        nnx.Linear(in_features=12, out_features=5, rngs=nnx.Rngs(0)),
        qt.QtProvider(q_rules),
        model_input,
    )
    qt_linear(model_input)
    quant_stats = nnx.state(qt_linear, flax_util.QuantStat)
    quant_stat = quant_stats["dot_general0_lhs"].value

    self.assertEqual(quant_stat["count"].shape, ())
    self.assertEqual(quant_stat["count"], 1)
    if act_calibration_method == "minmax":
      self.assertEqual(quant_stat["sum_of_min"].shape, (1, 1))
      self.assertEqual(quant_stat["sum_of_max"].shape, (1, 1))
    else:
      self.assertEqual(quant_stat["sum_of_absmax"].shape, (1, 1))

    # PTQ method 1: quantize_model also converts params and quant_stats.
    ptq_linear = qwix_model.quantize_model(
        qt_linear, ptq.PtqProvider(q_rules), model_input
    )
    self.assertIsInstance(ptq_linear.dot_general0_lhs_scale, ptq.WithAux)
    self.assertEqual(ptq_linear.dot_general0_lhs_scale.array.shape, (1, 1))
    if act_calibration_method == "minmax":
      self.assertEqual(ptq_linear.dot_general0_lhs_zero_point.shape, (1, 1))

    # PTQ method 2: manually call quantize_params.
    abs_ptq_linear = nnx.eval_shape(
        lambda: qwix_model.quantize_model(
            qt_linear, ptq.PtqProvider(q_rules), model_input
        )
    )
    qt_params = nnx.state(qt_linear, nnx.Param)
    quantized_params = ptq.quantize_params(
        nnx.to_pure_dict(qt_params),
        abs_ptq_linear,
        nnx.to_pure_dict(quant_stats),
    )
    # quantized_params can be updated to abs_ptq_linear.
    nnx.update(abs_ptq_linear, quantized_params)
    # Ensure that the model can be called.
    abs_ptq_linear(model_input)

    # The two methods should produce the same result.
    jax.tree.map_with_path(
        lambda kp, x, y: self.assertTrue(jnp.allclose(x, y), f"{kp} {x} {y}"),
        nnx.state(abs_ptq_linear),
        nnx.state(ptq_linear),
    )

  def test_dot_pallas_call(self):
    """pallas_call should not be intercepted."""

    class Model(nn.Module):

      @nn.compact
      def __call__(self, x):
        w1 = self.param("w1", nn.initializers.ones, (x.shape[-1], 1))
        w2 = self.param("w2", nn.initializers.ones, (1, x.shape[-1]))
        out_shape = jax.ShapeDtypeStruct(x.shape, x.dtype)

        @functools.partial(pl.pallas_call, out_shape=out_shape, interpret=True)
        def pallas_dot(x, y, out):
          out[...] = jax.lax.dot(x[...], y[...])

        return pallas_dot(jax.numpy.dot(x, w1), w2)

    model = Model()
    q_rules = [
        qconfig.QuantizationRule(
            module_path=".*",
            weight_qtype=jnp.int8,
            act_qtype=jnp.int8,
            tile_size=4,  # will trigger an error if pallas_dot is intercepted.
        )
    ]
    ptq_model = qwix_model.quantize_model(model, ptq.PtqProvider(q_rules))
    variables = ptq_model.init(jax.random.key(0), jnp.ones((16, 32)))
    self.assertIsInstance(variables["params"]["w1"], ptq.WithAux)
    self.assertIsInstance(variables["params"]["w2"], jax.Array)
    ptq_model.apply(variables, jnp.ones((16, 32)))


if __name__ == "__main__":
  absltest.main()
