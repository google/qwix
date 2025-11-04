"""PTQPad tests: parity with ptq_test but using PTQPadProvider and padding."""

import functools
import os

from absl.testing import absltest
from absl.testing import parameterized
from flax import linen as nn
from flax import nnx
import jax
from jax import export
from jax import numpy as jnp
from jax.experimental import pallas as pl
from qwix._src import flax_util
from qwix._src import model as qwix_model
from qwix._src import qconfig
from qwix._src.providers import qt
from qwix._src.providers import ptq_pad as ptq

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"

class PtqPadTest(parameterized.TestCase):

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
    ptq_dense = qwix_model.quantize_model(dense, ptq.PtqPadProvider(q_rules))
    model_input = jnp.ones((10, 13))
    ptq_params = ptq_dense.init(jax.random.key(0), model_input)["params"]
    ptq_abs_params = jax.eval_shape(
        ptq_dense.init, jax.random.key(0), model_input
    )["params"]
    qw = ptq_abs_params["kernel"]

    self.assertIsInstance(qw, ptq.WithAux)
    qw = qw.array
    self.assertIsInstance(qw.qvalue, nn.Partitioned)
    self.assertIsInstance(qw.scale, nn.Partitioned)
    self.assertEqual(qw.qvalue.value.dtype, jnp.int8)
    self.assertEqual(qw.qvalue.value.shape, (16, 5))
    self.assertEqual(qw.qvalue.names, ("contraction", "remaining"))
    self.assertEqual(qw.scale.value.shape, (4, 5))
    self.assertEqual(qw.scale.names, ("contraction", "remaining"))

    orig_params = dense.init(jax.random.key(0), model_input)["params"]
    orig_params = nn.unbox(orig_params)
    quantized_params = ptq.quantize_params(orig_params, ptq_abs_params)
    jax.tree.map(lambda *_: ..., quantized_params, ptq_abs_params)
    jax.tree.map_with_path(
        lambda kp, x, y: self.assertTrue(jnp.allclose(x, y), kp),
        quantized_params,
        nn.unbox(ptq_params),
    )
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

    _, new_vars = qt_dense.apply(
        qt_variables, model_input, mutable="quant_stats"
    )
    quant_stats = new_vars["quant_stats"]
    self.assertEqual(quant_stats["dot_general0_lhs"]["count"], 1)

    ptq_dense = qwix_model.quantize_model(dense, ptq.PtqPadProvider(q_rules))
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
    mesh = jax.make_mesh((2, 2), ("contraction", "remaining"))
    q_rules = [
        qconfig.QuantizationRule(
            module_path=".*", weight_qtype=jnp.int8, tile_size=4
        ),
    ]

    model_input = jnp.ones((10, 13))
    with jax.set_mesh(mesh):
      fp_linear = nnx.Linear(
          in_features=13,
          out_features=6,
          rngs=nnx.Rngs(0),
          kernel_init=nnx.with_partitioning(
              nnx.initializers.lecun_normal(), ("contraction", "remaining")
          ),
      )
      ptq_linear = qwix_model.quantize_model(
          fp_linear,
          ptq.PtqPadProvider(q_rules),
          model_input,
      )
    qw = ptq_linear.kernel
    self.assertIsInstance(qw, ptq.WithAux)
    qw = qw.array
    self.assertEqual(qw.qvalue.dtype, jnp.int8)
    self.assertEqual(qw.qvalue.shape, (16, 6))
    self.assertEqual(qw.qvalue.sharding, ("contraction", "remaining"))
    self.assertEqual(qw.scale.shape, (4, 6))
    self.assertEqual(qw.scale.sharding, ("contraction", "remaining"))

    with jax.set_mesh(mesh):
      abs_ptq_linear = nnx.eval_shape(
          lambda: qwix_model.quantize_model(
              fp_linear,
              ptq.PtqPadProvider(q_rules),
              model_input,
          ),
      )
    orig_params = nnx.state(fp_linear, nnx.Param)
    orig_params = nnx.to_pure_dict(orig_params)
    quantized_params = ptq.quantize_params(orig_params, abs_ptq_linear)
    nnx.update(abs_ptq_linear, quantized_params)
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

    model_input = jnp.ones((10, 1, 14))
    with jax.set_mesh(mesh):
      fp_einsum = nnx.Einsum(
          "btd,dnh->btnh",
          (14, 8, 10),
          (8, 10),
          rngs=nnx.Rngs(0),
          kernel_init=nnx.with_partitioning(
              nnx.initializers.lecun_normal(), ("fsdp", "tp", None)
          ),
          bias_init=nnx.with_partitioning(nnx.initializers.zeros, ("tp", None)),
      )

    unsharded_state = nnx.state(fp_einsum)
    sharding = nnx.get_named_sharding(unsharded_state, mesh)
    sharded_state = jax.device_put(unsharded_state, sharding)
    nnx.update(fp_einsum, sharded_state)
    self.assertEqual(fp_einsum.kernel.sharding, ("fsdp", "tp", None))
    self.assertEqual(fp_einsum.kernel.value.sharding.spec, ("fsdp", "tp", None))
    self.assertEqual(fp_einsum.bias.sharding, ("tp", None))
    self.assertEqual(fp_einsum.bias.value.sharding.spec, ("tp", None))

    with jax.set_mesh(mesh):
      # PTQ method 1: use quantize_model to convert both the model and params.
      ptq_einsum = qwix_model.quantize_model(
          fp_einsum,
          ptq.PtqPadProvider(q_rules),
          model_input,
      )

    def get_canonical_pspec(x: jax.Array):
      """The sharding.spec may be shorter than the ndim."""
      return x.sharding.spec + (None,) * (x.ndim - len(x.sharding.spec))

    qw = ptq_einsum.kernel
    self.assertIsInstance(qw, ptq.WithAux)
    qw = qw.array
    self.assertEqual(qw.qvalue.dtype, jnp.int8)
    self.assertEqual(qw.qvalue.shape, (16, 8, 10))
    self.assertEqual(qw.qvalue.sharding, ("fsdp", "tp", None))
    self.assertEqual(get_canonical_pspec(qw.qvalue.value), ("fsdp", "tp", None))
    self.assertEqual(qw.scale.shape, (4, 8, 10))
    self.assertEqual(qw.scale.sharding, ("fsdp", "tp", None))
    self.assertEqual(get_canonical_pspec(qw.scale.value), ("fsdp", "tp", None))

    # PTQ method 2: call quantize_model in eval_shape and quantize_params.
    with jax.set_mesh(mesh):
      abs_ptq_einsum = nnx.eval_shape(
          lambda: qwix_model.quantize_model(
              fp_einsum,
              ptq.PtqPadProvider(q_rules),
              model_input,
          ),
      )
    orig_params = nnx.state(fp_einsum, nnx.Param)
    orig_params = nnx.to_pure_dict(orig_params)
    quantized_params = ptq.quantize_params(orig_params, abs_ptq_einsum)
    nnx.update(abs_ptq_einsum, quantized_params)
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

    ptq_linear = qwix_model.quantize_model(
        qt_linear, ptq.PtqPadProvider(q_rules), model_input
    )
    self.assertIsInstance(ptq_linear.dot_general0_lhs_scale, ptq.WithAux)
    self.assertEqual(ptq_linear.dot_general0_lhs_scale.array.shape, (1, 1))
    if act_calibration_method == "minmax":
      self.assertEqual(ptq_linear.dot_general0_lhs_zero_point.shape, (1, 1))

    abs_ptq_linear = nnx.eval_shape(
        lambda: qwix_model.quantize_model(
            qt_linear, ptq.PtqPadProvider(q_rules), model_input
        )
    )
    qt_params = nnx.state(qt_linear, nnx.Param)
    quantized_params = ptq.quantize_params(
        nnx.to_pure_dict(qt_params),
        abs_ptq_linear,
        nnx.to_pure_dict(quant_stats),
    )
    nnx.update(abs_ptq_linear, quantized_params)
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
        assert qconfig.get_current_rule("dot_general") is not None

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
        tile_size=4,
        )
    ]
    ptq_model = qwix_model.quantize_model(model, ptq.PtqPadProvider(q_rules))
    variables = ptq_model.init(jax.random.key(0), jnp.ones((16, 32)))
    self.assertIsInstance(variables["params"]["w1"], ptq.WithAux)
    self.assertIsInstance(variables["params"]["w2"], jax.Array)
    ptq_model.apply(variables, jnp.ones((16, 32)))

  def test_symbolic_export(self):
    """Test jax export with symbolic shape."""
    model = nn.Dense(features=5)
    q_rules = [
        qconfig.QuantizationRule(
            module_path=".*",
            weight_qtype=jnp.int8,
            act_qtype=jnp.int8,
            tile_size=4,
        )
    ]
    ptq_model = qwix_model.quantize_model(model, ptq.PtqPadProvider(q_rules))
    variables = ptq_model.init(jax.random.key(0), jnp.ones((16, 32)))
    (b,) = export.symbolic_shape("b")
    exp = export.export(jax.jit(ptq_model.apply))(
        variables, jax.ShapeDtypeStruct((b, 32), jnp.float32)
    )
    self.assertEqual(exp.out_avals[0].shape, (b, 5))

  def test_nnx_scan(self):
    """Test nnx.scan with PTQ."""

    class ScanModel(nnx.Module):

      def __init__(self, n_layers: int, rngs: nnx.Rngs):
        @nnx.split_rngs(splits=n_layers)
        @nnx.vmap(axis_size=n_layers)
        def create_layer(rngs: nnx.Rngs):
          return nnx.Linear(in_features=12, out_features=12, rngs=rngs)

        self.layers = create_layer(rngs)

      def __call__(self, x):
        @nnx.scan(out_axes=nnx.Carry)
        def scan_fn(x: jax.Array, layer):
          return layer(x)

        return scan_fn(x, self.layers)

    model = ScanModel(n_layers=2, rngs=nnx.Rngs(0))
    self.assertEqual(model.layers.kernel.value.shape, (2, 12, 12))

    q_rules = [qconfig.QuantizationRule(weight_qtype=jnp.int8)]
    model_input = jnp.ones((10, 12))
    ptq_model = qwix_model.quantize_model(
        model, ptq.PtqPadProvider(q_rules), model_input
    )
    self.assertIsInstance(ptq_model.layers.kernel, ptq.WithAux)
    self.assertIsInstance(ptq_model.layers.kernel.array, ptq.qarray.QArray)
    self.assertEqual(ptq_model.layers.kernel.array.shape, (2, 12, 12))
    self.assertEqual(ptq_model.layers.kernel.array.qtype, jnp.int8)

    # Ensure that the model can be called.
    ptq_model(model_input)


if __name__ == "__main__":
  absltest.main()