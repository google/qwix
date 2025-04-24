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
from absl.testing import parameterized
from flax import linen as nn
from flax import nnx
import jax
from jax import numpy as jnp
from qwix import flax_util
from qwix import model as qwix_model
from qwix import ptq
from qwix import qat
from qwix import qconfig


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
    self.assertEqual(qw.qvalue.value.shape, (3, 4, 5))  # split from (12, 5)
    self.assertEqual(qw.qvalue.names, ("contraction", None, "remaining"))
    self.assertEqual(qw.scale.value.shape, (3, 1, 5))  # transposed
    self.assertEqual(qw.scale.names, ("contraction", None, "remaining"))

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

    # QAT to generate quant_stats.
    qat_dense = qwix_model.quantize_model(dense, qat.QatProvider(q_rules))
    model_input = jnp.ones((10, 12))
    qat_variables = qat_dense.init(jax.random.key(0), model_input)
    quant_stat = qat_variables["quant_stats"]["dot_general0_lhs"]
    self.assertEqual(quant_stat["count"].shape, ())
    self.assertEqual(quant_stat["count"], 0)
    if act_calibration_method == "minmax":
      self.assertEqual(quant_stat["sum_of_min"].shape, (1, 1))
      self.assertEqual(quant_stat["sum_of_max"].shape, (1, 1))
    else:
      self.assertEqual(quant_stat["sum_of_absmax"].shape, (1, 1))

    # Run the model to initialize the quant_stats.
    _, new_vars = qat_dense.apply(
        qat_variables, model_input, mutable="quant_stats"
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
        qat_variables["params"], ptq_abs_params, quant_stats
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
    # PTQ method 1: use quantize_model to convert both the model and params.
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
    self.assertEqual(qw.qvalue.shape, (3, 4, 5))  # split from (12, 5)
    self.assertEqual(qw.qvalue.sharding, ("contraction", None, "remaining"))
    self.assertEqual(qw.scale.shape, (3, 1, 5))  # transposed
    self.assertEqual(qw.scale.sharding, ("contraction", None, "remaining"))

    # PTQ method 2: call quantize_model in eval_shape and quantize_params.
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

  @parameterized.parameters(False, True)
  def test_nnx_srq(self, act_asymmetric):
    q_rules = [
        qconfig.QuantizationRule(
            module_path=".*",
            weight_qtype=jnp.int8,
            act_qtype=jnp.int8,
            act_static_scale=True,
            act_asymmetric=act_asymmetric,
        ),
    ]

    # QAT to generate quant_stats.
    model_input = jnp.ones((10, 12))
    qat_linear = qwix_model.quantize_model(
        nnx.Linear(in_features=12, out_features=5, rngs=nnx.Rngs(0)),
        qat.QatProvider(q_rules),
        model_input,
    )
    qat_linear(model_input)
    quant_stats = nnx.state(qat_linear, flax_util.QuantStat)
    quant_stat = quant_stats["dot_general0_lhs"].value

    self.assertEqual(quant_stat["count"].shape, ())
    self.assertEqual(quant_stat["count"], 1)
    if act_asymmetric:
      self.assertEqual(quant_stat["sum_of_min"].shape, (1, 1))
      self.assertEqual(quant_stat["sum_of_max"].shape, (1, 1))
    else:
      self.assertEqual(quant_stat["sum_of_absmax"].shape, (1, 1))

    # PTQ method 1: quantize_model also converts params and quant_stats.
    ptq_linear = qwix_model.quantize_model(
        qat_linear, ptq.PtqProvider(q_rules), model_input
    )
    self.assertIsInstance(ptq_linear.dot_general0_lhs_scale, ptq.WithAux)
    self.assertEqual(ptq_linear.dot_general0_lhs_scale.array.shape, (1, 1))
    if act_asymmetric:
      self.assertEqual(ptq_linear.dot_general0_lhs_zero_point.shape, (1, 1))

    # PTQ method 2: manually call quantize_params.
    abs_ptq_linear = nnx.eval_shape(
        lambda: qwix_model.quantize_model(
            qat_linear, ptq.PtqProvider(q_rules), model_input
        )
    )
    qat_params = nnx.state(qat_linear, nnx.Param)
    quantized_params = ptq.quantize_params(
        nnx.to_pure_dict(qat_params),
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


if __name__ == "__main__":
  absltest.main()
