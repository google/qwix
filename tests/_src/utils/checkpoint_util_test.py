# Copyright 2026 Google LLC
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
from typing import Any
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from flax import linen as nn
from flax import nnx
import jax
from jax import numpy as jnp
import numpy as np
from qwix._src import model as qwix_model
from qwix._src import qconfig
from qwix._src.core import qarray
from qwix._src.providers import ptq
from qwix._src.providers import qt
from qwix._src.utils import checkpoint_util

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"


def _assert_trees_allclose(
    testcase: absltest.TestCase, lhs: Any, rhs: Any
) -> None:
  jax.tree.map_with_path(
      lambda kp, x, y: testcase.assertTrue(jnp.allclose(x, y), f"{kp} {x} {y}"),
      lhs,
      rhs,
  )


def _get_canonical_named_sharding(x: jax.Array) -> jax.sharding.Sharding:
  sharding = x.sharding
  if not isinstance(sharding, jax.sharding.NamedSharding):
    return sharding
  padded_pspec = sharding.spec + (None,) * (x.ndim - len(sharding.spec))
  return sharding.update(spec=padded_pspec)


def _to_orbax_payload(tree: Any) -> Any:
  if isinstance(tree, dict):
    return {key: _to_orbax_payload(value) for key, value in tree.items()}
  if isinstance(tree, (list, tuple)):
    return type(tree)(_to_orbax_payload(x) for x in tree)
  return (
      np.asarray(jax.device_get(tree)) if isinstance(tree, jax.Array) else tree
  )


def _build_linear_reference(
    q_rules: list[qconfig.QuantizationRule],
    model_input: jax.Array,
    provider_class: type[qconfig.QuantizationProvider] = ptq.PtqProvider,
) -> tuple[nnx.Module, dict[str, Any], dict[str, Any]]:
  fp_linear = nnx.Linear(
      in_features=model_input.shape[-1],
      out_features=6,
      rngs=nnx.Rngs(0),
  )
  abs_quantized_linear = nnx.eval_shape(
      lambda: qwix_model.quantize_model(
          fp_linear,
          provider_class(q_rules),
          model_input,
      )
  )
  orig_params = nnx.to_pure_dict(nnx.state(fp_linear, nnx.Param))
  if provider_class is ptq.PtqProvider:
    reference_params = ptq.quantize_params(orig_params, abs_quantized_linear)
  else:
    reference_params = orig_params
  return abs_quantized_linear, orig_params, reference_params


class PrequantizedPtqTest(parameterized.TestCase):

  def test_process_prequantized_params_nnx(self):
    q_rules = [
        qconfig.QuantizationRule(
            module_path=".*", weight_qtype=jnp.int8, tile_size=4
        ),
    ]
    model_input = jnp.ones((10, 12))
    abs_ptq_linear, orig_params, reference_params = _build_linear_reference(
        q_rules, model_input
    )

    orbax_payload = _to_orbax_payload(reference_params)
    processed_params = checkpoint_util.process_prequantized_params(
        orbax_payload, abs_ptq_linear
    )

    self.assertTrue(jnp.allclose(processed_params["bias"], orig_params["bias"]))
    _assert_trees_allclose(self, processed_params, reference_params)

    nnx.update(abs_ptq_linear, processed_params)
    abs_ptq_linear(model_input)

  def test_process_prequantized_params_full_precision_override(self):
    q_rules = [
        qconfig.QuantizationRule(
            module_path=".*", weight_qtype=jnp.int8, tile_size=4
        ),
    ]
    model_input = jnp.ones((10, 12))
    abs_ptq_linear, orig_params, _ = _build_linear_reference(
        q_rules, model_input
    )

    # Simulate safetensors providing a pure full-precision array for a
    # quantized block
    full_precision_payload = {"kernel": orig_params["kernel"]}
    with self.assertRaisesRegex(
        ValueError, "Unhandled or invalid parameter combination"
    ):
      checkpoint_util.process_prequantized_params(
          full_precision_payload, abs_ptq_linear
      )

  def test_process_prequantized_params_nnx_einsum_sharding(self):
    mesh = jax.make_mesh(
        (2, 2),
        ("fsdp", "tp"),
        axis_types=(jax.sharding.AxisType.Auto,) * len(("fsdp", "tp")),
    )
    q_rules = [
        qconfig.QuantizationRule(
            module_path=".*", weight_qtype=jnp.int8, tile_size=4
        ),
    ]
    model_input = jnp.ones((10, 1, 16))
    with jax.set_mesh(mesh):
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

    unsharded_state = nnx.state(fp_einsum)
    sharding = nnx.get_named_sharding(unsharded_state, mesh)
    sharded_state = jax.device_put(unsharded_state, sharding)
    nnx.update(fp_einsum, sharded_state)

    with jax.set_mesh(mesh):
      abs_ptq_einsum = nnx.eval_shape(
          lambda: qwix_model.quantize_model(
              fp_einsum,
              ptq.PtqProvider(q_rules),
              model_input,
          ),
      )

    orig_params = nnx.to_pure_dict(nnx.state(fp_einsum, nnx.Param))
    reference_params = ptq.quantize_params(orig_params, abs_ptq_einsum)
    orbax_payload = _to_orbax_payload(reference_params)

    with jax.set_mesh(mesh):
      processed_params = checkpoint_util.process_prequantized_params(
          orbax_payload, abs_ptq_einsum
      )
    _assert_trees_allclose(self, processed_params, reference_params)
    jax.tree.map_with_path(
        lambda kp, x, y: self.assertEqual(
            _get_canonical_named_sharding(x),
            _get_canonical_named_sharding(y),
            f"{kp} {x.sharding} {y.sharding}",
        ),
        processed_params,
        reference_params,
    )

    with jax.set_mesh(mesh):
      nnx.update(abs_ptq_einsum, processed_params)
      abs_ptq_einsum(model_input)

  def test_process_prequantized_params_nnx_asymmetric(self):
    q_rules = [
        qconfig.QuantizationRule(
            module_path=".*",
            weight_qtype=jnp.int8,
            weight_calibration_method="minmax",
        ),
    ]
    model_input = jnp.ones((10, 12))
    abs_ptq_linear, _, reference_params = _build_linear_reference(
        q_rules, model_input
    )

    self.assertIn("zero_point", reference_params["kernel"]["array"])
    orbax_payload = _to_orbax_payload(reference_params)
    processed_params = checkpoint_util.process_prequantized_params(
        orbax_payload, abs_ptq_linear
    )

    _assert_trees_allclose(self, processed_params, reference_params)

  @parameterized.named_parameters(
      dict(
          testcase_name="missing_scale",
          modify_payload_fn=lambda p: p["kernel"]["array"].pop("scale"),
          expected_regex="scale",
      ),
      dict(
          testcase_name="wrong_shape",
          modify_payload_fn=lambda p: p["kernel"]["array"].update(
              {"qvalue": p["kernel"]["array"]["qvalue"][:-1]}
          ),
          expected_regex="shape",
      ),
      dict(
          testcase_name="rejects_unexpected_zero_point",
          modify_payload_fn=lambda p: p["kernel"]["array"].update({
              "zero_point": np.zeros_like(
                  p["kernel"]["array"]["scale"],
                  dtype=p["kernel"]["array"]["qvalue"].dtype,
              )
          }),
          expected_regex="unexpected",
      ),
  )
  def test_process_prequantized_params_errors(
      self, modify_payload_fn, expected_regex
  ):
    q_rules = [qconfig.QuantizationRule(weight_qtype=jnp.int8)]
    model_input = jnp.ones((10, 12))
    abs_ptq_linear, _, reference_params = _build_linear_reference(
        q_rules, model_input
    )
    orbax_payload = _to_orbax_payload(reference_params)
    modify_payload_fn(orbax_payload)

    with self.assertRaisesRegex(ValueError, expected_regex):
      checkpoint_util.process_prequantized_params(orbax_payload, abs_ptq_linear)

  def test_process_prequantized_params_allow_extra_params(self):
    q_rules = [qconfig.QuantizationRule(weight_qtype=jnp.int8)]
    model_input = jnp.ones((10, 12))
    abs_ptq_linear, _, reference_params = _build_linear_reference(
        q_rules, model_input
    )
    orbax_payload = _to_orbax_payload(reference_params)
    orbax_payload["extra"] = np.ones((1,), dtype=np.float32)

    with self.assertRaisesRegex(ValueError, "extra"):
      checkpoint_util.process_prequantized_params(
          orbax_payload,
          abs_ptq_linear,
          allow_extra_params=False,
      )

    processed_params = checkpoint_util.process_prequantized_params(
        orbax_payload,
        abs_ptq_linear,
        allow_extra_params=True,
    )
    self.assertNotIn("extra", processed_params)

  def test_process_prequantized_params_empty(self):
    q_rules = [qconfig.QuantizationRule(weight_qtype=jnp.int8)]
    model_input = jnp.ones((10, 12))
    abs_ptq_linear, _, _ = _build_linear_reference(q_rules, model_input)

    processed_params = checkpoint_util.process_prequantized_params(
        {}, abs_ptq_linear
    )
    self.assertEqual(processed_params, {})

  def test_process_prequantized_params_nested_modules(self):
    class MLP(nnx.Module):

      def __init__(self, rngs):
        self.dense1 = nnx.Linear(12, 12, rngs=rngs)
        self.dense2 = nnx.Linear(12, 6, rngs=rngs)

      def __call__(self, x):
        return self.dense2(self.dense1(x))

    rngs = nnx.Rngs(0)
    mlp = MLP(rngs)
    model_input = jnp.ones((10, 12))

    q_rules = [qconfig.QuantizationRule(weight_qtype=jnp.int8)]
    abs_ptq_mlp = nnx.eval_shape(
        lambda: qwix_model.quantize_model(
            mlp,
            ptq.PtqProvider(q_rules),
            model_input,
        )
    )

    orig_params = nnx.to_pure_dict(nnx.state(mlp, nnx.Param))
    reference_params = ptq.quantize_params(orig_params, abs_ptq_mlp)
    orbax_payload = _to_orbax_payload(reference_params)

    processed_params = checkpoint_util.process_prequantized_params(
        orbax_payload, abs_ptq_mlp
    )

    _assert_trees_allclose(self, processed_params, reference_params)

  def test_process_prequantized_params_unquantized_ignored_if_missing(self):
    q_rules = [qconfig.QuantizationRule(weight_qtype=jnp.int8)]
    model_input = jnp.ones((10, 12))
    abs_ptq_linear, _, reference_params = _build_linear_reference(
        q_rules, model_input
    )

    orbax_payload = _to_orbax_payload(reference_params)
    self.assertIn("bias", orbax_payload)
    del orbax_payload["bias"]

    processed_params = checkpoint_util.process_prequantized_params(
        orbax_payload, abs_ptq_linear
    )

    self.assertIn("kernel", processed_params)
    self.assertNotIn("bias", processed_params)

  def test_process_prequantized_params_rejects_linen_template(self):
    dense = nn.Dense(features=5)
    q_rules = [qconfig.QuantizationRule(weight_qtype=jnp.int8)]
    ptq_dense = qwix_model.quantize_model(dense, ptq.PtqProvider(q_rules))
    abs_ptq_params = jax.eval_shape(
        ptq_dense.init, jax.random.key(0), jnp.ones((10, 12))
    )["params"]

    with self.assertRaisesRegex(TypeError, "NNX PTQ/QT models"):
      checkpoint_util.process_prequantized_params({}, abs_ptq_params)

  def test_process_prequantized_params_with_lists(self):
    class ListModel(nnx.Module):

      def __init__(self, rngs):
        self.layers = nnx.List([
            nnx.Linear(12, 12, rngs=rngs),
            nnx.Linear(12, 6, rngs=rngs),
        ])

      def __call__(self, x):
        for layer in self.layers:
          x = layer(x)
        return x

    rngs = nnx.Rngs(0)
    model = ListModel(rngs)
    model_input = jnp.ones((10, 12))

    q_rules = [qconfig.QuantizationRule(weight_qtype=jnp.int8)]
    abs_ptq_model = nnx.eval_shape(
        lambda: qwix_model.quantize_model(
            model,
            ptq.PtqProvider(q_rules),
            model_input,
        )
    )

    orig_params = nnx.to_pure_dict(nnx.state(model, nnx.Param))
    reference_params = ptq.quantize_params(orig_params, abs_ptq_model)
    orbax_payload = _to_orbax_payload(reference_params)

    processed_params = checkpoint_util.process_prequantized_params(
        orbax_payload, abs_ptq_model
    )

    _assert_trees_allclose(self, processed_params, reference_params)
    nnx.update(abs_ptq_model, processed_params)

  def test_process_prequantized_params_manual_dict_template(self):
    template_model = nnx.Linear(12, 6, rngs=nnx.Rngs(0))
    # Override the non-intercepted but pre-quantized kernel with the manual
    # quantized dictionary structure.
    template_model.kernel = {
        "array": {
            "qvalue": nnx.Param(jax.ShapeDtypeStruct((12, 6), jnp.int8)),
            "scale": nnx.Param(jax.ShapeDtypeStruct((1, 6), jnp.float32)),
        }
    }

    reference_params = {
        "kernel": {
            "array": {
                "qvalue": jnp.ones((12, 6), dtype=jnp.int8),
                "scale": jnp.ones((1, 6), dtype=jnp.float32),
            }
        }
    }
    orbax_payload = _to_orbax_payload(reference_params)

    processed_params = checkpoint_util.process_prequantized_params(
        orbax_payload, template_model
    )

    _assert_trees_allclose(self, processed_params, reference_params)
    nnx.update(template_model, processed_params)


class PrequantizedQtTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name="symmetric",
          qt_rules=[
              qt.QtRule(
                  module_path=".*",
                  weight_qtype=jnp.int8,
                  act_qtype=jnp.int8,
                  bwd_qtype=jnp.int8,
                  tile_size=4,
              ),
          ],
          check_zero_point=False,
      ),
      dict(
          testcase_name="asymmetric",
          qt_rules=[
              qt.QtRule(
                  module_path=".*",
                  weight_qtype=jnp.int8,
                  weight_calibration_method="minmax",
                  act_qtype=jnp.int8,
                  bwd_qtype=jnp.int8,
              ),
          ],
          check_zero_point=True,
      ),
  )
  def test_process_prequantized_params_quantized(
      self, qt_rules, check_zero_point
  ):
    model_input = jnp.ones((10, 12))
    abs_qt_linear, orig_params, _ = _build_linear_reference(
        qt_rules, model_input, qt.QtProvider
    )
    _, _, ref_params_ptq = _build_linear_reference(qt_rules, model_input)
    self.assertEqual(
        check_zero_point, "zero_point" in ref_params_ptq["kernel"]["array"]
    )

    orbax_payload = _to_orbax_payload(ref_params_ptq)
    processed_params = checkpoint_util.process_prequantized_params(
        orbax_payload, abs_qt_linear
    )

    ref_kernel_dict = ref_params_ptq["kernel"]["array"]
    ref_qarray = qarray.QArray(
        qvalue=ref_kernel_dict["qvalue"],
        scale=ref_kernel_dict["scale"],
        zero_point=ref_kernel_dict.get("zero_point"),
    )
    expected_kernel = qarray.dequantize(ref_qarray)

    _assert_trees_allclose(self, processed_params["kernel"], expected_kernel)
    _assert_trees_allclose(self, processed_params["bias"], orig_params["bias"])
    nnx.update(abs_qt_linear, processed_params)
    abs_qt_linear(model_input)

  def test_process_prequantized_params_full_precision(self):
    qt_rules = [
        qt.QtRule(
            module_path=".*",
            weight_qtype=jnp.int8,
            act_qtype=jnp.int8,
            bwd_qtype=jnp.int8,
            tile_size=4,
        ),
    ]
    model_input = jnp.ones((10, 12))
    abs_qt_linear, orig_params, _ = _build_linear_reference(
        qt_rules, model_input, qt.QtProvider
    )

    orbax_payload = _to_orbax_payload(orig_params)
    processed_params = checkpoint_util.process_prequantized_params(
        orbax_payload, abs_qt_linear
    )

    _assert_trees_allclose(self, processed_params, orig_params)
    nnx.update(abs_qt_linear, processed_params)
    abs_qt_linear(model_input)

  @parameterized.named_parameters(
      dict(
          testcase_name="missing_scale",
          modify_payload_fn=lambda p: p["kernel"]["array"].pop("scale"),
          expected_regex="scale",
      ),
      dict(
          testcase_name="wrong_keys",
          modify_payload_fn=lambda p: p["kernel"]["array"].update({"extra": 1}),
          expected_regex="unsupported",
      ),
  )
  def test_process_prequantized_params_errors(
      self, modify_payload_fn, expected_regex
  ):
    qt_rules = [
        qt.QtRule(
            module_path=".*",
            weight_qtype=jnp.int8,
            act_qtype=jnp.int8,
            bwd_qtype=jnp.int8,
            tile_size=4,
        ),
    ]
    model_input = jnp.ones((10, 12))
    abs_qt_linear, _, _ = _build_linear_reference(
        qt_rules, model_input, qt.QtProvider
    )
    _, _, ref_params_ptq = _build_linear_reference(qt_rules, model_input)
    orbax_payload = _to_orbax_payload(ref_params_ptq)
    modify_payload_fn(orbax_payload)

    with self.assertRaisesRegex(ValueError, expected_regex):
      checkpoint_util.process_prequantized_params(orbax_payload, abs_qt_linear)

  def test_process_prequantized_params_param_with_array_attribute(self):
    class MyShapeDtypeStruct(jax.ShapeDtypeStruct):

      def __init__(self, shape, dtype, array_attr):
        super().__init__(shape, dtype)
        object.__setattr__(self, "array", array_attr)

    class MyModel(nnx.Module):

      def __init__(self):
        self.kernel = nnx.Param(
            MyShapeDtypeStruct(
                (12, 6),
                jnp.float32,
                jax.ShapeDtypeStruct((12, 6), jnp.float32),
            )
        )

    model = MyModel()
    checkpoint_payload = {
        "kernel": {
            "array": {
                "qvalue": jnp.ones((12, 6), dtype=jnp.int8),
                "scale": jnp.ones((1, 6), dtype=jnp.float32),
            }
        }
    }

    processed_params = checkpoint_util.process_prequantized_params(
        checkpoint_payload, model
    )

    self.assertIn("kernel", processed_params)
    self.assertIn("array", processed_params["kernel"])
    self.assertIsInstance(processed_params["kernel"]["array"], jax.Array)

  def test_multi_device_sharding(self):
    mesh = jax.make_mesh(
        (2, 2),
        ("fsdp", "tp"),
        axis_types=(jax.sharding.AxisType.Auto,) * len(("fsdp", "tp")),
    )
    qt_rules = [
        qt.QtRule(
            module_path=".*",
            weight_qtype=jnp.int8,
            act_qtype=jnp.int8,
            bwd_qtype=jnp.int8,
            tile_size=4,
        ),
    ]
    model_input = jnp.ones((10, 12))

    with jax.set_mesh(mesh):
      fp_linear = nnx.Linear(
          in_features=12,
          out_features=6,
          rngs=nnx.Rngs(0),
          kernel_init=nnx.with_partitioning(
              nnx.initializers.lecun_normal(), ("fsdp", "tp")
          ),
          bias_init=nnx.with_partitioning(nnx.initializers.zeros, ("tp",)),
      )

      unsharded_state = nnx.state(fp_linear)
      sharding = nnx.get_named_sharding(unsharded_state, mesh)
      sharded_state = jax.device_put(unsharded_state, sharding)
      nnx.update(fp_linear, sharded_state)

      abs_qt_linear = nnx.eval_shape(
          lambda: qwix_model.quantize_model(
              fp_linear,
              qt.QtProvider(qt_rules),
              model_input,
          )
      )

    dummy_payload = {
        "kernel": {
            "array": {
                "qvalue": np.ones((12, 6), dtype=np.int8),
                "scale": np.ones((3, 6), dtype=np.float32),
            }
        },
        "bias": np.zeros((6,), dtype=np.float32),
    }

    with jax.set_mesh(mesh):
      processed_params = checkpoint_util.process_prequantized_params(
          dummy_payload, abs_qt_linear
      )

    self.assertIn("kernel", processed_params)
    self.assertIn("bias", processed_params)

    kernel_array = processed_params["kernel"]
    bias_array = processed_params["bias"]

    self.assertIsInstance(kernel_array, jax.Array)
    self.assertIsInstance(bias_array, jax.Array)

    self.assertIsInstance(kernel_array.sharding, jax.sharding.NamedSharding)
    self.assertIsInstance(bias_array.sharding, jax.sharding.NamedSharding)

    self.assertEqual(
        kernel_array.sharding.spec, jax.sharding.PartitionSpec("fsdp", "tp")
    )
    self.assertEqual(bias_array.sharding.spec, jax.sharding.PartitionSpec("tp"))

    self.assertEqual(kernel_array.sharding.mesh, mesh)
    self.assertEqual(bias_array.sharding.mesh, mesh)

  def test_single_device_sharding(self):
    device = jax.devices()[1]
    sharding = jax.sharding.SingleDeviceSharding(device)
    qt_rules = [
        qt.QtRule(
            module_path=".*",
            weight_qtype=jnp.int8,
            act_qtype=jnp.int8,
            bwd_qtype=jnp.int8,
            tile_size=4,
        ),
    ]
    model_input = jnp.ones((10, 12))

    fp_linear = nnx.Linear(
        in_features=12,
        out_features=6,
        rngs=nnx.Rngs(0),
    )

    state = nnx.state(fp_linear)
    sharded_state = jax.tree.map(lambda x: jax.device_put(x, sharding), state)
    nnx.update(fp_linear, sharded_state)

    qt_linear = qwix_model.quantize_model(
        fp_linear,
        qt.QtProvider(qt_rules),
        model_input,
    )

    dummy_payload = {
        "kernel": {
            "array": {
                "qvalue": np.ones((12, 6), dtype=np.int8),
                "scale": np.ones((3, 6), dtype=np.float32),
            }
        },
        "bias": np.zeros((6,), dtype=np.float32),
    }

    processed_params = checkpoint_util.process_prequantized_params(
        dummy_payload, qt_linear
    )

    self.assertIn("kernel", processed_params)
    self.assertIn("bias", processed_params)

    kernel_array = processed_params["kernel"]
    bias_array = processed_params["bias"]

    self.assertIsInstance(
        kernel_array.sharding, jax.sharding.SingleDeviceSharding
    )
    self.assertIsInstance(
        bias_array.sharding, jax.sharding.SingleDeviceSharding
    )

    self.assertEqual(kernel_array.sharding, sharding)
    self.assertEqual(bias_array.sharding, sharding)

  def test_no_sharding(self):
    qt_rules = [
        qt.QtRule(
            module_path=".*",
            weight_qtype=jnp.int8,
            act_qtype=jnp.int8,
            bwd_qtype=jnp.int8,
            tile_size=4,
        ),
    ]
    model_input = jnp.ones((10, 12))

    fp_linear = nnx.Linear(
        in_features=12,
        out_features=6,
        rngs=nnx.Rngs(0),
    )

    abs_qt_linear = nnx.eval_shape(
        lambda: qwix_model.quantize_model(
            fp_linear,
            qt.QtProvider(qt_rules),
            model_input,
        )
    )

    dummy_payload = {
        "kernel": {
            "array": {
                "qvalue": np.ones((12, 6), dtype=np.int8),
                "scale": np.ones((3, 6), dtype=np.float32),
            }
        },
        "bias": np.zeros((6,), dtype=np.float32),
    }

    processed_params = checkpoint_util.process_prequantized_params(
        dummy_payload, abs_qt_linear
    )

    self.assertIn("kernel", processed_params)
    self.assertIn("bias", processed_params)

    kernel_array = processed_params["kernel"]
    bias_array = processed_params["bias"]

    self.assertEqual(list(kernel_array.devices()), [jax.devices()[0]])
    self.assertEqual(list(bias_array.devices()), [jax.devices()[0]])


class RestoreQuantizationRulesTest(parameterized.TestCase):

  def test_restore_rules_linear(self):
    checkpoint_params = {
        "kernel": {
            "qvalue": jnp.ones((4, 4), dtype=jnp.int8),
            "scale": jnp.ones((1, 4), dtype=jnp.float32),
        },
        "bias": jnp.zeros((4,), dtype=jnp.float32),
    }
    rules = checkpoint_util.restore_quantization_rules(
        checkpoint_params, qconfig.QuantizationRule, tile_size=4
    )
    expected_rules = [
        qconfig.QuantizationRule(
            module_path="",
            weight_qtype=jnp.int8,
            act_qtype=jnp.int8,
            weight_calibration_method="absmax",
            tile_size=4,
        )
    ]
    self.assertEqual(rules, expected_rules)

  def test_restore_rules_nested(self):
    checkpoint_params = {
        "dense1": {
            "kernel": {
                "qvalue": jnp.ones((4, 4), dtype=jnp.int8),
                "scale": jnp.ones((1, 4), dtype=jnp.float32),
            }
        },
        "dense2": {
            "kernel": {
                "qvalue": jnp.ones((4, 4), dtype=jnp.int16),
                "scale": jnp.ones((1, 4), dtype=jnp.float32),
            }
        },
    }
    rules = checkpoint_util.restore_quantization_rules(
        checkpoint_params, qconfig.QuantizationRule, tile_size=4
    )
    expected_rules = [
        qconfig.QuantizationRule(
            module_path="dense1",
            weight_qtype=jnp.int8,
            act_qtype=jnp.int8,
            weight_calibration_method="absmax",
            tile_size=4,
        ),
        qconfig.QuantizationRule(
            module_path="dense2",
            weight_qtype=jnp.int16,
            act_qtype=jnp.int16,
            weight_calibration_method="absmax",
            tile_size=4,
        ),
    ]
    rules = sorted(rules, key=lambda r: r.module_path)
    expected_rules = sorted(expected_rules, key=lambda r: r.module_path)
    self.assertEqual(rules, expected_rules)

  def test_restore_rules_wildcards_override(self):
    checkpoint_params = {
        "layers": {
            0: {
                "kernel": {
                    "qvalue": jnp.ones((4, 4), dtype=jnp.int4),
                    "scale": jnp.ones((1, 4), dtype=jnp.float32),
                }
            },
            1: {
                "kernel": {
                    "qvalue": jnp.ones((4, 4), dtype=jnp.int8),
                    "scale": jnp.ones((1, 4), dtype=jnp.float32),
                }
            },
        }
    }
    with mock.patch.object(checkpoint_util.logging, "warning") as mock_warning:
      rules = checkpoint_util.restore_quantization_rules(
          checkpoint_params, qconfig.QuantizationRule, tile_size=4
      )
    mock_warning.assert_called_once_with(
        "Conflicting quantization rules reconstructed for %s. Existing: %s,"
        " New: %s. The existing rule will be overwritten.",
        "layers/[^/]+",
        qconfig.QuantizationRule(
            module_path="layers/[^/]+",
            weight_qtype=jnp.int4,
            act_qtype=jnp.int4,
            weight_calibration_method="absmax",
            tile_size=4,
        ),
        qconfig.QuantizationRule(
            module_path="layers/[^/]+",
            weight_qtype=jnp.int8,
            act_qtype=jnp.int8,
            weight_calibration_method="absmax",
            tile_size=4,
        ),
    )
    expected_rules = [
        qconfig.QuantizationRule(
            module_path="layers/[^/]+",
            weight_qtype=jnp.int8,
            act_qtype=jnp.int8,
            weight_calibration_method="absmax",
            tile_size=4,
        )
    ]
    self.assertEqual(rules, expected_rules)

  def test_restore_rules_wildcards_no_override(self):
    checkpoint_params = {
        "layers": {
            0: {
                "kernel": {
                    "qvalue": jnp.ones((4, 4), dtype=jnp.int8),
                    "scale": jnp.ones((1, 4), dtype=jnp.float32),
                }
            },
            1: {
                "kernel": {
                    "qvalue": jnp.ones((4, 4), dtype=jnp.int8),
                    "scale": jnp.ones((1, 4), dtype=jnp.float32),
                }
            },
        }
    }
    with mock.patch.object(checkpoint_util.logging, "warning") as mock_warning:
      rules = checkpoint_util.restore_quantization_rules(
          checkpoint_params, qconfig.QuantizationRule, tile_size=4
      )
    mock_warning.assert_not_called()
    expected_rules = [
        qconfig.QuantizationRule(
            module_path="layers/[^/]+",
            weight_qtype=jnp.int8,
            act_qtype=jnp.int8,
            weight_calibration_method="absmax",
            tile_size=4,
        )
    ]
    self.assertEqual(rules, expected_rules)

  def test_restore_rules_no_act(self):
    checkpoint_params = {
        "kernel": {
            "qvalue": jnp.ones((4, 4), dtype=jnp.int8),
            "scale": jnp.ones((1, 4), dtype=jnp.float32),
        }
    }
    rules = checkpoint_util.restore_quantization_rules(
        checkpoint_params, qconfig.QuantizationRule, tile_size=4, act_qtype=None
    )
    expected_rules = [
        qconfig.QuantizationRule(
            module_path="",
            weight_qtype=jnp.int8,
            act_qtype=None,
            weight_calibration_method="absmax",
            tile_size=4,
        )
    ]
    self.assertEqual(rules, expected_rules)

  def test_restore_rules_custom_act(self):
    checkpoint_params = {
        "kernel": {
            "qvalue": jnp.ones((4, 4), dtype=jnp.int8),
            "scale": jnp.ones((1, 4), dtype=jnp.float32),
        }
    }
    rules = checkpoint_util.restore_quantization_rules(
        checkpoint_params,
        qconfig.QuantizationRule,
        tile_size=4,
        act_qtype=jnp.uint8,
    )
    expected_rules = [
        qconfig.QuantizationRule(
            module_path="",
            weight_qtype=jnp.int8,
            act_qtype=jnp.uint8,
            weight_calibration_method="absmax",
            tile_size=4,
        )
    ]
    self.assertEqual(rules, expected_rules)

  def test_restore_rules_asymmetric(self):
    checkpoint_params = {
        "kernel": {
            "qvalue": jnp.ones((4, 4), dtype=jnp.int8),
            "scale": jnp.ones((1, 4), dtype=jnp.float32),
            "zero_point": jnp.zeros((1, 4), dtype=jnp.int8),
        }
    }
    rules = checkpoint_util.restore_quantization_rules(
        checkpoint_params, qconfig.QuantizationRule, tile_size=4
    )
    expected_rules = [
        qconfig.QuantizationRule(
            module_path="",
            weight_qtype=jnp.int8,
            act_qtype=jnp.int8,
            weight_calibration_method="minmax",
            tile_size=4,
        )
    ]
    self.assertEqual(rules, expected_rules)

  def test_restore_rules_qt(self):
    checkpoint_params = {
        "kernel": {
            "qvalue": jnp.ones((4, 4), dtype=jnp.int8),
            "scale": jnp.ones((1, 4), dtype=jnp.float32),
        }
    }
    rules = checkpoint_util.restore_quantization_rules(
        checkpoint_params,
        qt.QtRule,
        tile_size=4,
        bwd_qtype=jnp.int8,
    )
    expected_rules = [
        qt.QtRule(
            module_path="",
            weight_qtype=jnp.int8,
            act_qtype=jnp.int8,
            weight_calibration_method="absmax",
            tile_size=4,
            bwd_qtype=jnp.int8,
        )
    ]
    self.assertEqual(rules, expected_rules)


if __name__ == "__main__":
  absltest.main()
