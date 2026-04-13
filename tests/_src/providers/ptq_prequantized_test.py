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
import os
from typing import Any

from absl.testing import absltest
from absl.testing import parameterized
from flax import linen as nn
from flax import nnx
import jax
from jax import numpy as jnp
import numpy as np
from qwix._src import model as qwix_model
from qwix._src import qconfig
from qwix._src.providers import ptq

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
    if set(tree) == {"array"} and isinstance(tree["array"], dict):
      array_payload = tree["array"]
      if set(array_payload) <= {"qvalue", "scale", "zero_point"}:
        return {
            key: _to_orbax_payload(value)
            for key, value in array_payload.items()
        }
    return {key: _to_orbax_payload(value) for key, value in tree.items()}
  return (
      np.asarray(jax.device_get(tree)) if isinstance(tree, jax.Array) else tree
  )


def _build_linear_reference(
    q_rules: list[qconfig.QuantizationRule], model_input: jax.Array
) -> tuple[nnx.Module, dict[str, Any], dict[str, Any]]:
  fp_linear = nnx.Linear(
      in_features=model_input.shape[-1],
      out_features=6,
      rngs=nnx.Rngs(0),
  )
  abs_ptq_linear = nnx.eval_shape(
      lambda: qwix_model.quantize_model(
          fp_linear,
          ptq.PtqProvider(q_rules),
          model_input,
      )
  )
  orig_params = nnx.to_pure_dict(nnx.state(fp_linear, nnx.Param))
  reference_params = ptq.quantize_params(orig_params, abs_ptq_linear)
  return abs_ptq_linear, orig_params, reference_params


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
    processed_params = ptq.process_prequantized_params(
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
      ptq.process_prequantized_params(full_precision_payload, abs_ptq_linear)

  def test_process_prequantized_params_fp_template(self):
    q_rules = [
        qconfig.QuantizationRule(
            module_path=".*", weight_qtype=jnp.int8, tile_size=4
        ),
    ]
    model_input = jnp.ones((10, 12))
    fp_linear = nnx.Linear(12, 6, rngs=nnx.Rngs(0))

    _, _, reference_params = _build_linear_reference(q_rules, model_input)
    orbax_payload = _to_orbax_payload(reference_params)

    processed_params = ptq.process_prequantized_params(orbax_payload, fp_linear)

    self.assertIsInstance(processed_params["kernel"], dict)
    self.assertIn("qvalue", processed_params["kernel"])
    self.assertIn("scale", processed_params["kernel"])
    self.assertNotIsInstance(processed_params["bias"], dict)

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
      processed_params = ptq.process_prequantized_params(
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
    processed_params = ptq.process_prequantized_params(
        orbax_payload, abs_ptq_linear
    )

    _assert_trees_allclose(self, processed_params, reference_params)

  @parameterized.named_parameters(
      dict(
          testcase_name="missing_scale",
          modify_payload_fn=lambda p: p["kernel"].pop("scale"),
          expected_regex="scale",
      ),
      dict(
          testcase_name="wrong_shape",
          modify_payload_fn=lambda p: p["kernel"].update(
              {"qvalue": p["kernel"]["qvalue"][:-1]}
          ),
          expected_regex="shape",
      ),
      dict(
          testcase_name="rejects_unexpected_zero_point",
          modify_payload_fn=lambda p: p["kernel"].update({
              "zero_point": np.zeros_like(
                  p["kernel"]["scale"], dtype=p["kernel"]["qvalue"].dtype
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
      ptq.process_prequantized_params(orbax_payload, abs_ptq_linear)

  def test_process_prequantized_params_allow_extra_params(self):
    q_rules = [qconfig.QuantizationRule(weight_qtype=jnp.int8)]
    model_input = jnp.ones((10, 12))
    abs_ptq_linear, _, reference_params = _build_linear_reference(
        q_rules, model_input
    )
    orbax_payload = _to_orbax_payload(reference_params)
    orbax_payload["extra"] = np.ones((1,), dtype=np.float32)

    with self.assertRaisesRegex(ValueError, "extra"):
      ptq.process_prequantized_params(
          orbax_payload,
          abs_ptq_linear,
          allow_extra_params=False,
      )

    processed_params = ptq.process_prequantized_params(
        orbax_payload,
        abs_ptq_linear,
        allow_extra_params=True,
    )
    self.assertNotIn("extra", processed_params)

  def test_process_prequantized_params_empty(self):
    q_rules = [qconfig.QuantizationRule(weight_qtype=jnp.int8)]
    model_input = jnp.ones((10, 12))
    abs_ptq_linear, _, _ = _build_linear_reference(q_rules, model_input)

    processed_params = ptq.process_prequantized_params({}, abs_ptq_linear)
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

    processed_params = ptq.process_prequantized_params(
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

    processed_params = ptq.process_prequantized_params(
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

    with self.assertRaisesRegex(TypeError, "NNX PTQ models"):
      ptq.process_prequantized_params({}, abs_ptq_params)

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

    processed_params = ptq.process_prequantized_params(
        orbax_payload, abs_ptq_model
    )

    _assert_trees_allclose(self, processed_params, reference_params)
    nnx.update(abs_ptq_model, processed_params)


if __name__ == "__main__":
  absltest.main()
