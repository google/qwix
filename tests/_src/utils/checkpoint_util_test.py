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
import math
import os
from typing import Any
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import flax
from flax import linen as nn
from flax import nnx
import jax
from jax import numpy as jnp
import numpy as np
from qwix._src import model as qwix_model
from qwix._src import qconfig
from qwix._src.core import qarray
from qwix._src.providers import boxed_param
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
  if isinstance(tree, boxed_param.WithAux):
    return {"array": _to_orbax_payload(tree.array)}
  if isinstance(tree, qarray.QArray):
    payload = {"qvalue": tree.qvalue, "scale": tree.scale}
    if tree.zero_point is not None:
      payload["zero_point"] = tree.zero_point
    return _to_orbax_payload(payload)
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


def _build_sharded_einsum_reference(
    q_rules: list[qconfig.QuantizationRule],
    model_input: jax.Array,
    mesh: jax.sharding.Mesh,
) -> tuple[nnx.Module, dict[str, Any], dict[str, Any]]:
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
  return abs_ptq_einsum, orig_params, reference_params


def _build_metadata_only_param(
    shape: tuple[int, ...],
    dtype: jax.typing.DTypeLike,
    out_sharding: tuple[str | None, ...],
) -> nnx.Param:
  # eager_sharding=False keeps the ShapeDtypeStruct unsharded instead of
  # resolving out_sharding against a mesh at construction time.
  return nnx.Param(
      jax.ShapeDtypeStruct(shape, dtype),
      eager_sharding=False,
      out_sharding=out_sharding,
  )


def _build_linen_reference(
    fp_model: nn.Module,
    q_rules: list[qconfig.QuantizationRule],
    model_input: jax.Array,
) -> tuple[nn.Module, dict[str, Any], dict[str, Any], dict[str, Any]]:
  quantized_model = qwix_model.quantize_model(
      fp_model, ptq.PtqProvider(q_rules)
  )
  abs_quantized_params = jax.eval_shape(
      quantized_model.init, jax.random.key(0), model_input
  )["params"]
  orig_params = nn.unbox(
      fp_model.init(jax.random.key(0), model_input)["params"]
  )
  reference_params = ptq.quantize_params(orig_params, abs_quantized_params)
  return quantized_model, abs_quantized_params, orig_params, reference_params


class _PartitionedLinenMlp(nn.Module):

  @nn.compact
  def __call__(self, x):
    x = nn.Dense(
        12,
        kernel_init=nn.with_partitioning(
            nn.initializers.lecun_normal(), ("fsdp", "tp")
        ),
        name="partitioned",
    )(x)
    return nn.Dense(
        6,
        kernel_init=nn.with_logical_partitioning(
            nn.initializers.lecun_normal(), ("embed", "mlp")
        ),
        name="logically_partitioned",
    )(x)


class _LinenBlock(nn.Module):

  @nn.compact
  def __call__(self, carry, _):
    return nn.Dense(16, name="dense")(carry), None


class _ScannedLinenModel(nn.Module):

  @nn.compact
  def __call__(self, x):
    blocks = nn.scan(
        _LinenBlock,
        variable_axes={"params": 0},
        split_rngs={"params": True},
        length=3,
    )
    x, _ = blocks(name="layers")(x, None)
    return x


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
    abs_ptq_einsum, _, reference_params = _build_sharded_einsum_reference(
        q_rules, model_input, mesh
    )
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
      dict(testcase_name="scalar", ckpt_scale_shape=()),
      dict(testcase_name="per_channel_rank_deficient", ckpt_scale_shape=(6,)),
      dict(testcase_name="per_channel", ckpt_scale_shape=(1, 6)),
  )
  def test_process_prequantized_params_coerced_not_requantized(
      self, ckpt_scale_shape
  ):
    q_rules = [
        qconfig.QuantizationRule(
            module_path=".*", weight_qtype=jnp.int8, tile_size=4
        ),
    ]
    model_input = jnp.ones((10, 12))
    abs_ptq_linear, _, reference_params = _build_linear_reference(
        q_rules, model_input
    )

    kernel_template = reference_params["kernel"]["array"]
    qvalue = kernel_template["qvalue"]
    self.assertEqual(kernel_template["scale"].shape, (3, 6))
    # Use distinct values per element to easily detect misaligned broadcasts.
    size = math.prod(ckpt_scale_shape)
    ckpt_scale = jnp.arange(1, size + 1, dtype=jnp.bfloat16).reshape(
        ckpt_scale_shape
    )
    orbax_payload = _to_orbax_payload({
        "kernel": {"array": {"qvalue": qvalue, "scale": ckpt_scale}},
        "bias": reference_params["bias"],
    })

    processed = checkpoint_util.process_prequantized_params(
        orbax_payload, abs_ptq_linear
    )
    processed_kernel = processed["kernel"]["array"]

    # The checkpoint scale is broadcastable up to the template scale, so it is
    # coerced not requantized.
    expected_scale = np.broadcast_to(
        np.reshape(
            jax.device_get(ckpt_scale),
            (1,) * (2 - len(ckpt_scale_shape)) + ckpt_scale_shape,
        ),
        kernel_template["scale"].shape,
    )
    np.testing.assert_array_equal(
        jax.device_get(processed_kernel["qvalue"]), jax.device_get(qvalue)
    )
    np.testing.assert_array_equal(
        jax.device_get(processed_kernel["scale"]), expected_scale
    )

  @parameterized.named_parameters(
      dict(testcase_name="scalar_scale", ckpt_scale_shape=()),
      dict(testcase_name="full_rank_scale", ckpt_scale_shape=(1, 1, 1)),
  )
  def test_process_prequantized_params_sharded_coerced_not_requantized(
      self, ckpt_scale_shape
  ):
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
    abs_ptq_einsum, _, reference_params = _build_sharded_einsum_reference(
        q_rules, model_input, mesh
    )

    kernel_template = reference_params["kernel"]["array"]
    qvalue = kernel_template["qvalue"]
    # The checkpoint scale is broadcastable up to the template scale, so it is
    # coerced not requantized.
    per_tensor_scale = jnp.full(ckpt_scale_shape, 0.5, dtype=jnp.bfloat16)
    orbax_payload = _to_orbax_payload({
        "kernel": {"array": {"qvalue": qvalue, "scale": per_tensor_scale}},
        "bias": reference_params["bias"],
    })

    with jax.set_mesh(mesh):
      processed_params = checkpoint_util.process_prequantized_params(
          orbax_payload, abs_ptq_einsum
      )
    processed_kernel = processed_params["kernel"]["array"]

    np.testing.assert_array_equal(
        jax.device_get(processed_kernel["qvalue"]), jax.device_get(qvalue)
    )
    np.testing.assert_array_equal(
        jax.device_get(processed_kernel["scale"]),
        np.full(kernel_template["scale"].shape, 0.5, dtype=jnp.bfloat16),
    )
    self.assertEqual(
        _get_canonical_named_sharding(processed_kernel["scale"]),
        _get_canonical_named_sharding(kernel_template["scale"]),
    )

    with jax.set_mesh(mesh):
      nnx.update(abs_ptq_einsum, processed_params)
      abs_ptq_einsum(model_input)

  def test_process_prequantized_params_finer_scale_is_requantized(self):
    q_rules = [
        qconfig.QuantizationRule(
            module_path=".*", weight_qtype=jnp.int8, tile_size=4
        ),
    ]
    model_input = jnp.ones((10, 12))
    abs_ptq_linear, orig_params, reference_params = _build_linear_reference(
        q_rules, model_input
    )

    kernel_template = reference_params["kernel"]["array"]
    qvalue = kernel_template["qvalue"]
    # A per-channel checkpoint scale is finer-grained than the tiled template
    # scale, so it cannot be broadcast up and must be requantized instead.
    per_channel_scale = jnp.full(qvalue.shape, 2, dtype=jnp.bfloat16)
    self.assertNotEqual(per_channel_scale.shape, kernel_template["scale"].shape)
    orbax_payload = _to_orbax_payload({
        "kernel": {"array": {"qvalue": qvalue, "scale": per_channel_scale}},
        "bias": reference_params["bias"],
    })

    processed_params = checkpoint_util.process_prequantized_params(
        orbax_payload, abs_ptq_linear
    )

    source_qarray = qarray.QArray(qvalue=qvalue, scale=per_channel_scale)
    dequantized_kernel = qarray.dequantize(source_qarray)
    expected_params = ptq.quantize_params(
        {"kernel": dequantized_kernel, "bias": orig_params["bias"]},
        abs_ptq_linear,
    )

    self.assertEqual(
        processed_params["kernel"]["array"]["scale"].shape,
        kernel_template["scale"].shape,
    )
    _assert_trees_allclose(self, processed_params, expected_params)
    nnx.update(abs_ptq_linear, processed_params)
    abs_ptq_linear(model_input)

  @parameterized.named_parameters(
      dict(
          testcase_name="symmetric",
          source_qtype=jnp.float8_e4m3fn,
          target_qtype=jnp.int4,
          calibration_method="absmax",
          check_zero_point=False,
      ),
      dict(
          testcase_name="asymmetric",
          source_qtype=jnp.int8,
          target_qtype=jnp.int4,
          calibration_method="minmax",
          check_zero_point=True,
      ),
  )
  def test_process_prequantized_params_requantized(
      self, source_qtype, target_qtype, calibration_method, check_zero_point
  ):
    model_input = jnp.ones((10, 12))
    source_rules = [
        qconfig.QuantizationRule(
            module_path=".*", weight_qtype=source_qtype, tile_size=4
        )
    ]
    _, orig_params, source_ref = _build_linear_reference(
        source_rules, model_input
    )
    target_rules = [
        qconfig.QuantizationRule(
            module_path=".*",
            weight_qtype=target_qtype,
            weight_calibration_method=calibration_method,
            tile_size=4,
        )
    ]
    abs_target_linear, _, _ = _build_linear_reference(target_rules, model_input)

    orbax_payload = _to_orbax_payload(source_ref)
    processed_params = checkpoint_util.process_prequantized_params(
        orbax_payload, abs_target_linear
    )
    source_kernel = source_ref["kernel"]["array"]
    source_qarray = qarray.QArray(
        qvalue=source_kernel["qvalue"],
        scale=source_kernel["scale"],
        zero_point=source_kernel.get("zero_point"),
    )
    dequantized_kernel = qarray.dequantize(source_qarray)
    expected_params = ptq.quantize_params(
        {"kernel": dequantized_kernel, "bias": orig_params["bias"]},
        abs_target_linear,
    )

    self.assertEqual(
        check_zero_point, "zero_point" in processed_params["kernel"]["array"]
    )
    self.assertEqual(
        processed_params["kernel"]["array"]["qvalue"].dtype, target_qtype
    )
    _assert_trees_allclose(self, processed_params, expected_params)
    nnx.update(abs_target_linear, processed_params)
    abs_target_linear(model_input)

  @parameterized.named_parameters(
      dict(
          testcase_name="asymmetric_to_symmetric",
          source_calibration_method="minmax",
          target_calibration_method="absmax",
          expect_zero_point=False,
      ),
      dict(
          testcase_name="symmetric_to_asymmetric",
          source_calibration_method="absmax",
          target_calibration_method="minmax",
          expect_zero_point=True,
      ),
  )
  def test_process_prequantized_params_requantized_calibration_change(
      self,
      source_calibration_method,
      target_calibration_method,
      expect_zero_point,
  ):
    model_input = jnp.ones((10, 12))
    source_rules = [
        qconfig.QuantizationRule(
            module_path=".*",
            weight_qtype=jnp.int8,
            weight_calibration_method=source_calibration_method,
            tile_size=4,
        )
    ]
    _, orig_params, source_ref = _build_linear_reference(
        source_rules, model_input
    )
    target_rules = [
        qconfig.QuantizationRule(
            module_path=".*",
            weight_qtype=jnp.int8,
            weight_calibration_method=target_calibration_method,
            tile_size=4,
        )
    ]
    abs_target_linear, _, _ = _build_linear_reference(target_rules, model_input)

    orbax_payload = _to_orbax_payload(source_ref)
    processed_params = checkpoint_util.process_prequantized_params(
        orbax_payload, abs_target_linear
    )

    source_kernel = source_ref["kernel"]["array"]
    source_qarray = qarray.QArray(
        qvalue=source_kernel["qvalue"],
        scale=source_kernel["scale"],
        zero_point=source_kernel.get("zero_point"),
    )
    dequantized_kernel = qarray.dequantize(source_qarray)
    expected_params = ptq.quantize_params(
        {"kernel": dequantized_kernel, "bias": orig_params["bias"]},
        abs_target_linear,
    )

    self.assertEqual(
        expect_zero_point, "zero_point" in processed_params["kernel"]["array"]
    )
    self.assertEqual(
        processed_params["kernel"]["array"]["qvalue"].dtype, jnp.int8
    )
    _assert_trees_allclose(self, processed_params, expected_params)
    nnx.update(abs_target_linear, processed_params)
    abs_target_linear(model_input)

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

  def test_process_prequantized_params_rejects_nnx_state_template(self):
    q_rules = [qconfig.QuantizationRule(weight_qtype=jnp.int8)]
    model_input = jnp.ones((10, 12))
    abs_ptq_linear, _, _ = _build_linear_reference(q_rules, model_input)

    with self.assertRaisesRegex(TypeError, "NNX model or a Linen params dict"):
      checkpoint_util.process_prequantized_params({}, nnx.state(abs_ptq_linear))

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

  def test_process_prequantized_params_requantized_sharding_from_metadata(self):
    mesh = jax.make_mesh(
        (2, 2),
        ("fsdp", "tp"),
        axis_types=(jax.sharding.AxisType.Auto,) * len(("fsdp", "tp")),
    )
    template_model = nnx.Linear(12, 6, rngs=nnx.Rngs(0))
    template_model.kernel = {
        "array": {
            "qvalue": _build_metadata_only_param(
                (12, 6), jnp.int4, ("fsdp", "tp")
            ),
            "scale": _build_metadata_only_param(
                (3, 6), jnp.float32, (None, "tp")
            ),
        }
    }
    dummy_payload = {
        "kernel": {
            "array": {
                "qvalue": np.ones((12, 6), dtype=np.int8),
                "scale": np.ones((3, 6), dtype=np.float32),
            }
        }
    }

    with (
        jax.set_mesh(mesh),
        mock.patch.object(
            qarray, "dequantize", wraps=qarray.dequantize
        ) as mock_dequantize,
    ):
      processed_params = checkpoint_util.process_prequantized_params(
          dummy_payload, template_model
      )

    # The checkpoint qvalue is placed onto the sharding recovered from the
    # template metadata before it is dequantized and requantized.
    mock_dequantize.assert_called_once()
    self.assertEqual(
        mock_dequantize.call_args.args[0].qvalue.sharding,
        jax.sharding.NamedSharding(
            mesh, jax.sharding.PartitionSpec("fsdp", "tp")
        ),
    )
    processed_kernel = processed_params["kernel"]["array"]
    self.assertEqual(processed_kernel["qvalue"].dtype, jnp.int4)
    self.assertEqual(
        processed_kernel["scale"].sharding,
        jax.sharding.NamedSharding(
            mesh, jax.sharding.PartitionSpec(None, "tp")
        ),
    )


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

  @parameterized.named_parameters(
      dict(testcase_name="symmetric", calibration_method="absmax"),
      dict(testcase_name="asymmetric", calibration_method="minmax"),
  )
  def test_process_prequantized_params_dequantized(self, calibration_method):
    qt_rules = [
        qt.QtRule(
            module_path=".*",
            weight_qtype=jnp.int8,
            weight_calibration_method=calibration_method,
            act_qtype=jnp.int8,
            bwd_qtype=jnp.int8,
        ),
    ]
    model_input = jnp.ones((10, 12), dtype=jnp.bfloat16)
    fp_linear = nnx.Linear(
        in_features=12,
        out_features=6,
        param_dtype=jnp.bfloat16,
        rngs=nnx.Rngs(0),
    )

    abs_quantized_linear = nnx.eval_shape(
        lambda: qwix_model.quantize_model(
            fp_linear,
            ptq.PtqProvider(qt_rules),
            model_input,
        )
    )
    orig_params = nnx.to_pure_dict(nnx.state(fp_linear, nnx.Param))
    ref_params_ptq = ptq.quantize_params(orig_params, abs_quantized_linear)
    orbax_payload = _to_orbax_payload(ref_params_ptq)
    processed_params = checkpoint_util.process_prequantized_params(
        orbax_payload, fp_linear
    )

    ref_kernel_dict = ref_params_ptq["kernel"]["array"]
    ref_qarray = qarray.QArray(
        qvalue=ref_kernel_dict["qvalue"],
        scale=ref_kernel_dict["scale"],
        zero_point=ref_kernel_dict.get("zero_point"),
    )
    expected_kernel = qarray.dequantize(ref_qarray)

    self.assertEqual(processed_params["kernel"].dtype, jnp.bfloat16)
    self.assertEqual(processed_params["bias"].dtype, jnp.bfloat16)
    _assert_trees_allclose(self, processed_params["kernel"], expected_kernel)
    _assert_trees_allclose(self, processed_params["bias"], orig_params["bias"])
    nnx.update(fp_linear, processed_params)
    fp_linear(model_input)

  @parameterized.named_parameters(
      dict(
          testcase_name="symmetric",
          source_qtype=jnp.float8_e4m3fn,
          target_qtype=jnp.int4,
          calibration_method="absmax",
      ),
      dict(
          testcase_name="asymmetric",
          source_qtype=jnp.int8,
          target_qtype=jnp.int4,
          calibration_method="minmax",
      ),
  )
  def test_process_prequantized_params_requantized(
      self, source_qtype, target_qtype, calibration_method
  ):
    model_input = jnp.ones((10, 12))
    source_rules = [
        qconfig.QuantizationRule(
            module_path=".*", weight_qtype=source_qtype, tile_size=4
        )
    ]
    _, orig_params, source_ref = _build_linear_reference(
        source_rules, model_input
    )
    target_rules = [
        qt.QtRule(
            module_path=".*",
            weight_qtype=target_qtype,
            weight_calibration_method=calibration_method,
            act_qtype=target_qtype,
            bwd_qtype=target_qtype,
            tile_size=4,
        )
    ]
    abs_target_linear, _, _ = _build_linear_reference(
        target_rules, model_input, qt.QtProvider
    )

    orbax_payload = _to_orbax_payload(source_ref)
    processed_params = checkpoint_util.process_prequantized_params(
        orbax_payload, abs_target_linear
    )
    source_kernel = source_ref["kernel"]["array"]
    source_qarray = qarray.QArray(
        qvalue=source_kernel["qvalue"],
        scale=source_kernel["scale"],
        zero_point=source_kernel.get("zero_point"),
    )
    expected_kernel = qarray.dequantize(source_qarray)

    _assert_trees_allclose(self, processed_params["kernel"], expected_kernel)
    _assert_trees_allclose(self, processed_params["bias"], orig_params["bias"])
    nnx.update(abs_target_linear, processed_params)
    abs_target_linear(model_input)

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

  def test_multi_device_sharding_from_metadata(self):
    mesh = jax.make_mesh(
        (2, 2),
        ("fsdp", "tp"),
        axis_types=(jax.sharding.AxisType.Auto,) * len(("fsdp", "tp")),
    )
    template_model = nnx.Linear(12, 6, rngs=nnx.Rngs(0))
    template_model.kernel = _build_metadata_only_param(
        (12, 6), jnp.float32, ("fsdp", "tp")
    )
    template_model.bias = _build_metadata_only_param((6,), jnp.float32, ("tp",))
    dummy_payload = {
        "kernel": {
            "array": {
                "qvalue": np.ones((12, 6), dtype=np.int8),
                "scale": np.ones((3, 6), dtype=np.float32),
            }
        },
        "bias": np.zeros((6,), dtype=np.float32),
    }

    with (
        jax.set_mesh(mesh),
        mock.patch.object(
            qarray, "dequantize", wraps=qarray.dequantize
        ) as mock_dequantize,
    ):
      processed_params = checkpoint_util.process_prequantized_params(
          dummy_payload, template_model
      )

    kernel_sharding = jax.sharding.NamedSharding(
        mesh, jax.sharding.PartitionSpec("fsdp", "tp")
    )
    # The checkpoint qvalue is placed onto the sharding recovered from the
    # template metadata before it is dequantized, rather than replicated.
    mock_dequantize.assert_called_once()
    self.assertEqual(
        mock_dequantize.call_args.args[0].qvalue.sharding, kernel_sharding
    )
    self.assertEqual(processed_params["kernel"].sharding, kernel_sharding)
    self.assertEqual(
        processed_params["bias"].sharding,
        jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("tp")),
    )
    np.testing.assert_array_equal(
        jax.device_get(processed_params["kernel"]),
        np.ones((12, 6), dtype=np.float32),
    )

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


class PrequantizedLinenTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(testcase_name="symmetric", calibration_method="absmax"),
      dict(testcase_name="asymmetric", calibration_method="minmax"),
  )
  def test_process_prequantized_params_linen(self, calibration_method):
    q_rules = [
        qconfig.QuantizationRule(
            module_path=".*",
            weight_qtype=jnp.int8,
            weight_calibration_method=calibration_method,
            tile_size=4,
        ),
    ]
    model_input = jnp.ones((10, 12))
    ptq_model, abs_ptq_params, _, reference_params = _build_linen_reference(
        nn.Dense(features=6), q_rules, model_input
    )

    orbax_payload = _to_orbax_payload(reference_params)
    processed_params = checkpoint_util.process_prequantized_params(
        orbax_payload, abs_ptq_params
    )

    # Linen consumes params as given, so the processed kernel keeps the
    # template's WithAux and QArray containers.
    self.assertIsInstance(processed_params["kernel"], boxed_param.WithAux)
    _assert_trees_allclose(self, processed_params, reference_params)
    ptq_model.apply({"params": processed_params}, model_input)

  def test_process_prequantized_params_linen_dequantized(self):
    q_rules = [
        qconfig.QuantizationRule(
            module_path=".*", weight_qtype=jnp.int8, tile_size=4
        ),
    ]
    model_input = jnp.ones((10, 12), dtype=jnp.bfloat16)
    fp_dense = nn.Dense(features=6, param_dtype=jnp.bfloat16)
    _, _, orig_params, ref_params_ptq = _build_linen_reference(
        fp_dense, q_rules, model_input
    )
    abs_fp_params = jax.eval_shape(
        fp_dense.init, jax.random.key(0), model_input
    )["params"]

    orbax_payload = _to_orbax_payload(ref_params_ptq)
    processed_params = checkpoint_util.process_prequantized_params(
        orbax_payload, abs_fp_params
    )

    expected_kernel = qarray.dequantize(ref_params_ptq["kernel"].array)
    self.assertEqual(processed_params["kernel"].dtype, jnp.bfloat16)
    self.assertEqual(processed_params["bias"].dtype, jnp.bfloat16)
    _assert_trees_allclose(self, processed_params["kernel"], expected_kernel)
    _assert_trees_allclose(self, processed_params["bias"], orig_params["bias"])
    fp_dense.apply({"params": processed_params}, model_input)

  def test_process_prequantized_params_linen_partitioned(self):
    q_rules = [
        qconfig.QuantizationRule(
            module_path=".*", weight_qtype=jnp.int8, tile_size=4
        ),
    ]
    model_input = jnp.ones((10, 12))
    ptq_mlp, abs_ptq_params, _, reference_params = _build_linen_reference(
        _PartitionedLinenMlp(), q_rules, model_input
    )
    # The template boxes each QArray field with its partitioning metadata.
    self.assertIsInstance(
        abs_ptq_params["partitioned"]["kernel"].array.qvalue, nn.Partitioned
    )
    self.assertIsInstance(
        abs_ptq_params["logically_partitioned"]["kernel"].array.qvalue,
        nn.LogicallyPartitioned,
    )

    orbax_payload = _to_orbax_payload(reference_params)
    processed_params = checkpoint_util.process_prequantized_params(
        orbax_payload, abs_ptq_params
    )

    # The processed params are unboxed, like the ptq.quantize_params reference.
    _assert_trees_allclose(self, processed_params, reference_params)
    ptq_mlp.apply({"params": processed_params}, model_input)

  def test_process_prequantized_params_linen_manual_dict_template(self):
    # The non-intercepted but pre-quantized kernel is overridden with the
    # manual quantized dictionary structure.
    abs_params = {
        "kernel": {
            "array": {
                "qvalue": jax.ShapeDtypeStruct((12, 6), jnp.int8),
                "scale": jax.ShapeDtypeStruct((1, 6), jnp.float32),
                "zero_point": jax.ShapeDtypeStruct((1, 6), jnp.int8),
            }
        },
        "bias": jax.ShapeDtypeStruct((6,), jnp.float32),
    }
    reference_params = {
        "kernel": {
            "array": {
                "qvalue": jnp.ones((12, 6), dtype=jnp.int8),
                "scale": jnp.ones((1, 6), dtype=jnp.float32),
                "zero_point": jnp.zeros((1, 6), dtype=jnp.int8),
            }
        },
        "bias": jnp.zeros((6,), dtype=jnp.float32),
    }
    orbax_payload = _to_orbax_payload(reference_params)

    processed_params = checkpoint_util.process_prequantized_params(
        orbax_payload, abs_params
    )

    _assert_trees_allclose(self, processed_params, reference_params)

  def test_process_prequantized_params_linen_scan_requantized(self):
    q_rules = [
        qconfig.QuantizationRule(
            module_path=".*", weight_qtype=jnp.int4, tile_size=8
        )
    ]
    model_input = jnp.ones((4, 16))
    fp_model = _ScannedLinenModel()
    ptq_model = qwix_model.quantize_model(fp_model, ptq.PtqProvider(q_rules))
    abs_ptq_params = jax.eval_shape(
        ptq_model.init, jax.random.key(0), model_input
    )["params"]
    fp_params = fp_model.init(jax.random.key(0), model_input)["params"]
    fp_dense = fp_params["layers"]["dense"]
    # ptq.quantize_params can't apply the per-layer WithAux.how to the stacked
    # (3, 16, 16) kernel, so quantize it with the stacked equivalent: the layer
    # axis is channelwise and the per-layer axes shift by one.
    source_kernel = qarray.quantize(
        fp_dense["kernel"],
        qarray.HowToQuantize(
            qtype=jnp.int8, channelwise_axes=(0, 2), tiled_axes={1: 4}
        ),
    )
    orbax_payload = _to_orbax_payload({
        "layers": {
            "dense": {
                "kernel": {"array": source_kernel},
                "bias": fp_dense["bias"],
            }
        }
    })

    processed_params = checkpoint_util.process_prequantized_params(
        orbax_payload, abs_ptq_params
    )

    processed_kernel = processed_params["layers"]["dense"]["kernel"]
    expected_kernel = qarray.quantize(
        qarray.dequantize(source_kernel),
        qarray.HowToQuantize(
            qtype=jnp.int4, channelwise_axes=(0, 2), tiled_axes={1: 8}
        ),
    )
    # WithAux.how keeps the per-layer quantization of the template.
    self.assertEqual(
        processed_kernel.how, abs_ptq_params["layers"]["dense"]["kernel"].how
    )
    _assert_trees_allclose(self, processed_kernel.array, expected_kernel)
    ptq_model.apply({"params": processed_params}, model_input)

  def test_process_prequantized_params_linen_static_act_scale(self):
    q_rules = [
        qconfig.QuantizationRule(
            module_path=".*",
            weight_qtype=jnp.int8,
            act_qtype=jnp.int8,
            act_static_scale=True,
            act_calibration_method="minmax",
        ),
    ]
    model_input = jax.random.normal(jax.random.key(1), (10, 12))
    fp_dense = nn.Dense(features=6)
    qt_dense = qwix_model.quantize_model(fp_dense, qt.QtProvider(q_rules))
    qt_variables = qt_dense.init(jax.random.key(0), model_input)
    _, new_vars = qt_dense.apply(
        qt_variables, model_input, mutable="quant_stats"
    )
    ptq_dense = qwix_model.quantize_model(fp_dense, ptq.PtqProvider(q_rules))
    abs_ptq_params = jax.eval_shape(
        ptq_dense.init, jax.random.key(0), model_input
    )["params"]
    reference_params = ptq.quantize_params(
        qt_variables["params"], abs_ptq_params, new_vars["quant_stats"]
    )

    orbax_payload = _to_orbax_payload(reference_params)
    processed_params = checkpoint_util.process_prequantized_params(
        orbax_payload, abs_ptq_params
    )

    # The static activation scale is a WithAux around a full-precision array.
    self.assertIsInstance(
        processed_params["dot_general0_lhs_scale"], boxed_param.WithAux
    )
    _assert_trees_allclose(self, processed_params, reference_params)
    ptq_dense.apply({"params": processed_params}, model_input)

  def test_process_prequantized_params_linen_frozen_dict(self):
    q_rules = [qconfig.QuantizationRule(weight_qtype=jnp.int8, tile_size=4)]
    model_input = jnp.ones((10, 12))
    _, abs_ptq_params, _, reference_params = _build_linen_reference(
        nn.Dense(features=6), q_rules, model_input
    )

    processed_params = checkpoint_util.process_prequantized_params(
        flax.core.freeze(_to_orbax_payload(reference_params)),
        flax.core.freeze(abs_ptq_params),
    )

    _assert_trees_allclose(self, processed_params, reference_params)

  def test_process_prequantized_params_linen_variables_template(self):
    q_rules = [qconfig.QuantizationRule(weight_qtype=jnp.int8, tile_size=4)]
    model_input = jnp.ones((10, 12))
    _, abs_ptq_params, _, reference_params = _build_linen_reference(
        nn.Dense(features=6), q_rules, model_input
    )
    orbax_payload = _to_orbax_payload(reference_params)

    with self.assertRaisesRegex(ValueError, r"variables\['params'\]"):
      checkpoint_util.process_prequantized_params(
          orbax_payload, {"params": abs_ptq_params}
      )


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

  @parameterized.named_parameters(
      dict(testcase_name="dict", container=dict),
      dict(testcase_name="frozen_dict", container=flax.core.freeze),
  )
  def test_restore_rules_linen(self, container):
    q_rules = [
        qconfig.QuantizationRule(
            module_path=".*", weight_qtype=jnp.int8, tile_size=4
        )
    ]
    model_input = jnp.ones((10, 12))
    _, _, _, reference_params = _build_linen_reference(
        _PartitionedLinenMlp(), q_rules, model_input
    )
    orbax_payload = container(_to_orbax_payload(reference_params))

    rules = checkpoint_util.restore_quantization_rules(
        orbax_payload, qconfig.QuantizationRule, tile_size=4, act_qtype=None
    )

    self.assertCountEqual(
        [rule.module_path for rule in rules],
        ["partitioned", "logically_partitioned"],
    )
    # The restored module paths match the Linen module paths.
    provider = ptq.PtqProvider(rules)
    ptq_mlp = qwix_model.quantize_model(_PartitionedLinenMlp(), provider)
    jax.eval_shape(ptq_mlp.init, jax.random.key(0), model_input)
    self.assertEmpty(provider.get_unused_rules())


if __name__ == "__main__":
  absltest.main()
