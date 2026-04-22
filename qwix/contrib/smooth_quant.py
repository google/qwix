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

"""Smooth Quantization (SQ) implementation in Qwix.

Arxiv Link: https://arxiv.org/abs/2211.10438
ICML Link: https://icml.cc/virtual/2023/poster/25228
GitHub Link: https://github.com/mit-han-lab/smoothquant
"""

import dataclasses
from typing import Any

import flax
from flax import nnx
import flax.linen as nn
import jax
import jax.numpy as jnp
from qwix._src import averaging
from qwix._src import flax_util
from qwix._src import qconfig
from qwix._src.core import dot_general
from qwix._src.core import qarray
from qwix._src.providers import ptq
from qwix.contrib import calibration


def compute_scales_ratio(
    act_stats: jax.Array, weight_stats: jax.Array, alpha: float
) -> jax.Array:
  """Compute the scales ratio for Smooth Quantization (SQ)."""
  # https://github.com/mit-han-lab/smoothquant/blob/main/smoothquant/smooth.py
  scales = act_stats**alpha / weight_stats ** (1 - alpha)
  return jnp.clip(scales, min=1e-5).astype(weight_stats.dtype)


@dataclasses.dataclass(frozen=True, kw_only=True)
class SqRule(qconfig.QuantizationRule):
  """Use this rule to enable SQ.

  Attributes:
    alpha: Smooth Quant parameter. Default is 0.5.
  """

  alpha: float = 0.5


@flax.struct.dataclass(kw_only=True)
class WithSqScale(ptq.WithAux[qarray.QArray]):
  """A quantized array with SQ per-channel scales.

  This wrapper stores the quantized weights along with the per-channel SQ
  scales. During inference, the SqInferenceProvider dequantizes the weights
  and divides by the SQ scales to get the final weights.

  Attributes:
    inv_sq_scale: Per-channel SQ scales with shape (in_features,). This is a 1D
      array that will be broadcast along the contracting axis during inference.
      This is the inverse of the scale factor, so we multiply when using with
      activations.
    contracting_axis: The axis of the weight that is contracted in dot_general.
  """

  inv_sq_scale: jax.Array
  contracting_axis: int = flax.struct.field(pytree_node=False)


# Register as NNX data to allow JAX arrays in Module attributes.
nnx.register_data_type(WithSqScale)


class SqCalibrationProvider(calibration.SinglePassCalibrationProvider):
  """Calibration provider for SQ.

  This provider collects `sq_quant_stats` (per-channel activation scales) by
  using `CalibrationProvider` to intercept compatible operations. These
  statistics are used by `quantize_params` to compute SQ scales. This provider
  does not perform actual quantization or use quantized operations.
  """

  def get_rule_type(self) -> type[qconfig.QuantizationRule]:
    return SqRule

  def compute_stats(self, lhs: jax.Array) -> dict[str, Any]:
    # Get the alpha parameter from the current rule.
    rule, _ = self._get_current_rule_and_op_id("dot_general")
    assert isinstance(
        rule, SqRule
    ), f"Rule type {type(rule)} does not match expected type SqRule."
    alpha = rule.alpha

    # Access module parameters
    module = flax_util.get_current_module()
    try:
      if isinstance(module, nn.Module):
        assert module.scope is not None
        kernel = module.scope._collection("params")["kernel"]  # pylint: disable=protected-access
      else:
        kernel = module.kernel.get_value()
    except KeyError as exc:
      raise NotImplementedError(
          f"Failed to extract kernel from module {module}. Only "
          "Flax nn.Module and nnx.Module are supported."
      ) from exc

    # Compute weight scales
    how = qarray.HowToQuantize(
        qtype=rule.weight_qtype,
        channelwise_axes=(0,),
        calibration_method=rule.weight_calibration_method,
    )
    calib = qarray.calibrate(kernel, how)
    weight_scale, zero_point = qarray.compute_scale_zero_point(calib, how.qtype)
    assert zero_point is None, (
        "SQ does not support zero-points. Select a symmetric quantization"
        " scheme."
    )

    # Compute the act scales
    if rule.act_qtype is None:
      raise ValueError(
          f"Rule {rule} has no activation quantization type specified. Cannot"
          " compute activation scales for SQ."
      )
    how = qarray.HowToQuantize(
        qtype=rule.act_qtype,
        channelwise_axes=(0,),
        calibration_method=rule.act_calibration_method,
    )
    calib = qarray.calibrate(lhs, how)
    act_scale, zero_point = qarray.compute_scale_zero_point(calib, how.qtype)
    assert zero_point is None, (
        "SQ does not support zero-points. Select a symmetric quantization"
        " scheme."
    )
    assert act_scale.shape == weight_scale.shape, (
        f"Activation scales {act_scale.shape} do not match weight scales"
        f" {weight_scale.shape}"
    )

    # Compute the SQ scales and their inverses
    sq_scales = compute_scales_ratio(
        act_scale, weight_scale, alpha=alpha
    ).reshape(-1)
    return {"sq_scale": sq_scales, "inv_sq_scale": 1 / sq_scales}

  def get_stats_suffix(self) -> str:
    return "_sq"


def quantize_params(
    params: Any,
    abstract_quantized_params: Any,
    sq_quant_stats: Any,
    *,
    allow_extra_params: bool = False,
) -> Any:
  """Quantizes parameters with Smooth Quantization (SQ).

  Args:
    params: The floating-point param tree to quantize.
    abstract_quantized_params: The param tree generated by the PTQ model,
      containing WithAux wrappers with HowToQuantize information.
    sq_quant_stats: The quant_stats dict from SqCalibrationProvider. For params
      with no sq_quant_stats, they will be quantized with the default PTQ
      algorithm.
    allow_extra_params: If True, allow extra parameters not in
      abstract_quantized_params.

  Returns:
    The quantized params consumable by SqInferenceProvider. For SQ-quantized
    weights, returns WithSqScale wrappers containing the QArray and per-channel
    SQ scales. For non-SQ weights, returns WithAux wrappers (same as PTQ).
  """
  quantized_params = {}
  not_quantized_params = {}
  for path, w in flax.traverse_util.flatten_dict(params).items():
    abs_w = ptq.get_value_from_path(abstract_quantized_params, path)
    sq_stats_path = (*path[:-1], path[-1] + "_sq")
    sq_stats = ptq.get_value_from_path(sq_quant_stats, sq_stats_path)

    if not isinstance(abs_w, ptq.WithAux) or sq_stats is None:
      # Not quantized by SQ.
      not_quantized_params[path] = w
      continue

    # Get the contracting axis by assuming that all non-contracting axes
    # are in channelwise_axes as in AWQ.
    contracting_axis = set(range(w.ndim)) - set(abs_w.how.channelwise_axes)
    if len(contracting_axis) != 1:
      # Fallback to PTQ if we can't identify a single contracting axis.
      not_quantized_params[path] = w
      continue

    contracting_axis = list(contracting_axis)[0]

    # Normalize the weight to (ra, ca) format.
    w, restore_shape = calibration.normalize_weight(w, contracting_axis)
    how = dataclasses.replace(abs_w.how, channelwise_axes=[0])
    if contracting_axis in how.tiled_axes:
      how = dataclasses.replace(
          how, tiled_axes={1: how.tiled_axes[contracting_axis]}
      )

    # Get the activation scale, which should be (ca,).
    calibration_stats = averaging.SimpleMovingAverage().get_calibration(
        sq_stats
    )
    sq_scales = calibration_stats["sq_scale"]
    assert sq_scales.shape[0] == w.shape[1]

    # Quantize the weight with SQ.
    # Scale up salient channels before quantization.
    w_scaled = w * sq_scales

    # Quantize the scaled weights (may use groupwise quantization).
    w_q = qarray.quantize(w_scaled, how)

    # Store SQ scales separately for per-channel compensation during inference.
    quantized_params[path] = WithSqScale(
        array=restore_shape(w_q),
        inv_sq_scale=calibration_stats["inv_sq_scale"],
        contracting_axis=contracting_axis,
        how=abs_w.how,
    )

  # Quantize the non-SQ params with PTQ.
  not_quantized_params = flax.traverse_util.unflatten_dict(not_quantized_params)
  ptq_quantized_params = ptq.quantize_params(
      not_quantized_params,
      abstract_quantized_params,
      allow_extra_params=allow_extra_params,
  )
  ptq_quantized_params = flax.traverse_util.flatten_dict(ptq_quantized_params)
  quantized_params.update(ptq_quantized_params)

  return flax.traverse_util.unflatten_dict(quantized_params)


class SqInferenceProvider(ptq.PtqProvider):
  """Inference provider for SQ."""

  def _apply_sq_scale(
      self, lhs: jax.Array, inv_sq_scale: jax.Array
  ) -> jax.Array:
    """Applies per-channel SQ scale compensation."""
    if isinstance(lhs, qarray.QArray):
      raise NotImplementedError("LHS inputs to SQ should not be pre-quantized.")
    else:
      lhs_dq = lhs
    return lhs_dq * inv_sq_scale

  def dot_general(
      self,
      lhs: jax.Array,
      rhs: jax.Array | WithSqScale | ptq.WithAux[qarray.QArray],
      dimension_numbers: jax.lax.DotDimensionNumbers,
      precision: jax.lax.PrecisionLike = None,
      preferred_element_type: jax.typing.DTypeLike | None = None,
      *,
      out_sharding: jax.sharding.NamedSharding | None = None,
  ) -> jax.Array:
    # Handle SQ-quantized weights with per-channel scale compensation.
    if isinstance(rhs, WithSqScale):
      lhs = self._apply_sq_scale(lhs, rhs.inv_sq_scale)
      rhs = rhs.array

    return dot_general.dot_general(
        lhs,
        rhs,
        dimension_numbers,
        precision=precision,
        preferred_element_type=preferred_element_type,
        out_sharding=out_sharding,
    )

  def get_intercept_map(self):
    """Used for interception."""
    return super().get_intercept_map() | {
        "jax.lax.dot_general": self.dot_general,
    }
