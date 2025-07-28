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
"""Quantized training (QT) support."""

import dataclasses
from typing import Any, Callable, Sequence

from flax import linen as nn
from flax import nnx
import flax.linen.dtypes
import jax
from jax import numpy as jnp
from jax.experimental import pallas as pl
from qwix import aux_data
from qwix import averaging
from qwix import flax_util
from qwix import interception
from qwix import qconfig
from qwix.core import dot_general_qt


@dataclasses.dataclass(frozen=True, kw_only=True)
class QtRule(qconfig.QuantizationRule):
  """QuantizationRule with all settings specific to Quantized Training (QT)."""

  # In backward pass, quantize residuals and gradients to the given type.
  bwd_qtype: jax.typing.DTypeLike | None = None
  # In backward pass, calibrate residuals and gradients using the given method.
  bwd_calibration_method: str = 'absmax'
  # In backward pass, enable subchannel for contraction axes when calculating
  # the gradient of weights. Note that the tiling is actually applied to the
  # the incoming gradient and the activation residual rather than any "weight".
  bwd_weight_grad_tile_size: int | float | None = None
  # If True, disable channelwise axes.
  disable_channelwise_axes: bool = True
  # If True, use the original residuals instead of the quantized residuals.
  use_original_residuals: bool = False


class QtProvider(qconfig.QuantizationProvider):
  """Quantization provider for Quantized Training(QT)."""

  def dot_general(
      self,
      lhs: jax.Array,
      rhs: jax.Array,
      dimension_numbers: jax.lax.DotDimensionNumbers,
      precision: jax.lax.PrecisionLike = None,
      preferred_element_type: jax.typing.DTypeLike | None = None,
      *,
      out_sharding=None,
  ) -> jax.Array:
    """QT dot_general."""
    rule, op_id = self._get_current_rule_and_op_id('dot_general')
    if rule is None or rule.weight_qtype is None:
      return jax.lax.dot_general(
          lhs,
          rhs,
          dimension_numbers,
          precision=precision,
          preferred_element_type=preferred_element_type,
          out_sharding=out_sharding,
      )
    config = self._create_dot_general_qt_config(rule, op_id, rhs)
    return dot_general_qt.dot_general_qt(lhs, rhs, dimension_numbers, config)

  def einsum(
      self,
      einsum_str: str,
      *operands: jax.Array,
      precision: jax.lax.PrecisionLike = None,
      preferred_element_type: jax.typing.DTypeLike | None = None,
      _dot_general: Callable[..., jax.Array] = jax.lax.dot_general,  # pylint: disable=invalid-name
      out_sharding=None,
  ) -> jax.Array:
    """QT einsum."""
    rule, op_id = self._get_current_rule_and_op_id('einsum')
    if rule is None or rule.weight_qtype is None:
      return jnp.einsum(
          einsum_str,
          *operands,
          precision=precision,
          preferred_element_type=preferred_element_type,
          _dot_general=_dot_general,
          out_sharding=out_sharding,
      )
    if not isinstance(einsum_str, str) or len(operands) != 2:
      raise ValueError(f'Unsupported einsum format: {einsum_str=} {operands=}')
    _, rhs = operands
    # TODO(jiwonshin): Enforce rhs to be always weight.
    config = self._create_dot_general_qt_config(rule, op_id, rhs)

    custom_dot_general = lambda *args, **kwargs: dot_general_qt.dot_general_qt(
        *args[:3], config
    )

    with jax.disable_jit():
      return jnp.einsum(
          einsum_str,
          *operands,
          precision=precision,
          preferred_element_type=preferred_element_type,
          _dot_general=custom_dot_general,
          out_sharding=out_sharding,
      )

  def nn_param(
      self,
      module: nn.Module,
      name: str,
      init_fn: Callable[..., Any],
      *init_args,
      unbox: bool = True,
      **init_kwargs,
  ) -> jax.Array | nn.meta.AxisMetadata[jax.Array]:
    """Intercepts nn.Module.param."""
    ret = nn.Module.param(
        module, name, init_fn, *init_args, unbox=unbox, **init_kwargs
    )
    aux_data.clear(ret if unbox else ret.unbox())
    aux_data.set(ret if unbox else ret.unbox(), 'weight_name', name)
    return ret

  def promote_dtype(self, *args, **kwargs):
    """Intercepts flax.{linen,nnx.nn}.dtypes.promote_dtype."""
    if len(args) == 1 and isinstance(args[0], Sequence):
      args = args[0]  # nnx version
    ret = flax.linen.dtypes.promote_dtype(*args, **kwargs)
    # Forward weight_name aux_data.
    for x, y in zip(ret, args):
      if y is not None and (name := aux_data.get(y, 'weight_name', None)):
        aux_data.set(x, 'weight_name', name)
    return ret

  def get_intercept_map(self):
    """Used for interception."""
    return {
        # TODO(jiwonshin): add support for quantized conv_general_dilated.
        'jax.lax.dot_general': self.dot_general,
        'jax.numpy.einsum': self.einsum,
        # Disable interception for ops in pallas_call.
        'jax.experimental.pallas.pallas_call': (
            lambda *args, **kwargs: interception.disable_interceptions(
                pl.pallas_call(*args, **kwargs)
            )
        ),
        'flax.linen.Module.param': self.nn_param,
        'flax.linen.dtypes.promote_dtype': self.promote_dtype,
        'flax.nnx.nn.dtypes.promote_dtype': self.promote_dtype,
    }

  def process_model_inputs(
      self, model: nn.Module | nnx.Module, model_args: Any, model_kwargs: Any
  ) -> tuple[nn.Module | nnx.Module, Any, Any]:
    """Processes the nnx.Module instance before it is called."""
    if isinstance(model, nnx.Module):
      for path, node in nnx.iter_graph(model):
        if isinstance(node, nnx.Module):
          aux_data.clear(node)  # clear the op_count.
        elif isinstance(node, nnx.Param):
          # weight_name is used to distinguish weights from activations.
          aux_data.clear(node.value)
          aux_data.set(node.value, 'weight_name', path[-1])
    return model, model_args, model_kwargs

  def _collect_quant_stat(
      self,
      name: str,
      calibration: averaging.Calibration,
  ) -> averaging.Calibration:
    """Collects the quantization statistics."""
    aggregator = averaging.SimpleMovingAverage()
    quant_stat = flax_util.get_or_create_variable(
        'quant_stats', name, lambda: aggregator.init(calibration)
    )

    if flax_util.should_update_quant_stats():
      quant_stat.value = aggregator.update(quant_stat.value, calibration)

    return aggregator.get_calibration(quant_stat.value, calibration)

  def _create_dot_general_qt_config(
      self, rule: qconfig.QuantizationRule, op_id: str, rhs: jax.Array
  ) -> dot_general_qt.DotGeneralQtConfig:
    """Creates a DotGeneralQtConfig for dot_general and einsum."""
    if not isinstance(rule, QtRule):
      rule = QtRule(**dataclasses.asdict(rule))
    # LHS is always considered an activation for quantization purposes.
    lhs_qtype = None
    lhs_quant_stat_name = None
    lhs_calibration_method = None
    lhs_batch_axes = ()
    if rule.act_qtype is not None:
      lhs_qtype = rule.act_qtype
      lhs_calibration_method = rule.act_calibration_method
      if rule.act_static_scale:
        lhs_quant_stat_name = f'{op_id}_lhs'
        lhs_batch_axes = rule.act_batch_axes

    # RHS configs based on whether it's a weight or an activation.
    rhs_qtype = None
    rhs_quant_stat_name = None
    rhs_calibration_method = None
    rhs_batch_axes = ()
    is_weight = aux_data.get(rhs, 'weight_name', None) is not None

    if is_weight:
      rhs_qtype = rule.weight_qtype
      rhs_calibration_method = rule.weight_calibration_method
    elif rule.act_qtype is not None:
      rhs_qtype = rule.act_qtype
      rhs_calibration_method = rule.act_calibration_method
      if rule.act_static_scale:
        rhs_quant_stat_name = f'{op_id}_rhs'
        rhs_batch_axes = rule.act_batch_axes

    return dot_general_qt.DotGeneralQtConfig(
        lhs_qtype=lhs_qtype,
        rhs_qtype=rhs_qtype,
        bwd_qtype=rule.bwd_qtype,
        bwd_drhs_tile_size=rule.bwd_weight_grad_tile_size,
        tile_size=rule.tile_size,
        lhs_calibration_method=lhs_calibration_method,
        lhs_batch_axes=lhs_batch_axes,
        lhs_quant_stat_name=lhs_quant_stat_name,
        rhs_calibration_method=rhs_calibration_method,
        rhs_batch_axes=rhs_batch_axes,
        rhs_quant_stat_name=rhs_quant_stat_name,
        bwd_calibration_method=rule.bwd_calibration_method,
        collect_quant_stat=self._collect_quant_stat,
        disable_channelwise_axes=rule.disable_channelwise_axes,
        use_original_residuals=rule.use_original_residuals,
    )
