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
import functools
from typing import Any, Callable, Mapping, Sequence

from flax import linen as nn
from flax import nnx
import jax
from jax import numpy as jnp
from qwix._src import aux_data
from qwix._src import averaging
from qwix._src import flax_util
from qwix._src import qconfig
from qwix._src.core import conv_general_qt
from qwix._src.core import dot_general_qt


@dataclasses.dataclass(frozen=True, kw_only=True)
class QtRule(qconfig.QuantizationRule):
  """QuantizationRule with all settings specific to Quantized Training (QT)."""

  # In backward pass, quantize the gradients to the given type. If set, the
  # residuals will also be quantized with the same qtype as in the forward pass.
  bwd_qtype: jax.typing.DTypeLike | None = None

  # In backward pass, calibrate the gradients using the given method.
  bwd_calibration_method: str = 'absmax'

  # In backward pass, enable subchannel for contraction axes when calculating
  # the gradient of weights. Note that the tiling is actually applied to the
  # the incoming gradient and the residual activation rather than any "weight".
  bwd_weight_grad_tile_size: int | float | None = None

  # If True, disable channelwise axes for both forward and backward passes.
  disable_channelwise_axes: bool = False

  # If True, use the original values instead of the quantized values as the
  # residuals for backward pass.
  bwd_use_original_residuals: bool = False

  # Override any fields in DotGeneralQtConfig.
  additional_qt_config: Mapping[str, Any] | None = None


class QtProvider(qconfig.QuantizationProvider):
  """Quantization provider for Quantized Training (QT)."""

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
    config = self._create_dot_general_qt_config(rule, op_id, lhs, rhs)
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

    def custom_dot_general(
        lhs,
        rhs,
        dimension_numbers,
        precision,
        preferred_element_type,
        **kwargs,
    ):
      # TODO(dangyi): support preferred_element_type.
      del precision, preferred_element_type, kwargs
      return dot_general_qt.dot_general_qt(
          lhs,
          rhs,
          dimension_numbers,
          # lhs and rhs might be flipped by einsum so we cannot use the operands
          # from the einsum call.
          self._create_dot_general_qt_config(rule, op_id, lhs, rhs),
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

  def conv_general_dilated(
      self,
      lhs: jax.Array,
      rhs: jax.Array,
      window_strides: Sequence[int],
      padding: str | Sequence[tuple[int, int]],
      lhs_dilation: Sequence[int] | None = None,
      rhs_dilation: Sequence[int] | None = None,
      dimension_numbers: jax.lax.ConvGeneralDilatedDimensionNumbers = None,
      feature_group_count: int = 1,
      batch_group_count: int = 1,
      precision: jax.lax.PrecisionLike = None,
      preferred_element_type: jax.typing.DTypeLike | None = None,
  ) -> jax.Array:
    """QT conv_general_dilated."""
    rule, op_id = self._get_current_rule_and_op_id('conv_general_dilated')
    if rule is None or rule.weight_qtype is None:
      return jax.lax.conv_general_dilated(
          lhs,
          rhs,
          window_strides,
          padding,
          lhs_dilation=lhs_dilation,
          rhs_dilation=rhs_dilation,
          dimension_numbers=dimension_numbers,
          feature_group_count=feature_group_count,
          batch_group_count=batch_group_count,
          precision=precision,
          preferred_element_type=preferred_element_type,
      )
    if rule.tile_size:
      raise ValueError('subchannel is not supported for conv_general_dilated.')
    config = self._create_conv_general_qt_config(rule, op_id, lhs, rhs)
    return conv_general_qt.conv_general_qt(
        lhs,
        rhs,
        config,
        window_strides,
        padding,
        lhs_dilation,
        rhs_dilation,
        dimension_numbers,
        feature_group_count,
        batch_group_count,
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

  def asarray(self, a, *args, **kwargs):
    """Intercepts jnp.asarray."""
    ret = jnp.asarray(a, *args, **kwargs)
    # Forward weight_name aux_data.
    if name := aux_data.get(a, 'weight_name', None):
      aux_data.set(ret, 'weight_name', name)
    return ret

  def get_intercept_map(self):
    """Used for interception."""
    return super().get_intercept_map() | {
        'jax.lax.conv_general_dilated': self.conv_general_dilated,
        'jax.lax.dot_general': self.dot_general,
        'jax.numpy.einsum': self.einsum,
        'flax.linen.Module.param': self.nn_param,  # to associate weight_name.
        'jax.numpy.asarray': self.asarray,  # to forward weight_name.
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
      batch_axes: tuple[int, ...],
      calibration: averaging.Calibration,
  ) -> averaging.Calibration:
    """Collects the quantization statistics."""
    # Calculate the mean over the batch axes.
    calibration = jax.tree.map(
        lambda x: x.mean(axis=batch_axes, keepdims=True), calibration
    )

    aggregator = averaging.SimpleMovingAverage()
    quant_stat = flax_util.get_or_create_variable(
        'quant_stats', name, lambda: aggregator.init(calibration)
    )

    if flax_util.should_update_quant_stats():
      quant_stat.value = aggregator.update(quant_stat.value, calibration)

    return aggregator.get_calibration(quant_stat.value, calibration)

  def _create_conv_general_qt_config(
      self,
      rule: qconfig.QuantizationRule,
      op_id: str,
      lhs: jax.Array,
      rhs: jax.Array,
  ) -> conv_general_qt.ConvGeneralQtConfig:
    """Creates a ConvGeneralQtConfig for conv_general_dilated."""
    if not isinstance(rule, QtRule):
      rule = QtRule(**dataclasses.asdict(rule))

    if rule.weight_qtype != rule.act_qtype:
      raise ValueError(
          'conv_general_qt requires the same weight_qtype and act_qtype.'
      )
    if rule.weight_calibration_method != rule.act_calibration_method:
      # This is not strictly required, but ConvGeneralQtConfig doesn't support
      # individual configurations for now.
      raise ValueError(
          'conv_general_qt requires the same weight_calibration_method and'
          ' act_calibration_method.'
      )
    if rule.bwd_qtype is not None:
      if rule.bwd_qtype != rule.weight_qtype:
        raise ValueError(
            'conv_general_qt requires the same bwd_qtype as weight_qtype.'
        )
      if rule.bwd_calibration_method != rule.weight_calibration_method:
        raise ValueError(
            'conv_general_qt requires the same bwd_calibration_method as'
            ' weight_calibration_method.'
        )

    fwd_qtype = rule.weight_qtype
    fwd_calibration_method = rule.weight_calibration_method

    lhs_is_weight = aux_data.get(lhs, 'weight_name', None) is not None
    lhs_collect_quant_stat = None
    if (
        not lhs_is_weight
        and rule.act_qtype is not None
        and rule.act_static_scale
    ):
      lhs_collect_quant_stat = functools.partial(
          self._collect_quant_stat, f'{op_id}_lhs', rule.act_batch_axes
      )

    rhs_is_weight = aux_data.get(rhs, 'weight_name', None) is not None
    rhs_collect_quant_stat = None
    if (
        not rhs_is_weight
        and rule.act_qtype is not None
        and rule.act_static_scale
    ):
      rhs_collect_quant_stat = functools.partial(
          self._collect_quant_stat, f'{op_id}_rhs', rule.act_batch_axes
      )

    return conv_general_qt.ConvGeneralQtConfig(
        # fwd configs.
        fwd_qtype=fwd_qtype,
        fwd_calibration_method=fwd_calibration_method,
        lhs_collect_quant_stat=lhs_collect_quant_stat,
        rhs_collect_quant_stat=rhs_collect_quant_stat,
        # bwd configs.
        bwd_qtype=rule.bwd_qtype,
        bwd_calibration_method=rule.bwd_calibration_method,
        # misc.
        disable_channelwise_axes=rule.disable_channelwise_axes,
        bwd_use_original_residuals=rule.bwd_use_original_residuals,
    )

  def _create_dot_general_qt_config(
      self,
      rule: qconfig.QuantizationRule,
      op_id: str,
      lhs: jax.Array,
      rhs: jax.Array,
  ) -> dot_general_qt.DotGeneralQtConfig:
    """Creates a DotGeneralQtConfig for dot_general and einsum."""
    if not isinstance(rule, QtRule):
      rule = QtRule(**dataclasses.asdict(rule))

    # LHS configs based on whether it's a weight or an activation.
    lhs_qtype = None
    lhs_calibration_method = None
    lhs_is_weight = aux_data.get(lhs, 'weight_name', None) is not None
    lhs_collect_quant_stat = None

    if lhs_is_weight:
      if rule.weight_qtype is not None:
        lhs_qtype = rule.weight_qtype
        lhs_calibration_method = rule.weight_calibration_method
    elif rule.act_qtype is not None:
      lhs_qtype = rule.act_qtype
      lhs_calibration_method = rule.act_calibration_method
      if rule.act_static_scale:
        lhs_collect_quant_stat = functools.partial(
            self._collect_quant_stat, f'{op_id}_lhs', rule.act_batch_axes
        )

    # RHS configs based on whether it's a weight or an activation.
    rhs_qtype = None
    rhs_calibration_method = None
    rhs_is_weight = aux_data.get(rhs, 'weight_name', None) is not None
    rhs_collect_quant_stat = None

    if rhs_is_weight:
      if rule.weight_qtype is not None:
        rhs_qtype = rule.weight_qtype
        rhs_calibration_method = rule.weight_calibration_method
    elif rule.act_qtype is not None:
      rhs_qtype = rule.act_qtype
      rhs_calibration_method = rule.act_calibration_method
      if rule.act_static_scale:
        rhs_collect_quant_stat = functools.partial(
            self._collect_quant_stat, f'{op_id}_rhs', rule.act_batch_axes
        )

    # bwd config, which is only enabled when bwd_qtype is set.
    dlhs_rhs_qtype = None
    dlhs_tile_size = None
    drhs_rhs_qtype = None
    drhs_tile_size = None

    if rule.bwd_qtype is not None:
      dlhs_rhs_qtype = rhs_qtype  # dlhs_rhs is the residual rhs.
      drhs_rhs_qtype = lhs_qtype  # drhs_rhs is the residual lhs.
      if lhs_is_weight:
        dlhs_tile_size = rule.bwd_weight_grad_tile_size
      if rhs_is_weight:
        drhs_tile_size = rule.bwd_weight_grad_tile_size

    qt_config = dot_general_qt.DotGeneralQtConfig(
        # fwd configs.
        lhs_qtype=lhs_qtype,
        rhs_qtype=rhs_qtype,
        tile_size=rule.tile_size,
        lhs_calibration_method=lhs_calibration_method,
        rhs_calibration_method=rhs_calibration_method,
        lhs_collect_quant_stat=lhs_collect_quant_stat,
        rhs_collect_quant_stat=rhs_collect_quant_stat,
        # dlhs configs.
        dlhs_lhs_qtype=rule.bwd_qtype,
        dlhs_rhs_qtype=dlhs_rhs_qtype,
        dlhs_tile_size=dlhs_tile_size,
        dlhs_lhs_calibration_method=rule.bwd_calibration_method,
        dlhs_rhs_calibration_method=rhs_calibration_method,
        # drhs configs.
        drhs_lhs_qtype=rule.bwd_qtype,
        drhs_rhs_qtype=drhs_rhs_qtype,
        drhs_tile_size=drhs_tile_size,
        drhs_lhs_calibration_method=rule.bwd_calibration_method,
        drhs_rhs_calibration_method=lhs_calibration_method,
        # misc.
        disable_channelwise_axes=rule.disable_channelwise_axes,
        bwd_use_original_residuals=rule.bwd_use_original_residuals,
    )

    if rule.additional_qt_config:
      qt_config = dataclasses.replace(qt_config, **rule.additional_qt_config)
    return qt_config
