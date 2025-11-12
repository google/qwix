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

import jax
from jax import numpy as jnp
from qwix._src import averaging
from qwix._src import flax_util
from qwix._src import qconfig
from qwix._src.core import conv_general_qt
from qwix._src.core import dot_general_qt
from qwix._src.core import ragged_dot_qt
from qwix._src.core import stochastic_rounding


@dataclasses.dataclass(frozen=True, kw_only=True)
class QtRule(qconfig.QuantizationRule):
  """QuantizationRule with all settings specific to Quantized Training (QT)."""

  # In backward pass, quantize the gradients to the given type. This doesn't
  # affect the residuals as the residuals will reuse the quantization in the
  # forward pass.
  bwd_qtype: jax.typing.DTypeLike | None = None

  # In backward pass, calibrate the gradients using the given method.
  bwd_calibration_method: str = 'absmax'

  # In backward pass, enable subchannel for contraction axes when calculating
  # the gradient of weights. Note that the tiling is actually applied to the
  # the incoming gradient and the residual activation rather than any "weight".
  bwd_weight_grad_tile_size: int | float | None = None

  # If True, disable channelwise axes for both forward and backward passes.
  disable_channelwise_axes: bool = False

  # Use stochastic rounding for the gradients.
  # Currently supports "uniform" and "low_bit_uniform".
  bwd_stochastic_rounding: str | None = None

  # Use channelwise noise for stochastic rounding. By default, it will generate
  # noise for the 0th dimension and broadcast it over remaining dimensions.
  channelwise_noise_axes: Sequence[int] = (0,)

  # Override any fields in DotGeneralQtConfig or ConvGeneralQtConfig. This is
  # highly experimental and subjects to changes with no backward compatibility
  # guarantees.
  additional_qt_config: Mapping[str, Any] | None = None


class QtProvider(qconfig.QuantizationProvider):
  """Quantization provider for Quantized Training (QT)."""

  def _init_rule(self, rule: qconfig.QuantizationRule) -> QtRule:
    rule = super()._init_rule(rule)
    if not isinstance(rule, QtRule):
      rule = QtRule(**dataclasses.asdict(rule))
    return rule

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

  def ragged_dot(
      self,
      lhs: jax.Array,
      rhs: jax.Array,
      group_sizes: jax.Array,
      precision: jax.lax.PrecisionLike = None,
      preferred_element_type: jax.typing.DTypeLike | None = None,
      group_offset: jax.Array | None = None,
  ) -> jax.Array:
    """QT ragged_dot."""
    rule, _ = self._get_current_rule_and_op_id('ragged_dot')
    if rule is None or rule.weight_qtype is None:
      return jax.lax.ragged_dot(
          lhs,
          rhs,
          group_sizes,
          precision=precision,
          preferred_element_type=preferred_element_type,
          group_offset=group_offset,
      )
    config = self._create_ragged_dot_qt_config(rule)
    return ragged_dot_qt.ragged_dot_qt(
        lhs,
        rhs,
        group_sizes,
        config,
        precision,
        preferred_element_type,
        group_offset,
    )

  def get_intercept_map(self):
    """Used for interception."""
    return super().get_intercept_map() | {
        'jax.lax.conv_general_dilated': self.conv_general_dilated,
        'jax.lax.dot_general': self.dot_general,
        'jax.numpy.einsum': self.einsum,
        'jax.lax.ragged_dot': self.ragged_dot,
    }

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
    assert isinstance(rule, QtRule), '_init_rule should have been called.'

    # Assume LHS is an activation and RHS is a weight.
    del lhs
    lhs_collect_quant_stat = None
    if rule.act_qtype is not None and rule.act_static_scale:
      lhs_collect_quant_stat = functools.partial(
          self._collect_quant_stat, f'{op_id}_lhs', rule.act_batch_axes
      )
    assert flax_util.find_param(rhs) is not None

    return conv_general_qt.ConvGeneralQtConfig(
        # fwd configs.
        lhs_qtype=rule.act_qtype,
        rhs_qtype=rule.weight_qtype,
        lhs_calibration_method=rule.act_calibration_method,
        rhs_calibration_method=rule.weight_calibration_method,
        lhs_collect_quant_stat=lhs_collect_quant_stat,
        rhs_collect_quant_stat=None,
        # bwd configs.
        dlhs_grad_qtype=rule.bwd_qtype,
        dlhs_grad_calibration_method=rule.bwd_calibration_method,
        drhs_grad_qtype=rule.bwd_qtype,
        drhs_grad_calibration_method=rule.bwd_calibration_method,
        # misc.
        disable_channelwise_axes=rule.disable_channelwise_axes,
    )

  def _create_dot_general_qt_config(
      self,
      rule: qconfig.QuantizationRule,
      op_id: str,
      lhs: jax.Array,
      rhs: jax.Array,
  ) -> dot_general_qt.DotGeneralQtConfig:
    """Creates a DotGeneralQtConfig for dot_general and einsum."""
    assert isinstance(rule, QtRule), '_init_rule should have been called.'

    # LHS configs based on whether it's a weight or an activation.
    lhs_qtype = None
    lhs_calibration_method = None
    lhs_is_weight = flax_util.find_param(lhs) is not None
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
    rhs_is_weight = flax_util.find_param(rhs) is not None
    rhs_collect_quant_stat = None

    if rhs_is_weight:
      assert not lhs_is_weight, 'lhs and rhs cannot be both weights.'
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
    dlhs_tile_size = None
    drhs_tile_size = None
    bwd_stochastic_rounding_noise_fn = None

    if rule.bwd_qtype is not None:
      if lhs_is_weight:
        dlhs_tile_size = rule.bwd_weight_grad_tile_size
      if rhs_is_weight:
        drhs_tile_size = rule.bwd_weight_grad_tile_size
      if rule.bwd_stochastic_rounding is not None:
        bwd_stochastic_rounding_noise_fn = stochastic_rounding.get_noise_fn(
            method=rule.bwd_stochastic_rounding,
            key=flax_util.make_rng('stochastic_rounding'),
            channelwise_noise_axes=rule.channelwise_noise_axes,
        )

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
        dlhs_grad_qtype=rule.bwd_qtype,
        dlhs_grad_calibration_method=rule.bwd_calibration_method,
        dlhs_tile_size=dlhs_tile_size,
        dlhs_stochastic_rounding_noise_fn=bwd_stochastic_rounding_noise_fn,
        # drhs configs.
        drhs_grad_qtype=rule.bwd_qtype,
        drhs_tile_size=drhs_tile_size,
        drhs_grad_calibration_method=rule.bwd_calibration_method,
        drhs_stochastic_rounding_noise_fn=bwd_stochastic_rounding_noise_fn,
        # misc.
        disable_channelwise_axes=rule.disable_channelwise_axes,
    )

    if rule.additional_qt_config:
      qt_config = dataclasses.replace(qt_config, **rule.additional_qt_config)
    return qt_config

  def _create_ragged_dot_qt_config(
      self,
      rule: qconfig.QuantizationRule,
  ) -> ragged_dot_qt.RaggedDotQtConfig:
    """Creates a RaggedDotQtConfig for ragged_dot."""
    assert isinstance(rule, QtRule), '_init_rule should have been called.'
    # Assume LHS is an activation and RHS is a weight.
    return ragged_dot_qt.RaggedDotQtConfig(
        # fwd configs.
        lhs_qtype=rule.act_qtype,
        rhs_qtype=rule.weight_qtype,
        # bwd configs.
        dlhs_grad_qtype=rule.bwd_qtype,
        drhs_grad_qtype=rule.bwd_qtype,
    )
