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
"""Quantization-aware training (QAT) support."""

import dataclasses
import functools
from typing import Any, Callable, Sequence

from flax import linen as nn
from flax import nnx
import flax.linen.dtypes
import jax
from jax import numpy as jnp
from qwix import aux_data
from qwix import averaging
from qwix import flax_util
from qwix import qconfig
from qwix.core import conv_general
from qwix.core import dot_general
from qwix.core import einsum
from qwix.core import qarray


class QatProvider(qconfig.QuantizationProvider):
  """Quantization provider for QAT."""

  def dot_general(
      self,
      lhs: jax.Array,
      rhs: jax.Array,
      dimension_numbers: jax.lax.DotDimensionNumbers,
      precision: jax.lax.PrecisionLike = None,
      preferred_element_type: jax.typing.DTypeLike | None = None,
  ) -> jax.Array:
    """QAT dot_general."""
    rule, op_id = self._get_current_rule_and_op_id('dot_general')
    if rule is None or rule.weight_qtype is None:
      return jax.lax.dot_general(
          lhs,
          rhs,
          dimension_numbers,
          precision=precision,
          preferred_element_type=preferred_element_type,
      )

    get_how_to_quantize = functools.partial(
        dot_general.get_how_to_quantize,
        dimension_numbers=dimension_numbers,
        ndims=(len(lhs.shape), len(rhs.shape)),
    )

    # Prepare rhs.
    if aux_data.get(rhs, 'weight_name', None):
      rhs_how = get_how_to_quantize(
          for_lhs=False,
          qtype=rule.weight_qtype,
          tile_size=rule.tile_size,
          calibration_method=rule.weight_calibration_method,
          batch_axes=(),
      )
      rhs = self._fake_quant(rhs, rhs_how)
    elif rule.act_qtype is not None:
      rhs_how = get_how_to_quantize(
          for_lhs=False,
          qtype=rule.act_qtype,
          tile_size=rule.tile_size,
          calibration_method=rule.act_calibration_method,
          batch_axes=rule.act_batch_axes if rule.act_static_scale else (),
      )
      quant_stat_name = op_id + '_rhs' if rule.act_static_scale else None
      rhs = self._fake_quant(rhs, rhs_how, quant_stat_name)

    # Prepare lhs.
    if rule.act_qtype is not None:
      lhs_how = get_how_to_quantize(
          for_lhs=True,
          qtype=rule.act_qtype,
          tile_size=rule.tile_size,
          calibration_method=rule.act_calibration_method,
          batch_axes=rule.act_batch_axes if rule.act_static_scale else (),
      )
      quant_stat_name = op_id + '_lhs' if rule.act_static_scale else None
      lhs = self._fake_quant(lhs, lhs_how, quant_stat_name)

    return jax.lax.dot_general(lhs, rhs, dimension_numbers)

  def einsum(
      self,
      einsum_str: str,
      *operands: jax.Array,
      precision: jax.lax.PrecisionLike = None,
      preferred_element_type: jax.typing.DTypeLike | None = None,
      _dot_general: Callable[..., jax.Array] = jax.lax.dot_general,  # pylint: disable=invalid-name
  ) -> jax.Array:
    """QAT einsum."""
    rule, op_id = self._get_current_rule_and_op_id('einsum')
    if rule is None or rule.weight_qtype is None:
      return jax.numpy.einsum(
          einsum_str,
          *operands,
          precision=precision,
          preferred_element_type=preferred_element_type,
          _dot_general=_dot_general,
      )
    if not isinstance(einsum_str, str) or len(operands) != 2:
      raise ValueError(f'Unsupported einsum format: {einsum_str=} {operands=}')
    lhs, rhs = operands
    get_how_to_quantize = functools.partial(
        einsum.get_how_to_quantize,
        einsum_str=einsum_str,
        ndims=(len(lhs.shape), len(rhs.shape)),
    )

    # Prepare rhs.
    if aux_data.get(rhs, 'weight_name', None):
      rhs_how = get_how_to_quantize(
          for_lhs=False,
          qtype=rule.weight_qtype,
          tile_size=rule.tile_size,
          calibration_method=rule.weight_calibration_method,
          batch_axes=(),
      )
      rhs = self._fake_quant(rhs, rhs_how)
    elif rule.act_qtype is not None:
      rhs_how = get_how_to_quantize(
          for_lhs=False,
          qtype=rule.act_qtype,
          tile_size=rule.tile_size,
          calibration_method=rule.act_calibration_method,
          batch_axes=rule.act_batch_axes if rule.act_static_scale else (),
      )
      quant_stat_name = op_id + '_rhs' if rule.act_static_scale else None
      rhs = self._fake_quant(rhs, rhs_how, quant_stat_name)

    # Prepare lhs.
    if rule.act_qtype is not None:
      lhs_how = get_how_to_quantize(
          for_lhs=True,
          qtype=rule.act_qtype,
          tile_size=rule.tile_size,
          calibration_method=rule.act_calibration_method,
          batch_axes=rule.act_batch_axes if rule.act_static_scale else (),
      )
      quant_stat_name = op_id + '_lhs' if rule.act_static_scale else None
      lhs = self._fake_quant(lhs, lhs_how, quant_stat_name)

    return jnp.einsum(einsum_str, lhs, rhs)

  def conv_general_dilated(
      self,
      lhs: jax.Array,
      rhs: jax.Array,
      window_strides: Sequence[int],
      padding: str | Sequence[tuple[int, int]],
      dimension_numbers: jax.lax.ConvGeneralDilatedDimensionNumbers = None,
      precision: jax.lax.PrecisionLike = None,
      preferred_element_type: jax.typing.DTypeLike | None = None,
      **kwargs,
  ) -> jax.Array:
    """QAT conv_general_dilated."""
    rule, op_id = self._get_current_rule_and_op_id('conv_general_dilated')
    if rule is None or rule.weight_qtype is None:
      return jax.lax.conv_general_dilated(
          lhs,
          rhs,
          window_strides,
          padding,
          dimension_numbers=dimension_numbers,
          precision=precision,
          preferred_element_type=preferred_element_type,
          **kwargs,
      )
    if rule.tile_size:
      raise ValueError('subchannel is not supported for conv_general_dilated.')
    dimension_numbers = jax.lax.conv_dimension_numbers(
        lhs.shape, rhs.shape, dimension_numbers
    )

    # Prepare rhs.
    assert aux_data.get(rhs, 'weight_name', None)
    rhs_how = conv_general.get_how_to_quantize(
        dimension_numbers=dimension_numbers,
        for_lhs=False,
        qtype=rule.weight_qtype,
        calibration_method=rule.weight_calibration_method,
        batch_axes=(),
    )
    rhs = self._fake_quant(rhs, rhs_how)

    # Prepare lhs.
    if rule.act_qtype is not None:
      lhs_how = conv_general.get_how_to_quantize(
          dimension_numbers=dimension_numbers,
          for_lhs=True,
          qtype=rule.act_qtype,
          calibration_method=rule.act_calibration_method,
          batch_axes=rule.act_batch_axes if rule.act_static_scale else (),
      )
      quant_stat_name = op_id + '_lhs' if rule.act_static_scale else None
      lhs = self._fake_quant(lhs, lhs_how, quant_stat_name)

    return jax.lax.conv_general_dilated(
        lhs,
        rhs,
        window_strides,
        padding,
        dimension_numbers=dimension_numbers,
        **kwargs,
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
        'jax.lax.conv_general_dilated': self.conv_general_dilated,
        'jax.lax.dot_general': self.dot_general,
        'jax.numpy.einsum': self.einsum,
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

  def _fake_quant(
      self,
      array: jax.Array,
      how: qarray.HowToQuantize,
      quant_stat_name: str | None = None,
  ) -> jax.Array:
    """Apply fake quantization to array.

    This function can be used on both activations and weights. Gradient will be
    passed through.

    Args:
      array: The array to quantize.
      how: How to quantize the array.
      quant_stat_name: The name for the quantization statistics. If set, the
        quantization statistics will be collected and the scale will be computed
        from the statistics.

    Returns:
      The fake quantized array.
    """
    # TransposedQArray cannot be dequantized.
    how = dataclasses.replace(how, scale_transpose=None)

    calibration = qarray.calibrate(array, how)

    # Check and apply the static calibration asscociated with the array.
    static_calibration = aux_data.get(array, 'static_calibration', None)
    if static_calibration is not None:
      calibration = jax.tree.map(jnp.full_like, calibration, static_calibration)

    if quant_stat_name is not None:
      calibration = self._collect_quant_stat(
          quant_stat_name, calibration, static_calibration is not None
      )
    scale, zero_point = qarray.compute_scale_zero_point(calibration, how.qtype)
    q_array = qarray.quantize_with_scale_zero_point(
        array, how, scale, zero_point
    )
    return array + jax.lax.stop_gradient(qarray.dequantize(q_array) - array)

  def _collect_quant_stat(
      self,
      name: str,
      calibration: averaging.Calibration,
      calibration_is_static: bool,
  ) -> averaging.Calibration:
    """Collects the quantization statistics variables."""
    aggregator = averaging.SimpleMovingAverage()
    quant_stat = flax_util.get_or_create_variable(
        'quant_stats', name, lambda: aggregator.init(calibration)
    )

    if flax_util.should_update_quant_stats():
      if calibration_is_static:  # don't accumulate static calibration.
        quant_stat.value = aggregator.init(calibration)
      quant_stat.value = aggregator.update(quant_stat.value, calibration)

    return aggregator.get_calibration(quant_stat.value, calibration)
