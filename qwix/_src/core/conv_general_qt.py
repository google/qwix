# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Quantized jax.lax.conv_general_dilated with quantized backpropagation support."""

from collections.abc import Sequence
import dataclasses
import functools
import operator
from typing import Any, Callable

import jax
import numpy as np
from qwix._src import interception
from qwix._src.core import conv_general
from qwix._src.core import numerics
from qwix._src.core import qarray


@dataclasses.dataclass(slots=True, frozen=True, kw_only=True)
class ConvGeneralQtConfig:
  """Configuration for conv_general_qt."""

  # Forward pass.
  lhs_qtype: jax.typing.DTypeLike | None = None
  rhs_qtype: jax.typing.DTypeLike | None = None
  lhs_calibration_method: str = 'absmax'
  rhs_calibration_method: str = 'absmax'
  lhs_collect_quant_stat: Callable[[Any], Any] | None = None
  rhs_collect_quant_stat: Callable[[Any], Any] | None = None

  # Backward pass.
  dlhs_grad_qtype: jax.typing.DTypeLike | None = None
  dlhs_grad_calibration_method: str = 'absmax'
  drhs_grad_qtype: jax.typing.DTypeLike | None = None
  drhs_grad_calibration_method: str = 'absmax'

  # Misc.
  disable_channelwise_axes: bool = False


# Swaps the first two dimension indices of a specification.
# e.g., for an (N, C, H, W) spec, returns a (C, N, H, W) spec.
_conv_spec_transpose = lambda spec: (spec[1], spec[0]) + tuple(spec[2:])

# Extracts the spatial dimension indices from a specification.
# e.g., for an (N, C, H, W) spec, returns the (H, W) spec.
_conv_sdims = lambda spec: spec[2:]


def _compute_dilated_shape(
    shape: Sequence[int], dilation: Sequence[int]
) -> tuple[int, ...]:
  """Computes the shape of an array after dilation."""
  if not dilation or all(d == 1 for d in dilation):
    return tuple(shape)
  return tuple(
      s + (s - 1) * (d - 1) if s > 0 else 0 for s, d in zip(shape, dilation)
  )


def _conv_general_vjp_lhs_padding(
    in_shape,
    window_dimensions,
    window_strides,
    out_shape,
    padding,
    lhs_dilation,
    rhs_dilation,
) -> list[tuple[int, int]]:
  """Computes padding for the VJP wrt the LHS of a convolution."""
  lhs_dilated_shape = _compute_dilated_shape(in_shape, lhs_dilation)
  rhs_dilated_shape = _compute_dilated_shape(window_dimensions, rhs_dilation)
  out_dilated_shape = _compute_dilated_shape(out_shape, window_strides)
  pad_before = np.subtract(rhs_dilated_shape, [lo for lo, _ in padding]) - 1
  pad_after = (
      np.add(lhs_dilated_shape, rhs_dilated_shape)
      - 1
      - out_dilated_shape
      - pad_before
  )
  return list(zip(pad_before, pad_after))


def _conv_general_vjp_rhs_padding(
    in_shape,
    window_dimensions,
    window_strides,
    out_shape,
    padding,
    lhs_dilation,
    rhs_dilation,
):
  """Computes padding for the VJP wrt the RHS of a convolution."""
  if not in_shape:  # 0D conv
    return []
  lhs_dilated_shape = _compute_dilated_shape(in_shape, lhs_dilation)
  rhs_dilated_shape = _compute_dilated_shape(window_dimensions, rhs_dilation)
  out_dilated_shape = _compute_dilated_shape(out_shape, window_strides)
  pads_lo, _ = zip(*padding)
  pads_from_lhs = map(operator.sub, out_dilated_shape, lhs_dilated_shape)
  pads_from_rhs = tuple(
      rd - pd - 1 for rd, pd in zip(rhs_dilated_shape, pads_lo)
  )
  pads_hi = tuple(map(operator.add, pads_from_lhs, pads_from_rhs))
  return list(zip(pads_lo, pads_hi))


def _apply_fwd_scale_to_g(
    scale: jax.Array, g: jax.Array, g_axis: int
) -> jax.Array:
  """Applies the scale from the forward pass to the backward gradient.

  Since the scale can only have channelwise axes on one dimension, we only need
  to know which axis in g this scale applies to.

  Args:
    scale: The scale from the forward pass. Only 1 axis can be channelwise.
    g: The backward gradient.
    g_axis: The axis in g that the scale applies to.

  Returns:
    The backward gradient with the scale applied.
  """
  scale = scale.reshape([-1 if a == g_axis else 1 for a in range(g.ndim)])
  return g * scale


@interception.disable_interceptions
def conv_general_qt_fwd(
    lhs: jax.Array,
    rhs: jax.Array,
    config: ConvGeneralQtConfig,
    window_strides: Sequence[int],
    padding: str | Sequence[tuple[int, int]],
    lhs_dilation: Sequence[int] | None,
    rhs_dilation: Sequence[int] | None,
    dimension_numbers: jax.lax.ConvDimensionNumbers | None,
    feature_group_count: int,
    batch_group_count: int,
) -> tuple[jax.Array, tuple[qarray.MaybeQArray, qarray.MaybeQArray]]:
  """Forward pass for conv_general_qt custom VJP."""
  dnums = jax.lax.conv_dimension_numbers(
      lhs.shape, rhs.shape, dimension_numbers
  )

  def _quantize_operand(
      operand: jax.Array, *, for_lhs: bool
  ) -> qarray.MaybeQArray:
    """Quantizes a single operand for the forward pass if configured to do so."""
    qtype = config.lhs_qtype if for_lhs else config.rhs_qtype
    if not (qtype and numerics.should_quantize(operand.dtype)):
      return operand

    if for_lhs:
      calibration_method = config.lhs_calibration_method
      collect_quant_stat = config.lhs_collect_quant_stat
    else:
      calibration_method = config.rhs_calibration_method
      collect_quant_stat = config.rhs_collect_quant_stat

    how = conv_general.get_how_to_quantize(
        dimension_numbers=dnums,
        for_lhs=for_lhs,
        qtype=qtype,
        calibration_method=calibration_method,
    )
    if config.disable_channelwise_axes:
      how = dataclasses.replace(how, channelwise_axes=[])

    calibration = qarray.calibrate(operand, how)
    if collect_quant_stat:
      calibration = collect_quant_stat(calibration)
    scale, zero_point = qarray.compute_scale_zero_point(calibration, qtype)
    return qarray.quantize_with_scale_zero_point(
        operand, qtype, scale, zero_point
    )

  lhs = _quantize_operand(lhs, for_lhs=True)
  rhs = _quantize_operand(rhs, for_lhs=False)

  primal_out = conv_general.conv_general_dilated(
      lhs,
      rhs,
      window_strides,
      padding,
      lhs_dilation,
      rhs_dilation,
      dnums,
      feature_group_count,
      batch_group_count,
  )
  residuals = (lhs, rhs)

  return primal_out, residuals


def conv_general_qt_bwd(
    config: ConvGeneralQtConfig,
    window_strides: Sequence[int],
    padding: str | Sequence[tuple[int, int]],
    lhs_dilation: Sequence[int] | None,
    rhs_dilation: Sequence[int] | None,
    dimension_numbers: jax.lax.ConvDimensionNumbers | None,
    feature_group_count: int,
    batch_group_count: int,
    res: tuple[qarray.MaybeQArray, qarray.MaybeQArray],
    g: jax.Array,
):
  """Backward pass for conv_general_qt custom VJP."""
  lhs, rhs = res

  dnums = jax.lax.conv_dimension_numbers(
      lhs.shape, rhs.shape, dimension_numbers
  )
  if lhs_dilation is None:
    lhs_dilation = (1,) * (lhs.ndim - 2)
  if rhs_dilation is None:
    rhs_dilation = (1,) * (rhs.ndim - 2)

  lhs_sdims_indices, rhs_sdims_indices, out_sdims_indices = map(
      _conv_sdims, dnums
  )
  lhs_sdims_shape = tuple(np.take(lhs.shape, lhs_sdims_indices))
  rhs_sdims_shape = tuple(np.take(rhs.shape, rhs_sdims_indices))
  g_sdims_shape = tuple(np.take(g.shape, out_sdims_indices))

  if isinstance(padding, str):
    effective_rhs_sdims = _compute_dilated_shape(rhs_sdims_shape, rhs_dilation)
    padding = jax.lax.padtype_to_pads(
        lhs_sdims_shape, effective_rhs_sdims, window_strides, padding
    )

  # dlhs
  dlhs_dnums = jax.lax.ConvDimensionNumbers(
      lhs_spec=dnums.out_spec,
      rhs_spec=_conv_spec_transpose(dnums.rhs_spec),
      out_spec=dnums.lhs_spec,
  )
  dlhs_padding = _conv_general_vjp_lhs_padding(
      lhs_sdims_shape,
      rhs_sdims_shape,
      window_strides,
      g_sdims_shape,
      padding,
      lhs_dilation,
      rhs_dilation,
  )

  dlhs_g = g
  if config.dlhs_grad_qtype:
    if isinstance(rhs, qarray.QArray):
      # Apply rhs.scale to dlhs_g.
      dlhs_g = _apply_fwd_scale_to_g(rhs.scale, dlhs_g, dnums.out_spec[1])
      rhs = rhs.qvalue
    how = conv_general.get_how_to_quantize(
        dimension_numbers=dlhs_dnums,
        for_lhs=True,
        qtype=config.dlhs_grad_qtype,
        calibration_method=config.dlhs_grad_calibration_method,
    )
    if config.disable_channelwise_axes:
      how = dataclasses.replace(how, channelwise_axes=[])
    dlhs_g = qarray.quantize(dlhs_g, how)

  rhs = jax.tree.map(lambda x: jax.lax.rev(x, rhs_sdims_indices), rhs)

  dlhs = conv_general.conv_general_dilated(
      lhs=dlhs_g,
      rhs=rhs,
      window_strides=lhs_dilation,
      padding=dlhs_padding,
      lhs_dilation=window_strides,
      rhs_dilation=rhs_dilation,
      dimension_numbers=dlhs_dnums,
      feature_group_count=feature_group_count,
      batch_group_count=batch_group_count,
  )

  # drhs
  drhs_dnums = jax.lax.ConvDimensionNumbers(
      lhs_spec=_conv_spec_transpose(dnums.lhs_spec),
      rhs_spec=_conv_spec_transpose(dnums.out_spec),
      out_spec=_conv_spec_transpose(dnums.rhs_spec),
  )
  drhs_padding = _conv_general_vjp_rhs_padding(
      lhs_sdims_shape,
      rhs_sdims_shape,
      window_strides,
      g_sdims_shape,
      padding,
      lhs_dilation,
      rhs_dilation,
  )

  drhs_g = g
  if config.drhs_grad_qtype:
    if isinstance(lhs, qarray.QArray):
      # Apply lhs.scale to drhs_g.
      drhs_g = _apply_fwd_scale_to_g(lhs.scale, drhs_g, dnums.out_spec[0])
      lhs = lhs.qvalue
    how = conv_general.get_how_to_quantize(
        dimension_numbers=drhs_dnums,
        for_lhs=False,
        qtype=config.drhs_grad_qtype,
        calibration_method=config.drhs_grad_calibration_method,
    )
    if config.disable_channelwise_axes:
      how = dataclasses.replace(how, channelwise_axes=[])
    drhs_g = qarray.quantize(drhs_g, how)

  drhs = conv_general.conv_general_dilated(
      lhs=lhs,
      rhs=drhs_g,
      window_strides=rhs_dilation,
      padding=drhs_padding,
      lhs_dilation=lhs_dilation,
      rhs_dilation=window_strides,
      dimension_numbers=drhs_dnums,
      feature_group_count=feature_group_count,
      batch_group_count=batch_group_count,
  )

  return dlhs, drhs


@functools.partial(jax.custom_vjp, nondiff_argnums=(2, 3, 4, 5, 6, 7, 8, 9))
def conv_general_qt(
    lhs: jax.Array,
    rhs: jax.Array,
    config: ConvGeneralQtConfig,
    window_strides: Sequence[int],
    padding: str | Sequence[tuple[int, int]],
    lhs_dilation: Sequence[int] | None = None,
    rhs_dilation: Sequence[int] | None = None,
    dimension_numbers: jax.lax.ConvDimensionNumbers | None = None,
    feature_group_count: int = 1,
    batch_group_count: int = 1,
) -> jax.Array:
  """Quantized conv_general using a simple, hashable config dataclass."""
  result, _ = conv_general_qt_fwd(
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
  return result


conv_general_qt.defvjp(conv_general_qt_fwd, conv_general_qt_bwd)
