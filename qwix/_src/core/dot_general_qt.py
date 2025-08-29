# Copyright 2025 Google LLC
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
"""Quantized jax.lax.dot_general with quantized backpropagation support."""

import dataclasses
import functools
from typing import Any, Callable

import jax
import numpy as np
from qwix._src import interception
from qwix._src.core import dot_general
from qwix._src.core import numerics
from qwix._src.core import qarray


@dataclasses.dataclass(slots=True, frozen=True, kw_only=True)
class DotGeneralQtConfig:
  """Configuration for dot_general_qt."""

  # Forward pass.
  lhs_qtype: jax.typing.DTypeLike | None = None
  rhs_qtype: jax.typing.DTypeLike | None = None
  tile_size: int | float | None = None
  lhs_calibration_method: str = 'absmax'
  rhs_calibration_method: str = 'absmax'
  lhs_collect_quant_stat: Callable[[Any], Any] | None = None
  rhs_collect_quant_stat: Callable[[Any], Any] | None = None

  # Backward pass (dlhs).
  dlhs_lhs_qtype: jax.typing.DTypeLike | None = None  # incoming gradient
  dlhs_rhs_qtype: jax.typing.DTypeLike | None = None  # residual rhs
  dlhs_tile_size: int | float | None = None
  dlhs_lhs_calibration_method: str = 'absmax'
  dlhs_rhs_calibration_method: str = 'absmax'

  # Backward pass (drhs).
  drhs_lhs_qtype: jax.typing.DTypeLike | None = None  # incoming gradient
  drhs_rhs_qtype: jax.typing.DTypeLike | None = None  # residual lhs
  drhs_tile_size: int | float | None = None
  drhs_lhs_calibration_method: str = 'absmax'
  drhs_rhs_calibration_method: str = 'absmax'

  # Misc.
  disable_channelwise_axes: bool = False
  bwd_use_original_residuals: bool = False  # what to use as residuals


def _ranges_like(*xs):
  start = 0
  for x in xs:
    yield tuple(range(start, start + len(x)))
    start += len(x)


def _update_dimension_numbers_for_backward(
    fwd_dimension_numbers: jax.lax.DotDimensionNumbers,
    fwd_ndims: tuple[int, int],
    *,
    for_dlhs: bool,
) -> tuple[jax.lax.DotDimensionNumbers, tuple[int, ...]]:
  """Generates a new dimension number for backward pass.

  The dimension number is used to calculate dlhs = dot(g, rhs), if for_dlhs is
  True, or drhs = dot(g, lhs), otherwise.

  Args:
    fwd_dimension_numbers: Dimension numbers from the forward pass.
    fwd_ndims: Number of dimensions of the operands in the forward pass.
    for_dlhs: Whether the backward pass is for dlhs or drhs.

  Returns:
    A tuple of (dimension numbers for backward pass, output transpose to be
    applied to the result of the backward dot to match the original dimensions).
  """
  # We use the following conventions:
  #   g: incoming gradient.
  #   x: output gradient, dlhs if for_dlhs, drhs otherwise.
  #   y: residual, rhs if for_dlhs, lhs otherwise.
  if for_dlhs:
    (x_ca, y_ca), (x_ba, y_ba) = fwd_dimension_numbers
    x_ndim, y_ndim = fwd_ndims
  else:
    (y_ca, x_ca), (y_ba, x_ba) = fwd_dimension_numbers
    y_ndim, x_ndim = fwd_ndims

  x_ra = tuple(sorted(set(range(x_ndim)) - set(x_ca) - set(x_ba)))
  y_ra = tuple(sorted(set(range(y_ndim)) - set(y_ca) - set(y_ba)))

  if for_dlhs:
    g_ba, _, g_ca = _ranges_like(x_ba, x_ra, y_ra)
  else:
    g_ba, g_ca, _ = _ranges_like(x_ba, y_ra, x_ra)
  dnums = ((g_ca, y_ra), (g_ba, y_ba))

  x_ca_sorted_by_y = tuple(np.take(x_ca, np.argsort(y_ca)))
  out_transpose_axes = tuple(np.argsort(tuple(x_ba) + x_ra + x_ca_sorted_by_y))
  return dnums, out_transpose_axes


# See test_scan_custom_vjp in interception_test.py for why we need to manually
# disable interceptions for dot_general_qt_fwd.
@interception.disable_interceptions
def dot_general_qt_fwd(
    lhs: jax.Array,
    rhs: jax.Array,
    dimension_numbers: jax.lax.DotDimensionNumbers,
    config: DotGeneralQtConfig,
):
  """Forward pass for dot_general_qt custom VJP."""
  ndims = (lhs.ndim, rhs.ndim)

  def _quantize_operand(
      operand: jax.Array, is_lhs: bool
  ) -> tuple[qarray.MaybeQArray, qarray.MaybeQArray]:
    """Quantizes a single operand for the forward pass if configured to do so.

    Args:
      operand: The array to quantize (either LHS or RHS).
      is_lhs: A boolean indicating if the operand is the LHS.

    Returns:
      A tuple of (quantized_operand, quantized_residual).
    """
    if is_lhs:
      qtype = config.lhs_qtype
      calibration_method = config.lhs_calibration_method
      collect_quant_stat = config.lhs_collect_quant_stat
      # lhs is used as the rhs in drhs.
      bwd_dnums, _ = _update_dimension_numbers_for_backward(
          dimension_numbers, ndims, for_dlhs=False
      )
      bwd_qtype = config.drhs_rhs_qtype
      bwd_tile_size = config.drhs_tile_size
      bwd_calibration_method = config.drhs_rhs_calibration_method
    else:
      qtype = config.rhs_qtype
      calibration_method = config.rhs_calibration_method
      collect_quant_stat = config.rhs_collect_quant_stat
      # rhs is used as the rhs in dlhs.
      bwd_dnums, _ = _update_dimension_numbers_for_backward(
          dimension_numbers, ndims, for_dlhs=True
      )
      bwd_qtype = config.dlhs_rhs_qtype
      bwd_tile_size = config.dlhs_tile_size
      bwd_calibration_method = config.dlhs_rhs_calibration_method

    if not (qtype and numerics.should_quantize(operand.dtype)):
      return operand, operand

    how = dot_general.get_how_to_quantize(
        dimension_numbers=dimension_numbers,
        ndims=ndims,
        for_lhs=is_lhs,
        qtype=qtype,
        tile_size=config.tile_size,
        calibration_method=calibration_method,
    )
    if config.disable_channelwise_axes:
      how = dataclasses.replace(how, channelwise_axes=[])
    bwd_how = None
    if bwd_qtype:
      bwd_how = dot_general.get_how_to_quantize(
          dimension_numbers=bwd_dnums,
          ndims=(0, operand.ndim),  # g_ndim doesn't matter.
          for_lhs=False,  # residual is always the rhs.
          qtype=bwd_qtype,
          tile_size=bwd_tile_size,
          calibration_method=bwd_calibration_method,
      )
      if config.disable_channelwise_axes:
        bwd_how = dataclasses.replace(bwd_how, channelwise_axes=[])

    calibration = qarray.calibrate(operand, how)
    if collect_quant_stat:
      calibration = collect_quant_stat(calibration)
    scale, zero_point = qarray.compute_scale_zero_point(calibration, qtype)
    q_operand = qarray.quantize_with_scale_zero_point(
        operand, how.qtype, scale, zero_point
    )

    if bwd_how is None and config.bwd_use_original_residuals:
      res_operand = operand
    elif bwd_how is None or bwd_how == how:
      # Delays dequantization if bwd_how is None, or avoids double
      # quantization if bwd_how == how.
      res_operand = q_operand
    elif config.bwd_use_original_residuals:
      res_operand = qarray.quantize(operand, bwd_how)
    else:
      res_operand = qarray.quantize(qarray.dequantize(q_operand), bwd_how)
    return q_operand, res_operand

  lhs, res_lhs = _quantize_operand(lhs, is_lhs=True)
  rhs, res_rhs = _quantize_operand(rhs, is_lhs=False)

  primal_out = dot_general.dot_general(lhs, rhs, dimension_numbers)
  return primal_out, (res_lhs, res_rhs)


def dot_general_qt_bwd(
    fwd_dimension_numbers: jax.lax.DotDimensionNumbers,
    config: DotGeneralQtConfig,
    residuals: tuple[qarray.MaybeQArray, qarray.MaybeQArray],
    g: jax.Array,
):
  """Backward pass for dot_general_qt custom VJP."""
  lhs, rhs = residuals
  fwd_ndims = (lhs.ndim, rhs.ndim)

  def _compute_gradient_for_operand(
      g: jax.Array, y: qarray.MaybeQArray, *, for_dlhs: bool
  ):
    """Compute dot_general for gradient and other_fwd_operand."""
    bwd_dnums, transpose_axes = _update_dimension_numbers_for_backward(
        fwd_dimension_numbers, fwd_ndims, for_dlhs=for_dlhs
    )
    if for_dlhs:
      g_qtype = config.dlhs_lhs_qtype
      g_tile_size = config.dlhs_tile_size
      g_calibration_method = config.dlhs_lhs_calibration_method
      y_qtype = config.dlhs_rhs_qtype
    else:
      g_qtype = config.drhs_lhs_qtype
      g_tile_size = config.drhs_tile_size
      g_calibration_method = config.drhs_lhs_calibration_method
      y_qtype = config.drhs_rhs_qtype

    if g_qtype and numerics.should_quantize(g.dtype):
      g_how = dot_general.get_how_to_quantize(
          dimension_numbers=bwd_dnums,
          ndims=(g.ndim, y.ndim),
          for_lhs=True,
          qtype=g_qtype,
          tile_size=g_tile_size,
          calibration_method=g_calibration_method,
      )
      if config.disable_channelwise_axes:
        g_how = dataclasses.replace(g_how, channelwise_axes=[])
      g = qarray.quantize(g, g_how)

    if y_qtype is not None:
      assert isinstance(y, qarray.QArray), 'y should be quantized in fwd pass.'
    elif isinstance(y, qarray.QArray):
      y = qarray.dequantize(y)

    grad_res = dot_general.dot_general(g, y, bwd_dnums)
    return jax.lax.transpose(grad_res, transpose_axes)

  dlhs = _compute_gradient_for_operand(g, rhs, for_dlhs=True)
  drhs = _compute_gradient_for_operand(g, lhs, for_dlhs=False)

  return dlhs, drhs


@functools.partial(jax.custom_vjp, nondiff_argnums=(2, 3))
def dot_general_qt(
    lhs: jax.Array,
    rhs: jax.Array,
    dimension_numbers: jax.lax.DotDimensionNumbers,
    config: DotGeneralQtConfig,
) -> jax.Array:
  """Quantized dot_general with backpropagation support."""
  result, _ = dot_general_qt_fwd(lhs, rhs, dimension_numbers, config)
  return result


dot_general_qt.defvjp(dot_general_qt_fwd, dot_general_qt_bwd)
