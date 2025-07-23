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

from collections.abc import Collection
import dataclasses
import functools

import jax
import numpy as np
from qwix.core import dot_general
from qwix.core import numerics
from qwix.core import qarray


@dataclasses.dataclass(slots=True, frozen=True, kw_only=True)
class DotGeneralQtConfig:
  """Configuration for dot_general_qt."""

  lhs_qtype: jax.typing.DTypeLike | None = None
  rhs_qtype: jax.typing.DTypeLike | None = None
  bwd_qtype: jax.typing.DTypeLike | None = None
  tile_size: int | None = None
  calibration_method: str = 'absmax'
  disable_channelwise_axes: bool = False
  use_original_residuals: bool = False


def _get_remaining_axes(
    ndim: int, contracting_axes: Collection[int], batch_axes: Collection[int]
) -> list[int]:
  all_axes = set(range(ndim))
  remaining_axes_set = all_axes - set(contracting_axes) - set(batch_axes)
  return sorted(list(remaining_axes_set))


def _ranges_like(*xs):
  start = 0
  for x in xs:
    yield tuple(range(start, start + len(x)))
    start += len(x)


def _update_dimension_numbers_for_backward_internal(
    fwd_dimension_numbers: jax.lax.DotDimensionNumbers,
    y_is_fwd_lhs: bool,
    gradient_rank: int,
    y_rank: int,
) -> tuple[jax.lax.DotDimensionNumbers, tuple[int, ...]]:
  """Generates a new dimension number for backward pass.

  Args:
    fwd_dimension_numbers: Dimension numbers from the forward pass.
    y_is_fwd_lhs: If True, the original forward LHS (`fwd_lhs`) is playing the
      role of `y` (computing dRHS = dot(g, fwd_lhs)). If False, `fwd_rhs` plays
      the role of `y` (computing dLHS = dot(g, fwd_rhs)).
    gradient_rank: Rank of the gradient `g` (the first operand).
    y_rank: Rank of the `other_fwd_operand` (the second operand).

  Returns:
    A tuple of (dimension numbers for gradient dot_general, transpose axes to be
    applied on the gradient dot_general's output to match the original
    argument's dimension).
  """
  (fwd_lhs_ca, fwd_rhs_ca), (fwd_lhs_ba, fwd_rhs_ba) = fwd_dimension_numbers
  if y_is_fwd_lhs:
    x_ca, y_ca = fwd_rhs_ca, fwd_lhs_ca
    x_ba, y_ba = fwd_rhs_ba, fwd_lhs_ba
  else:
    x_ca, y_ca = fwd_lhs_ca, fwd_rhs_ca
    x_ba, y_ba = fwd_lhs_ba, fwd_rhs_ba

  effective_gradient_rank = gradient_rank - y_rank + len(x_ba) + 2 * len(x_ca)
  x_ra = tuple(_get_remaining_axes(effective_gradient_rank, x_ca, x_ba))
  y_ra = tuple(_get_remaining_axes(y_rank, y_ca, y_ba))
  if y_is_fwd_lhs:
    g_ba, g_ca, _ = _ranges_like(x_ba, y_ra, x_ra)
  else:
    g_ba, _, g_ca = _ranges_like(x_ba, x_ra, y_ra)
  dims = ((g_ca, y_ra), (g_ba, y_ba))

  x_ca_sorted_by_y = tuple(np.take(x_ca, np.argsort(y_ca)))
  out_transpose_axes = tuple(np.argsort(tuple(x_ba) + x_ra + x_ca_sorted_by_y))
  return dims, out_transpose_axes


def dot_general_qt_fwd(
    lhs: jax.Array,
    rhs: jax.Array,
    dimension_numbers: jax.lax.DotDimensionNumbers,
    config: DotGeneralQtConfig,
):
  """Forward pass for dot_general_qt custom VJP."""
  ndims = (lhs.ndim, rhs.ndim)
  res_lhs, res_rhs = lhs, rhs

  # Quantize lhs and rhs.
  if config.lhs_qtype and numerics.should_quantize(lhs.dtype):
    lhs_how = dot_general.get_how_to_quantize(
        dimension_numbers=dimension_numbers,
        ndims=ndims,
        for_lhs=True,
        qtype=config.lhs_qtype,
        tile_size=config.tile_size,
        calibration_method=config.calibration_method,
        batch_axes=(),
    )
    if config.disable_channelwise_axes:
      lhs_how = dataclasses.replace(lhs_how, channelwise_axes=[])
    lhs = qarray.quantize(lhs, lhs_how)
    if not config.use_original_residuals:
      res_lhs = lhs

  if config.rhs_qtype and numerics.should_quantize(rhs.dtype):
    rhs_how = dot_general.get_how_to_quantize(
        dimension_numbers=dimension_numbers,
        ndims=ndims,
        for_lhs=False,
        qtype=config.rhs_qtype,
        tile_size=config.tile_size,
        calibration_method=config.calibration_method,
        batch_axes=(),
    )
    if config.disable_channelwise_axes:
      rhs_how = dataclasses.replace(rhs_how, channelwise_axes=[])
    rhs = qarray.quantize(rhs, rhs_how)
    if not config.use_original_residuals:
      res_rhs = rhs

  primal_out = dot_general.dot_general(lhs, rhs, dimension_numbers)
  return primal_out, (res_lhs, res_rhs)


def dot_general_qt_bwd(
    dimension_numbers: jax.lax.DotDimensionNumbers,
    config: DotGeneralQtConfig,
    res: tuple[qarray.MaybeQArray, qarray.MaybeQArray],
    g: jax.Array,
):
  """Backward pass for dot_general_qt custom VJP."""

  def _compute_gradient_for_operand(
      y: jax.Array,
      y_is_fwd_lhs: bool,
  ):
    """Compute dot_general for gradient and other_fwd_operand."""
    grad_dnums, transpose_axes = (
        _update_dimension_numbers_for_backward_internal(
            dimension_numbers, y_is_fwd_lhs, g.ndim, y.ndim
        )
    )
    if config.bwd_qtype:
      g_how = dot_general.get_how_to_quantize(
          dimension_numbers=grad_dnums,
          ndims=(g.ndim, y.ndim),
          for_lhs=True,
          qtype=config.bwd_qtype,
          tile_size=config.tile_size,
          calibration_method=config.calibration_method,
          batch_axes=(),
      )
      if config.disable_channelwise_axes:
        g_how = dataclasses.replace(g_how, channelwise_axes=[])
      q_g = (
          qarray.quantize(g, g_how) if numerics.should_quantize(g.dtype) else g
      )
      y_how = dot_general.get_how_to_quantize(
          dimension_numbers=grad_dnums,
          ndims=(g.ndim, y.ndim),
          for_lhs=False,
          qtype=config.bwd_qtype,
          tile_size=config.tile_size,
          calibration_method=config.calibration_method,
          batch_axes=(),
      )
      if config.disable_channelwise_axes:
        y_how = dataclasses.replace(y_how, channelwise_axes=[])
      q_y = (
          qarray.quantize(y, y_how) if numerics.should_quantize(y.dtype) else y
      )
      grad_res = dot_general.dot_general(q_g, q_y, grad_dnums)
    else:
      grad_res = jax.lax.dot_general(g, y, grad_dnums)
    return jax.lax.transpose(grad_res, transpose_axes)

  lhs, rhs = res
  # lhs/rhs are quantized with different channelwise axes when in the forward
  # pass, so they need to be dequantized first.
  lhs = qarray.dequantize(lhs) if isinstance(lhs, qarray.QArray) else lhs
  rhs = qarray.dequantize(rhs) if isinstance(rhs, qarray.QArray) else rhs

  dlhs = _compute_gradient_for_operand(rhs, False)
  drhs = _compute_gradient_for_operand(lhs, True)

  return dlhs, drhs


@functools.partial(jax.custom_vjp, nondiff_argnums=(2, 3))
def dot_general_qt(
    lhs: jax.Array,
    rhs: jax.Array,
    dimension_numbers: jax.lax.DotDimensionNumbers,
    config: DotGeneralQtConfig,
) -> jax.Array:
  """Quantized dot_general using a simple, hashable config dataclass."""
  result, _ = dot_general_qt_fwd(lhs, rhs, dimension_numbers, config)
  return result


dot_general_qt.defvjp(dot_general_qt_fwd, dot_general_qt_bwd)
