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
from typing import Any

import flax
import jax
import numpy as onp
from qwix.core import dot_general
from qwix.core import qarray

dataclass = flax.struct.dataclass
field = flax.struct.field


@dataclass(slots=True, frozen=True, kw_only=True)
class HowToQuantizeQt:
  """A dataclass for holding quantization config."""

  qtype: jax.typing.DTypeLike = field(pytree_node=False)
  calibration_method: str = field(pytree_node=False)
  tile_size: int | None


@dataclass(slots=True, frozen=True, kw_only=True)
class DotGeneralResidual:
  """A dataclass for holding residual data for dot_general_qt custom VJP."""

  fwd_lhs: jax.Array
  fwd_rhs: jax.Array
  fwd_dimension_numbers: jax.lax.DotDimensionNumbers = field(pytree_node=False)
  back_how: HowToQuantizeQt | None = field(pytree_node=False)
  out_sharding: jax.sharding.NamedSharding | None = field(pytree_node=False)


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

  x_ca_sorted_by_y = tuple(onp.take(x_ca, onp.argsort(y_ca)))
  out_transpose_axes = tuple(onp.argsort(tuple(x_ba) + x_ra + x_ca_sorted_by_y))
  return dims, out_transpose_axes


def dot_general_qt_fwd(
    lhs: jax.Array,
    rhs: jax.Array,
    dimension_numbers: jax.lax.DotDimensionNumbers,
    lhs_how: HowToQuantizeQt | None,
    rhs_how: HowToQuantizeQt | None,
    back_how: HowToQuantizeQt | None,
    out_sharding: jax.sharding.NamedSharding | None,
):
  """Forward pass for dot_general_qt custom VJP."""
  lhs_how_full = None
  if lhs_how:
    lhs_how_full = dot_general.get_how_to_quantize(
        dimension_numbers=dimension_numbers,
        ndims=(lhs.ndim, rhs.ndim),
        for_lhs=True,
        qtype=lhs_how.qtype,
        tile_size=lhs_how.tile_size,
        calibration_method=lhs_how.calibration_method,
        batch_axes=(),
    )
  rhs_how_full = None
  if rhs_how:
    rhs_how_full = dot_general.get_how_to_quantize(
        dimension_numbers=dimension_numbers,
        ndims=(lhs.ndim, rhs.ndim),
        for_lhs=False,
        qtype=rhs_how.qtype,
        tile_size=rhs_how.tile_size,
        calibration_method=rhs_how.calibration_method,
        batch_axes=(),
    )

  lhs_quantized = qarray.quantize(lhs, lhs_how_full) if lhs_how_full else lhs
  rhs_quantized = qarray.quantize(rhs, rhs_how_full) if rhs_how_full else rhs
  primal_out = dot_general.dot_general(
      lhs_quantized, rhs_quantized, dimension_numbers, out_sharding
  )

  res = DotGeneralResidual(
      fwd_lhs=lhs,
      fwd_rhs=rhs,
      fwd_dimension_numbers=dimension_numbers,
      back_how=back_how,
      out_sharding=out_sharding,
  )
  return (primal_out, res)


def dot_general_qt_bwd(res: DotGeneralResidual, g: Any):
  """Backward pass for dot_general_qt custom VJP."""
  fwd_lhs, fwd_rhs, back_how, fwd_dnums, out_sharding = (
      res.fwd_lhs,
      res.fwd_rhs,
      res.back_how,
      res.fwd_dimension_numbers,
      res.out_sharding,
  )

  def _compute_gradient_for_operand(
      other_fwd_operand: jax.Array,
      y_is_fwd_lhs: bool,
  ):
    """Compute dot_general for gradient and other_fwd_operand."""
    grad_dnums, transpose_axes = (
        _update_dimension_numbers_for_backward_internal(
            fwd_dnums, y_is_fwd_lhs, g.ndim, other_fwd_operand.ndim
        )
    )

    bwd_lhs_input, bwd_rhs_input = g, other_fwd_operand
    if back_how:
      back_lhs_how = dot_general.get_how_to_quantize(
          dimension_numbers=grad_dnums,
          ndims=(g.ndim, other_fwd_operand.ndim),
          for_lhs=True,
          qtype=back_how.qtype,
          tile_size=back_how.tile_size,
          calibration_method=back_how.calibration_method,
          batch_axes=(),
      )
      back_rhs_how = dot_general.get_how_to_quantize(
          dimension_numbers=grad_dnums,
          ndims=(g.ndim, other_fwd_operand.ndim),
          for_lhs=False,
          qtype=back_how.qtype,
          tile_size=back_how.tile_size,
          calibration_method=back_how.calibration_method,
          batch_axes=(),
      )
      bwd_lhs_input = qarray.quantize(g, back_lhs_how)
      bwd_rhs_input = qarray.quantize(other_fwd_operand, back_rhs_how)

    grad_res = dot_general.dot_general(
        bwd_lhs_input, bwd_rhs_input, grad_dnums, out_sharding
    )
    return jax.lax.transpose(grad_res, transpose_axes)

  dlhs = _compute_gradient_for_operand(fwd_rhs, False)
  drhs = _compute_gradient_for_operand(fwd_lhs, True)

  return (dlhs, drhs, None, None, None, None, None)


@jax.custom_vjp
def dot_general_qt(
    lhs: jax.Array,
    rhs: jax.Array,
    dimension_numbers: jax.lax.DotDimensionNumbers,
    lhs_how: HowToQuantizeQt | None = None,
    rhs_how: HowToQuantizeQt | None = None,
    back_how: HowToQuantizeQt | None = None,
    out_sharding: jax.sharding.NamedSharding | None = None,
) -> jax.Array:
  """Quantized dot_general using a simple, hashable config dataclass."""
  result, _ = dot_general_qt_fwd(
      lhs, rhs, dimension_numbers, lhs_how, rhs_how, back_how, out_sharding
  )
  return result


dot_general_qt.defvjp(dot_general_qt_fwd, dot_general_qt_bwd)
