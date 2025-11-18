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
from jax import numpy as jnp
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
  dlhs_grad_qtype: jax.typing.DTypeLike | None = None  # incoming gradient
  dlhs_grad_calibration_method: str = 'absmax'
  dlhs_tile_size: int | float | None = None
  dlhs_stochastic_rounding_noise_fn: numerics.NoiseFn | None = None

  # Backward pass (drhs).
  drhs_grad_qtype: jax.typing.DTypeLike | None = None  # incoming gradient
  drhs_grad_calibration_method: str = 'absmax'
  drhs_tile_size: int | float | None = None
  drhs_stochastic_rounding_noise_fn: numerics.NoiseFn | None = None

  # Misc.
  disable_channelwise_axes: bool = False


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


def _apply_rhs_scale_to_lhs(lhs, rhs_scale, dnums):
  """Applies the rhs_scale to lhs."""
  (lhs_ca, rhs_ca), (lhs_ba, rhs_ba) = dnums
  lhs_axes_to_rhs = dict(zip(lhs_ca, rhs_ca)) | dict(zip(lhs_ba, rhs_ba))
  lhs_scale = qarray.transpose_array(
      rhs_scale, [lhs_axes_to_rhs.get(a) for a in range(lhs.ndim)]
  )
  return qarray.call_with_generic_broadcast(jnp.multiply, lhs, lhs_scale)


# See test_scan_custom_vjp in interception_test.py for why we need to manually
# disable interceptions for dot_general_qt_fwd.
@interception.disable_interceptions
def dot_general_qt_fwd(
    lhs: jax.Array,
    rhs: jax.Array,
    lhs_calibration: dict[str, jax.Array] | None,
    rhs_calibration: dict[str, jax.Array] | None,
    dimension_numbers: jax.lax.DotDimensionNumbers,
    config: DotGeneralQtConfig,
):
  """Forward pass for dot_general_qt custom VJP."""
  lhs_in, rhs_in = lhs, rhs
  if lhs_calibration is not None:
    scale, zero_point = qarray.compute_scale_zero_point(
        lhs_calibration, config.lhs_qtype
    )
    lhs = qarray.quantize_with_scale_zero_point(
        lhs, config.lhs_qtype, scale, zero_point
    )
  if rhs_calibration is not None:
    scale, zero_point = qarray.compute_scale_zero_point(
        rhs_calibration, config.rhs_qtype
    )
    rhs = qarray.quantize_with_scale_zero_point(
        rhs, config.rhs_qtype, scale, zero_point
    )
  residuals = (lhs_in, rhs_in, lhs, rhs, lhs_calibration, rhs_calibration)
  return dot_general.dot_general(lhs, rhs, dimension_numbers), residuals


def dot_general_qt_bwd(
    fwd_dimension_numbers: jax.lax.DotDimensionNumbers,
    config: DotGeneralQtConfig,
    residuals: tuple[
        jax.Array,
        jax.Array,
        qarray.MaybeQArray,
        qarray.MaybeQArray,
        dict[str, jax.Array] | None,
        dict[str, jax.Array] | None,
    ],
    g: jax.Array,
):
  """Backward pass for dot_general_qt custom VJP."""
  lhs_in, rhs_in, lhs, rhs, lhs_calibration, rhs_calibration = residuals

  def _compute_gradient_for_operand(
      g: jax.Array, y: qarray.MaybeQArray, *, for_dlhs: bool
  ):
    """Compute dot_general for gradient and other_fwd_operand."""
    bwd_dnums, transpose_axes = _update_dimension_numbers_for_backward(
        fwd_dimension_numbers, (lhs.ndim, rhs.ndim), for_dlhs=for_dlhs
    )
    if for_dlhs:
      g_qtype = config.dlhs_grad_qtype
      g_tile_size = config.dlhs_tile_size
      g_calibration_method = config.dlhs_grad_calibration_method
      g_noise_fn = config.dlhs_stochastic_rounding_noise_fn
    else:
      g_qtype = config.drhs_grad_qtype
      g_tile_size = config.drhs_tile_size
      g_calibration_method = config.drhs_grad_calibration_method
      g_noise_fn = config.drhs_stochastic_rounding_noise_fn

    if g_qtype and numerics.should_quantize(g.dtype):
      if isinstance(y, qarray.QArray) and not qarray.get_tiled_axes(y):
        # Apply the scale of y to g, this trick avoids requantizing y because
        # the y from fwd pass has different channelwise_axes.
        assert y.zero_point is None and y.qtype == y.qvalue.dtype
        g = _apply_rhs_scale_to_lhs(g, y.scale, bwd_dnums)
        y = y.qvalue

      g_how = dot_general.get_how_to_quantize(
          dimension_numbers=bwd_dnums,
          ndims=(g.ndim, y.ndim),
          for_lhs=True,
          qtype=g_qtype,
          tile_size=g_tile_size,
          calibration_method=g_calibration_method,
          noise_fn=g_noise_fn,
      )
      if config.disable_channelwise_axes:
        g_how = dataclasses.replace(g_how, channelwise_axes=[])

      g = qarray.quantize(g, g_how)

    grad_res = dot_general.dot_general(g, y, bwd_dnums)
    return jax.lax.transpose(grad_res, transpose_axes)

  dlhs = _compute_gradient_for_operand(g, rhs, for_dlhs=True)
  drhs = _compute_gradient_for_operand(g, lhs, for_dlhs=False)

  if lhs_calibration is not None:
    dlhs = qarray.clip_gradient_to_calibration(
        dlhs, lhs_in, lhs_calibration, config.lhs_calibration_method
    )
  if rhs_calibration is not None:
    drhs = qarray.clip_gradient_to_calibration(
        drhs, rhs_in, rhs_calibration, config.rhs_calibration_method
    )

  return dlhs, drhs, None, None


@functools.partial(jax.custom_vjp, nondiff_argnums=(4, 5))
def dot_general_qt_fwd_bwd(
    lhs: jax.Array,
    rhs: jax.Array,
    lhs_calibration: dict[str, jax.Array] | None,
    rhs_calibration: dict[str, jax.Array] | None,
    dimension_numbers: jax.lax.DotDimensionNumbers,
    config: DotGeneralQtConfig,
) -> jax.Array:
  """dot_general custom VJP."""
  result, _ = dot_general_qt_fwd(
      lhs, rhs, lhs_calibration, rhs_calibration, dimension_numbers, config
  )
  return result


dot_general_qt_fwd_bwd.defvjp(dot_general_qt_fwd, dot_general_qt_bwd)


def dot_general_qt(
    lhs: jax.Array,
    rhs: jax.Array,
    dimension_numbers: jax.lax.DotDimensionNumbers,
    config: DotGeneralQtConfig,
) -> jax.Array:
  """Quantized dot_general with backpropagation support."""
  lhs_calibration = None
  rhs_calibration = None

  if config.lhs_qtype and numerics.should_quantize(lhs.dtype):
    lhs_how = dot_general.get_how_to_quantize(
        dimension_numbers=dimension_numbers,
        ndims=(lhs.ndim, rhs.ndim),
        for_lhs=True,
        qtype=config.lhs_qtype,
        tile_size=config.tile_size,
        calibration_method=config.lhs_calibration_method,
    )
    if config.disable_channelwise_axes:
      lhs_how = dataclasses.replace(lhs_how, channelwise_axes=[])
    lhs_calibration = qarray.calibrate(lhs, lhs_how)
    if config.lhs_collect_quant_stat:
      lhs_calibration = config.lhs_collect_quant_stat(lhs_calibration)

  if config.rhs_qtype and numerics.should_quantize(rhs.dtype):
    rhs_how = dot_general.get_how_to_quantize(
        dimension_numbers=dimension_numbers,
        ndims=(lhs.ndim, rhs.ndim),
        for_lhs=False,
        qtype=config.rhs_qtype,
        tile_size=config.tile_size,
        calibration_method=config.rhs_calibration_method,
    )
    if config.disable_channelwise_axes:
      rhs_how = dataclasses.replace(rhs_how, channelwise_axes=[])
    rhs_calibration = qarray.calibrate(rhs, rhs_how)
    if config.rhs_collect_quant_stat:
      rhs_calibration = config.rhs_collect_quant_stat(rhs_calibration)

  return dot_general_qt_fwd_bwd(
      lhs, rhs, lhs_calibration, rhs_calibration, dimension_numbers, config
  )
