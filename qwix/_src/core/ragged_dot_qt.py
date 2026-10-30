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

"""Quantized jax.lax.ragged_dot with quantized back propagation support."""

import dataclasses
import functools
from typing import Any
import jax
from jax import numpy as jnp
from qwix._src import interception
from qwix._src.core import multipass_dot
from qwix._src.core import qarray
from qwix._src.core import ragged_dot


@dataclasses.dataclass(slots=True, frozen=True, kw_only=True)
class RaggedDotQtConfig:
  """Configuration for ragged_dot_qt."""

  # Forward pass settings
  lhs_qtype: jax.typing.DTypeLike | None = None
  rhs_qtype: jax.typing.DTypeLike | None = None
  tile_size: int | float | None = None
  multipass_mode: Any = None

  # Backward pass settings
  dlhs_grad_qtype: jax.typing.DTypeLike | None = None
  drhs_grad_qtype: jax.typing.DTypeLike | None = None
  dlhs_multipass_mode: Any = None
  drhs_multipass_mode: Any = None


def _multipass_ragged_dot_general(
    lhs: jax.Array,
    rhs: jax.Array,
    group_sizes: jax.Array,
    dimension_numbers: jax.lax.RaggedDotDimensionNumbers,
    multipass_mode: Any,
    *,
    lhs_channelwise_axes: tuple[int, ...],
    rhs_channelwise_axes: tuple[int, ...],
    tile_size: int | float | None = None,
    precision: jax.lax.PrecisionLike = None,
    preferred_element_type: jax.typing.DTypeLike | None = None,
    group_offset: jax.Array | None = None,
) -> jax.Array:
  """Executes multi-pass ragged_dot_general over decomposed operands."""
  resolved = multipass_dot.resolve_multipass_config(
      multipass_mode, tile_size=tile_size
  )
  if resolved is None:
    raise ValueError(
        f'Unsupported multipass_mode for ragged_dot_qt: {multipass_mode!r}'
    )
  dot_dnums = dimension_numbers.dot_dimension_numbers
  ndims = (lhs.ndim, rhs.ndim)
  a_passes = multipass_dot.decompose_multipass_operand(
      lhs,
      resolved.lhs,
      dimension_numbers=dot_dnums,
      ndims=ndims,
      for_lhs=True,
      channelwise_axes=lhs_channelwise_axes,
  )
  b_passes = multipass_dot.decompose_multipass_operand(
      rhs,
      resolved.rhs,
      dimension_numbers=dot_dnums,
      ndims=ndims,
      for_lhs=False,
      channelwise_axes=rhs_channelwise_axes,
  )
  target_out_type = (
      preferred_element_type
      if preferred_element_type is not None
      else jnp.result_type(lhs.dtype, rhs.dtype)
  )
  acc = None
  for i, j in resolved.get_active_terms():
    term = ragged_dot.ragged_dot_general(
        a_passes[i],
        b_passes[j],
        group_sizes,
        dimension_numbers=dimension_numbers,
        precision=precision,
        preferred_element_type=resolved.acc_dtype,
        group_offset=group_offset,
    ).astype(resolved.acc_dtype)
    acc = term if acc is None else acc + term
  assert acc is not None
  return acc.astype(target_out_type)


@interception.disable_interceptions
def ragged_dot_qt_fwd(
    lhs: jax.Array,
    rhs: jax.Array,
    group_sizes: jax.Array,
    config: RaggedDotQtConfig,
    precision: jax.lax.PrecisionLike = None,
    preferred_element_type: jax.typing.DTypeLike | None = None,
    group_offset: jax.Array | None = None,
):
  """Forward pass for ragged_dot_qt custom VJP."""
  tiled_axes = {1: config.tile_size} if config.tile_size else {}
  if config.lhs_qtype is not None:
    lhs_how = qarray.HowToQuantize(
        qtype=config.lhs_qtype, channelwise_axes=[0], tiled_axes=tiled_axes
    )
    qlhs = qarray.quantize(lhs, lhs_how)
  else:
    qlhs = lhs

  if config.rhs_qtype is not None:
    rhs_how = qarray.HowToQuantize(
        qtype=config.rhs_qtype, channelwise_axes=[2], tiled_axes=tiled_axes
    )
    qrhs = qarray.quantize(rhs, rhs_how)
  else:
    qrhs = rhs

  if config.multipass_mode is not None:
    primal_out = _multipass_ragged_dot_general(
        lhs,
        rhs,
        group_sizes,
        ragged_dot._BASIC_RAGGED_DOT_DIMENSION_NUMBERS,  # pylint: disable=protected-access
        config.multipass_mode,
        lhs_channelwise_axes=(0,),
        rhs_channelwise_axes=(2,),
        tile_size=config.tile_size,
        precision=precision,
        preferred_element_type=preferred_element_type,
        group_offset=group_offset,
    )
  else:
    primal_out = ragged_dot.ragged_dot(
        qlhs, qrhs, group_sizes, precision, preferred_element_type, group_offset
    )

  res_lhs = (
      lhs
      if (config.tile_size or config.drhs_multipass_mode is not None)
      else qlhs
  )
  res_rhs = (
      rhs
      if (config.tile_size or config.dlhs_multipass_mode is not None)
      else qrhs
  )
  return primal_out, (res_lhs, res_rhs, group_sizes)


def ragged_dot_qt_bwd(
    # Nondiff args passed from fwd pass
    config: RaggedDotQtConfig,
    precision: jax.lax.PrecisionLike,
    preferred_element_type: jax.typing.DTypeLike | None,
    group_offset: jax.Array | None,
    # Residuals from fwd pass
    residuals: tuple[qarray.MaybeQArray, qarray.MaybeQArray, jax.Array],
    g: jax.Array,
) -> tuple[jax.Array, jax.Array, None]:
  """Backward pass for ragged_dot_qt custom VJP."""
  lhs, rhs, group_sizes = residuals  # lhs [M, K], rhs [G, K, N], g [M, N]

  # dlhs = ragged_dot(g, rhs.swapaxes(1, 2))
  # [M, K] = [M, N] @ [G, N, K]
  # Transpose rhs as chlo.ragged_dot supports limited DimensionNumbers variants.
  g_for_dlhs = g
  rhs = rhs.swapaxes(1, 2)  # [G, N, K]
  if config.dlhs_multipass_mode is not None:
    if isinstance(rhs, qarray.QArray):
      rhs = qarray.dequantize(rhs)
    dlhs = _multipass_ragged_dot_general(
        g_for_dlhs,
        rhs,
        group_sizes,
        ragged_dot._BASIC_RAGGED_DOT_DIMENSION_NUMBERS,  # pylint: disable=protected-access
        config.dlhs_multipass_mode,
        lhs_channelwise_axes=(0,),
        rhs_channelwise_axes=(2,),
        tile_size=config.tile_size,
        precision=precision,
        preferred_element_type=preferred_element_type,
        group_offset=group_offset,
    )
  else:
    if config.dlhs_grad_qtype:
      if isinstance(rhs, qarray.QArray):
        assert rhs.zero_point is None
        # Support channelwise only on non-group and non-contracting axes.
        g_for_dlhs = g * rhs.scale.squeeze(axis=2)  # [M, N] * [1, N]
        rhs = rhs.qvalue
      g_how = qarray.HowToQuantize(
          qtype=config.dlhs_grad_qtype,
          channelwise_axes=[0],  # [M, N]
      )
      g_for_dlhs = qarray.quantize(g_for_dlhs, g_how)
    dlhs = ragged_dot.ragged_dot(
        g_for_dlhs,
        rhs,
        group_sizes,
        precision=precision,
        preferred_element_type=preferred_element_type,
        group_offset=group_offset,
    )

  # drhs = ragged_dot_general(lhs, g)
  # [G, K, N] = [M, K] @ [M, N]
  drhs_dnums = jax.lax.RaggedDotDimensionNumbers(
      dot_dimension_numbers=(((0,), (0,)), ((), ())),
      lhs_ragged_dimensions=[0],
      rhs_group_dimensions=[],
  )
  g_for_drhs = g
  if config.drhs_multipass_mode is not None:
    if isinstance(lhs, qarray.QArray):
      lhs = qarray.dequantize(lhs)
    drhs = _multipass_ragged_dot_general(
        lhs,
        g_for_drhs,
        group_sizes,
        drhs_dnums,
        config.drhs_multipass_mode,
        lhs_channelwise_axes=(1,),
        rhs_channelwise_axes=(1,),
        tile_size=config.tile_size,
        precision=precision,
        preferred_element_type=preferred_element_type,
        group_offset=group_offset,
    )
  else:
    if config.drhs_grad_qtype:
      if isinstance(lhs, qarray.QArray):
        assert lhs.zero_point is None
        g_for_drhs = g * lhs.scale  # [M, N] * [M, 1]
        lhs = lhs.qvalue
      g_how = qarray.HowToQuantize(
          qtype=config.drhs_grad_qtype,
          channelwise_axes=[1],  # [M, N]
      )
      g_for_drhs = qarray.quantize(g_for_drhs, g_how)
    drhs = ragged_dot.ragged_dot_general(
        lhs,
        g_for_drhs,
        group_sizes,
        dimension_numbers=drhs_dnums,
        precision=precision,
        preferred_element_type=preferred_element_type,
        group_offset=group_offset,
    )

  return dlhs, drhs, None


@functools.partial(jax.custom_vjp, nondiff_argnums=(3, 4, 5, 6))
def ragged_dot_qt(
    lhs: jax.Array,
    rhs: jax.Array,
    group_sizes: jax.Array,
    config: RaggedDotQtConfig,
    precision: jax.lax.PrecisionLike = None,
    preferred_element_type: jax.typing.DTypeLike | None = None,
    group_offset: jax.Array | None = None,
) -> jax.Array:
  """Quantized ragged_dot with backpropagation support."""
  result, _ = ragged_dot_qt_fwd(
      lhs,
      rhs,
      group_sizes,
      config,
      precision,
      preferred_element_type,
      group_offset,
  )
  return result


ragged_dot_qt.defvjp(ragged_dot_qt_fwd, ragged_dot_qt_bwd)
