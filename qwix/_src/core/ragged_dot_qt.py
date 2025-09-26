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
import jax
from jax import numpy as jnp
from qwix._src.core import numerics
from qwix._src.core import qarray
from qwix._src.core import ragged_dot
from qwix._src.core import ragged_dot_general


@dataclasses.dataclass(slots=True, frozen=True, kw_only=True)
class RaggedDotQtConfig:
  """Configuration for ragged_dot_qt."""

  # Forward pass settings
  lhs_qtype: jax.typing.DTypeLike | None = None
  rhs_qtype: jax.typing.DTypeLike | None = None

  # Backward pass settings
  dlhs_grad_qtype: jax.typing.DTypeLike | None = None
  drhs_grad_qtype: jax.typing.DTypeLike | None = None


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
  qlhs = qarray.quantize(
      lhs, ragged_dot.get_how_to_quantize(for_lhs=True, qtype=config.lhs_qtype)
  )
  qrhs = qarray.quantize(
      rhs, ragged_dot.get_how_to_quantize(for_lhs=False, qtype=config.rhs_qtype)
  )
  primal_out = ragged_dot.ragged_dot(
      qlhs, qrhs, group_sizes, precision, preferred_element_type, group_offset
  )
  return primal_out, (qlhs, qrhs)


def ragged_dot_qt_bwd(
    # Nondiff args passed from fwd pass
    group_sizes: jax.Array,
    config: RaggedDotQtConfig,
    precision: jax.lax.PrecisionLike,
    preferred_element_type: jax.typing.DTypeLike | None,
    group_offset: jax.Array | None,
    # Residuals from fwd pass
    residuals: tuple[qarray.MaybeQArray, qarray.MaybeQArray],
    g: jax.Array,
) -> tuple[jax.Array, jax.Array]:
  """Backward pass for ragged_dot_qt custom VJP."""
  (lhs, rhs) = residuals  # lhs [M, K], rhs [G, K, N], g [M, N]
  num_segments = len(group_sizes)
  segment_ids = jnp.repeat(jnp.arange(num_segments), group_sizes)

  # dlhs = ragged_dot_general(g, rhs)
  # [M, K] = [M, N] @ [G, K, N]
  dlhs_dnums = jax.lax.RaggedDotDimensionNumbers(
      dot_dimension_numbers=(((1,), (2,)), ((), ())),
      lhs_ragged_dimensions=[0],
      rhs_group_dimensions=[0],
  )
  if (
      isinstance(rhs, qarray.QArray)
      and config.dlhs_grad_qtype is not None
      and rhs.zero_point is None
  ):
    rhs_scale_repeated = rhs.scale[segment_ids]  # [M, 1, N]
    rhs_scale_repeated = rhs_scale_repeated.squeeze(axis=1)  # [M, N]
    g_for_dlhs = g * rhs_scale_repeated  # [M, N] * [M, N]
    rhs_val = rhs.qvalue  # [G, K, N]
  else:
    g_for_dlhs = g  # [M, N]
    rhs_val = rhs  # [G, K, N]

  if config.dlhs_grad_qtype and numerics.should_quantize(g_for_dlhs.dtype):
    g_how = qarray.HowToQuantize(
        qtype=config.dlhs_grad_qtype,
        channelwise_axes=[0],  # [M, N]
    )
    g_for_dlhs = qarray.quantize(g_for_dlhs, g_how)

  dlhs = ragged_dot_general.ragged_dot_general(
      g_for_dlhs,
      rhs_val,
      group_sizes,
      dimension_numbers=dlhs_dnums,
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
  if (
      isinstance(lhs, qarray.QArray)
      and config.drhs_grad_qtype is not None
      and lhs.zero_point is None
  ):
    g_for_drhs = g * lhs.scale  # [M, N] * [M, 1]
    lhs_val = lhs.qvalue  # [M, K]
  else:
    g_for_drhs = g  # [M, N]
    lhs_val = lhs  # [M, K]

  if config.drhs_grad_qtype and numerics.should_quantize(g_for_drhs.dtype):
    g_how = qarray.HowToQuantize(
        qtype=config.drhs_grad_qtype,
        channelwise_axes=[1],  # [M, N]
    )
    g_for_drhs = qarray.quantize(g_for_drhs, g_how)

  drhs = ragged_dot_general.ragged_dot_general(
      lhs_val,
      g_for_drhs,
      group_sizes,
      dimension_numbers=drhs_dnums,
      precision=precision,
      preferred_element_type=preferred_element_type,
      group_offset=group_offset,
  )

  return dlhs, drhs


@functools.partial(jax.custom_vjp, nondiff_argnums=(2, 3, 4, 5, 6))
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
