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
from qwix._src import interception
from qwix._src.core import qarray
from qwix._src.core import ragged_dot


@dataclasses.dataclass(slots=True, frozen=True, kw_only=True)
class RaggedDotQtConfig:
  """Configuration for ragged_dot_qt."""

  # Forward pass settings
  lhs_qtype: jax.typing.DTypeLike | None = None
  rhs_qtype: jax.typing.DTypeLike | None = None

  # Backward pass settings
  dlhs_grad_qtype: jax.typing.DTypeLike | None = None
  drhs_grad_qtype: jax.typing.DTypeLike | None = None


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
  qlhs = qarray.quantize(
      # lhs shape [M, K]: contracting axis=1, channelwise axis=0
      lhs,
      qarray.HowToQuantize(qtype=config.lhs_qtype, channelwise_axes=[0]),
  )
  qrhs = qarray.quantize(
      # rhs shape [G, K, N]: contracting axis=1, channelwise axes=2
      rhs,
      qarray.HowToQuantize(qtype=config.rhs_qtype, channelwise_axes=[2]),
  )
  primal_out = ragged_dot.ragged_dot(
      qlhs, qrhs, group_sizes, precision, preferred_element_type, group_offset
  )
  return primal_out, (qlhs, qrhs, group_sizes)


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
  (lhs, rhs, group_sizes) = residuals  # lhs [M, K], rhs [G, K, N], g [M, N]

  # dlhs = ragged_dot(g, rhs.swapaxes(1, 2))
  # [M, K] = [M, N] @ [G, N, K]
  # Transpose rhs as chlo.ragged_dot supports limited DimensionNumbers variants.
  g_for_dlhs = g
  rhs = rhs.swapaxes(1, 2)  # [G, N, K]
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
