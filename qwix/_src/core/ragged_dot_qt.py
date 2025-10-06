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


def _ragged_dot_general(
    lhs,
    rhs,
    group_sizes,
    dimension_numbers,
    precision: jax.lax.PrecisionLike = None,
    preferred_element_type: jax.typing.DTypeLike | None = None,
    group_offset: jax.Array | None = None,
):
  """Quantized ragged_dot_general using in backward ragged_dot_qt.

  In the ragged_dot_qt backward pass, quantization is applied to incoming
  gradient only. In dlhs calculation, gradient(lhs of backward pass) has
  lhs_scale of shape [M, 1]. In drhs calculation, gradient(rhs of backward pass)
  has rhs_scale of shape [1, N].

  Args:
    lhs: The left-hand side array.
    rhs: The right-hand side array.
    group_sizes: An array specifying the sizes of the groups.
    dimension_numbers: Dimension numbers for the ragged dot general.
    precision: Optional precision for the dot product.
    preferred_element_type: Optional type promotion for the output.
    group_offset: Optional offset for group indices.

  Returns:
    The result of the ragged dot general operation.
  """
  lhs_val = lhs.qvalue if isinstance(lhs, qarray.QArray) else lhs
  rhs_val = rhs.qvalue if isinstance(rhs, qarray.QArray) else rhs
  lhs_scale = lhs.scale if isinstance(lhs, qarray.QArray) else None
  rhs_scale = rhs.scale if isinstance(rhs, qarray.QArray) else None

  preferred_element_type, result_type = qarray.get_accumulator_and_result_type(
      lhs, rhs, preferred_element_type=preferred_element_type
  )

  out = jax.lax.ragged_dot_general(
      lhs_val,
      rhs_val,
      group_sizes,
      dimension_numbers,
      precision=precision,
      preferred_element_type=preferred_element_type,
      group_offset=group_offset,
  )

  if lhs_scale is not None:  # [M, 1]
    out = qarray.call_with_generic_broadcast(jnp.multiply, out, lhs_scale)
  if rhs_scale is not None:  # [1, N]
    rhs_scale = qarray.transpose_array(rhs_scale, (0, None, 1))
    out = qarray.call_with_generic_broadcast(jnp.multiply, out, rhs_scale)
  return out.astype(result_type)


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

  # dlhs = ragged_dot_general(g, rhs), dims are [M, K] = [M, N] @ [G, K, N]
  dlhs_dnums = jax.lax.RaggedDotDimensionNumbers(
      dot_dimension_numbers=(((1,), (2,)), ((), ())),
      lhs_ragged_dimensions=[0],
      rhs_group_dimensions=[0],
  )
  g_for_dlhs = g
  if config.dlhs_grad_qtype:
    if isinstance(rhs, qarray.QArray):
      assert rhs.zero_point is None
      # Support channelwise only on non-group and non-contracting axes.
      g_for_dlhs = g * rhs.scale.squeeze(axis=0)  # [M, N] * [1, N]
      rhs_for_dlhs = rhs.qvalue
    else:
      rhs_for_dlhs = rhs
    g_how = qarray.HowToQuantize(
        qtype=config.dlhs_grad_qtype,
        channelwise_axes=[0],  # [M, N]
    )
    g_for_dlhs = qarray.quantize(g_for_dlhs, g_how)
  else:
    # If dlhs_grad_qtype is None, calculate dlhs in fp.
    rhs_for_dlhs = (
        qarray.dequantize(rhs) if isinstance(rhs, qarray.QArray) else rhs
    )
  dlhs = _ragged_dot_general(
      g_for_dlhs,
      rhs_for_dlhs,
      group_sizes,
      dimension_numbers=dlhs_dnums,
      precision=precision,
      preferred_element_type=preferred_element_type,
      group_offset=group_offset,
  )

  # drhs = ragged_dot_general(lhs, g), dims are [G, K, N] = [M, K] @ [M, N]
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
      lhs_for_drhs = lhs.qvalue
    else:
      lhs_for_drhs = lhs
    g_how = qarray.HowToQuantize(
        qtype=config.drhs_grad_qtype,
        channelwise_axes=[1],  # [M, N]
    )
    g_for_drhs = qarray.quantize(g_for_drhs, g_how)
  else:
    # If drhs_grad_qtype is None, calculate drhs in fp.
    lhs_for_drhs = (
        qarray.dequantize(lhs) if isinstance(lhs, qarray.QArray) else lhs
    )
  drhs = _ragged_dot_general(
      lhs_for_drhs,
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
