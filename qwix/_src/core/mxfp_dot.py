# Copyright 2026 Google LLC
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
"""Block-scaled dot_general dispatch for eligible hardware devices."""

from collections.abc import Sequence
import math
from typing import Any, TypeGuard
import jax
from jax import numpy as jnp
from qwix._src.core import qarray


# Dimension numbers of the 3D form produced by _flatten_to_3d, i.e.
# (B, M, K) x (B, N, K) -> (B, M, N).
_DIMENSION_NUMBERS_3D = (((2,), (2,)), ((0,), (0,)))


def mxfp_dot_general(
    lhs: qarray.MaybeQArray,
    rhs: qarray.MaybeQArray,
    dimension_numbers: jax.lax.DotDimensionNumbers,
    preferred_element_type: jax.typing.DTypeLike | None = None,
) -> jax.Array | None:
  """Handles MXFP dot_general using `jax.lax.scaled_dot`.

  This dispatcher attempts to accelerate or decompose OCP/NVIDIA microscaled
  matmuls using `jax.lax.scaled_dot`, which emits an `xla.scaled_dot` composite.
  Today only the XLA GPU backend rewrites that composite into a native
  block-scaled dot; every other backend decomposes it into a regular
  dot_general.

  Note that `scaled_dot` is supported when BOTH operands are microscaled
  formats (MXFP8, MXFP4, or NVFP4) with matching batch and contracting scale
  dimensions. One-sided microscaled operations or mismatched scale dimensions
  will cleanly return `None` to fall back to standard float emulation.

  Args:
    lhs: Left hand side operand.
    rhs: Right hand side operand.
    dimension_numbers: Dot dimension numbers.
    preferred_element_type: Preferred element type for output.

  Returns:
    A jax.Array with the result, or None to fall back to emulation.
  """
  if not (_is_mxfp(lhs) and _is_mxfp(rhs)):
    return None

  (lhs_ca, rhs_ca), (lhs_ba, rhs_ba) = dimension_numbers
  lhs_val_3d, lhs_scale_3d = _flatten_to_3d(lhs, lhs_ca, lhs_ba)
  rhs_val_3d, rhs_scale_3d = _flatten_to_3d(rhs, rhs_ca, rhs_ba)

  if not _inputs_compatible(lhs_val_3d, rhs_val_3d, lhs_scale_3d, rhs_scale_3d):
    return None

  _, result_type = qarray.get_accumulator_and_result_type(
      lhs, rhs, preferred_element_type=preferred_element_type
  )

  # TODO(b/538686860): Only the XLA GPU backend rewrites the `xla.scaled_dot`
  # composite into a native block-scaled dot. Revisit once TPUs with native
  # MXFP MXUs lower it natively too.
  try:
    out_3d = jax.lax.scaled_dot(
        lhs_val_3d,
        rhs_val_3d,
        lhs_scale=lhs_scale_3d,
        rhs_scale=rhs_scale_3d,
        dimension_numbers=_DIMENSION_NUMBERS_3D,
        preferred_element_type=result_type,
    )
  except Exception:  # pylint: disable=broad-except
    return None

  return _unflatten_from_3d(out_3d, lhs, rhs, dimension_numbers)


def _is_mxfp(operand: Any) -> TypeGuard[qarray.QArray]:
  """Verifies whether the operand is an OCP/NVIDIA microscaled format."""
  return isinstance(operand, qarray.QArray) and operand.qtype in (
      "mxfp8",
      "mxfp8_16",
      "mxfp4",
      "nvfp4",
  )


def _inputs_compatible(
    lhs_val_3d: jax.Array,
    rhs_val_3d: jax.Array,
    lhs_scale_3d: jax.Array,
    rhs_scale_3d: jax.Array,
) -> bool:
  """Checks 3D value and scale tensors are compatible for scaled_dot."""
  return (
      lhs_val_3d.shape[0] == rhs_val_3d.shape[0]
      and lhs_val_3d.shape[2] == rhs_val_3d.shape[2]
      and lhs_scale_3d.shape[0] == rhs_scale_3d.shape[0]
      and lhs_scale_3d.shape[2] == rhs_scale_3d.shape[2]
      and _subchannel_supported(lhs_val_3d, lhs_scale_3d)
      and _subchannel_supported(rhs_val_3d, rhs_scale_3d)
  )


def _subchannel_supported(val_3d: jax.Array, scale_3d: jax.Array) -> bool:
  """Checks scaled_dot's subchannel constraints on the contracting dim.

  `jax.lax.scaled_dot` requires the contracting dim to be a multiple of, and at
  least twice as large as, the scale's contracting dim. Degenerate contractions
  (e.g. outer products, which flatten to a contracting size of 1) don't qualify.

  Args:
    val_3d: The 3D value tensor.
    scale_3d: The 3D scale tensor.

  Returns:
    Whether the contracting dim satisfies scaled_dot's subchannel constraints.
  """
  contracting_size = val_3d.shape[2]
  scale_size = scale_3d.shape[2]
  if contracting_size % scale_size != 0:
    return False
  return contracting_size // scale_size >= 2


def _flatten_to_3d(
    operand: qarray.QArray,
    ca: Sequence[int],
    ba: Sequence[int],
) -> tuple[jax.Array, jax.Array]:
  """Flattens a QArray operand and its scale to 3D for scaled_dot."""
  val = operand.qvalue
  scale = operand.scale
  ndim = operand.ndim
  free_axes = [i for i in range(ndim) if i not in ca and i not in ba]
  perm = list(ba) + free_axes + list(ca)

  val_t = jnp.transpose(val, perm)

  batch_size = math.prod(operand.shape[a] for a in ba)
  free_size = math.prod(operand.shape[a] for a in free_axes)
  contracting_size = math.prod(operand.shape[a] for a in ca)

  val_3d = jnp.reshape(val_t, (batch_size, free_size, contracting_size))

  # Broadcast scale to match batch and free dimensions of val.
  scale_broadcast_shape = [
      scale.shape[i] if i in ca else operand.shape[i] for i in range(ndim)
  ]
  scale = jnp.broadcast_to(scale, tuple(scale_broadcast_shape))

  scale_t = jnp.transpose(scale, perm)
  contracting_scale_size = math.prod(scale.shape[a] for a in ca)
  scale_3d = jnp.reshape(
      scale_t, (batch_size, free_size, contracting_scale_size)
  )

  return val_3d, scale_3d


def _unflatten_from_3d(
    out_3d: jax.Array,
    lhs: qarray.QArray,
    rhs: qarray.QArray,
    dimension_numbers: jax.lax.DotDimensionNumbers,
) -> jax.Array:
  """Reshapes the 3D scaled_dot output back to the expected target shape."""
  (lhs_ca, rhs_ca), (lhs_ba, rhs_ba) = dimension_numbers
  batch_shape = [lhs.shape[i] for i in lhs_ba]
  lhs_free_shape = [
      lhs.shape[i]
      for i in range(lhs.ndim)
      if i not in lhs_ca and i not in lhs_ba
  ]
  rhs_free_shape = [
      rhs.shape[i]
      for i in range(rhs.ndim)
      if i not in rhs_ca and i not in rhs_ba
  ]

  target_shape = batch_shape + lhs_free_shape + rhs_free_shape
  return jnp.reshape(out_3d, target_shape)
