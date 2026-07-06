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
import functools
import math
from typing import Any
import jax
from jax import numpy as jnp
from qwix._src.core import qarray


def mxfp_dot_general(
    lhs: qarray.MaybeQArray,
    rhs: qarray.MaybeQArray,
    dimension_numbers: jax.lax.DotDimensionNumbers,
    preferred_element_type: jax.typing.DTypeLike | None = None,
) -> jax.Array | None:
  """Handles MXFP dot_general on ZFC and Blackwell GPUs.

  This dispatcher attempts to accelerate OCP/NVIDIA microscaled matmuls using
  GPU hardware Tensor Cores via `jax.nn.scaled_matmul`.

  Note that hardware acceleration is strictly supported only when BOTH operands
  are microscaled formats (MXFP8, MXFP4, or NVFP4). One-sided microscaled
  operations are not supported by the hardware, and will cleanly return `None`
  to fall back to standard float emulation.

  Args:
    lhs: Left hand side operand.
    rhs: Right hand side operand.
    dimension_numbers: Dot dimension numbers.
    preferred_element_type: Preferred element type for output.

  Returns:
    A jax.Array with the result, or None to fall back to emulation.
  """
  # jax.nn.scaled_matmul is the fused GPU fast path (natively on Blackwell via
  # cuDNN, emulated on legacy GPUs). On TPU/CPU there is no scaled_matmul
  # lowering, so we return None and the caller falls back to the native tiled
  # fp8 dot_general path (dot_general._fast_dot_general), which is correct and
  # MXU-accelerated on TPU. If/when scaled_matmul gains a TPU lowering, it can be
  # enabled here as an additional fast path.
  if _get_primary_platform() == "gpu":
    return _gpu_mxfp_dot(lhs, rhs, dimension_numbers, preferred_element_type)

  return None


def _is_mxfp(operand: Any) -> bool:
  """Verifies whether the operand is an OCP/NVIDIA microscaled format."""
  if isinstance(operand, qarray.QArray):
    return operand.qtype in ("mxfp8", "mxfp4", "nvfp4")
  return False


@functools.cache
def _get_primary_platform() -> str:
  """Returns the JAX platform name cached to avoid tracer round-trips."""
  return jax.devices()[0].platform


def _gpu_mxfp_dot(lhs, rhs, dimension_numbers, preferred_element_type):
  """GPU specific MXFP dot."""
  if not (_is_mxfp(lhs) and _is_mxfp(rhs)):
    return None

  (lhs_ca, rhs_ca), (lhs_ba, rhs_ba) = dimension_numbers
  lhs_val_3d, lhs_scale_3d = _flatten_to_3d(lhs, lhs_ca, lhs_ba)
  rhs_val_3d, rhs_scale_3d = _flatten_to_3d(rhs, rhs_ca, rhs_ba)

  # jax.nn.scaled_matmul is natively accelerated on Blackwell GPUs (via cuDNN
  # scaled matmul kernels) and emulated/decomposed on legacy GPUs (like H100).
  out_3d = jax.nn.scaled_matmul(
      lhs_val_3d,
      rhs_val_3d,
      lhs_scale_3d,
      rhs_scale_3d,
      preferred_element_type=preferred_element_type or jnp.float32,
  )

  return _unflatten_from_3d(out_3d, lhs, rhs, dimension_numbers)


def _flatten_to_3d(
    operand: qarray.QArray,
    ca: Sequence[int],
    ba: Sequence[int],
) -> tuple[jax.Array, jax.Array]:
  """Flattens a QArray operand and its scale to 3D for scaled_matmul."""
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
  """Reshapes the 3D scaled_matmul output back to the expected target shape."""
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
