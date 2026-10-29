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

_MXFP4_BLOCK_SIZE = 32
# 6 * 2**13 fits float8_e5m2; blocks 2**26 below the channel max flush to zero.
_E5M2_HEADROOM = 2.0**13


def mxfp_dot_general(
    lhs: qarray.MaybeQArray,
    rhs: qarray.MaybeQArray,
    dimension_numbers: jax.lax.DotDimensionNumbers,
    preferred_element_type: jax.typing.DTypeLike | None = None,
) -> jax.Array | None:
  """Handles MXFP dot_general using `jax.lax.scaled_dot`.

  This dispatcher attempts to accelerate or decompose OCP/NVIDIA microscaled
  matmuls using `jax.lax.scaled_dot` (hardware Tensor Cores on Blackwell, or
  JAX emulation/decomposition on TPUs, CPUs, and legacy GPUs).

  Note that `scaled_dot` is supported when BOTH operands are microscaled
  formats (MXFP8, MXFP4, or NVFP4) with matching batch and contracting scale
  dimensions. One-sided microscaled operations or mismatched scale dimensions
  will first attempt to use the `_mxfp4_weight_dot_general` path, and otherwise
  return `None` to fall back to standard float emulation.

  Note: `jax.lax.scaled_dot` only supports floating-point microscaled formats
  and does not support integer microscaled formats (e.g. mxint8). Integer
  microscaled formats bypass this dispatcher and fall back to Qwix's reference
  dot_general path.

  Note: scaled_dot hardware acceleration is currently restricted to
  single-axis contraction (len(ca) == 1). For multi-axis contractions use
  reference emulation for numerical precision.

  Args:
    lhs: Left hand side operand.
    rhs: Right hand side operand.
    dimension_numbers: Dot dimension numbers.
    preferred_element_type: Preferred element type for output.

  Returns:
    A jax.Array with the result, or None to fall back to emulation.
  """
  if not (_is_mxfp(lhs) and _is_mxfp(rhs)):
    return _mxfp4_weight_dot_general(
        lhs, rhs, dimension_numbers, preferred_element_type
    )

  (lhs_ca, rhs_ca), (lhs_ba, rhs_ba) = dimension_numbers

  # jax.lax.scaled_dot (and underlying hardware like NVIDIA Blackwell Tensor
  # Cores) natively operates on a single contiguous 1D reduction axis (K).
  # Attempting to flatten multiple contracting dimensions (such as batch and
  # sequence axes in backward pass weight gradient computations g^T y) into 1D
  # causes tile-scale boundary misalignment:
  # 1) If the inner contracting axis is not a multiple of the tile size, 1D
  #    blocks straddle across outer dimension boundaries, grouping elements
  #    from different slices under a single scale factor.
  # 2) If scale factors are shared/broadcast along an outer contracting axis
  #    (e.g., the batch axis in Qwix's backward pass), 1D flattening distorts
  #    the inferred block size (to B * T) and stride-shifts scales across
  #    slices, corrupting gradients (e.g., ~13 dB SQNR vs > 25 dB).
  # We therefore fall back to Qwix's reference microscaled dot_general
  # emulation for multi-axis contractions, preserving exact numerical accuracy.
  if len(lhs_ca) > 1 or len(rhs_ca) > 1:
    return None

  lhs_val_3d, lhs_scale_3d = _flatten_to_3d(lhs, lhs_ca, lhs_ba)
  rhs_val_3d, rhs_scale_3d = _flatten_to_3d(rhs, rhs_ca, rhs_ba)

  if not _inputs_compatible(lhs_val_3d, rhs_val_3d, lhs_scale_3d, rhs_scale_3d):
    return None

  _, result_type = qarray.get_accumulator_and_result_type(
      lhs, rhs, preferred_element_type=preferred_element_type
  )

  # TODO(b/538686860): Migrate to new API once it's available.
  # jax.lax.scaled_dot is natively accelerated on Blackwell GPUs (via cuDNN
  # scaled dot kernels, Triton kernels, or cuBLASLt as a fallback) and
  # emulated/decomposed on TPUs, CPUs, and legacy GPUs (like H100).
  try:
    out_3d = jax.lax.scaled_dot(
        lhs_val_3d,
        rhs_val_3d,
        lhs_scale=lhs_scale_3d,
        rhs_scale=rhs_scale_3d,
        dimension_numbers=(((2,), (2,)), ((0,), (0,))),
        preferred_element_type=result_type,
    )
  except Exception:  # pylint: disable=broad-except
    return None

  return _unflatten_from_3d(out_3d, lhs, rhs, dimension_numbers)


def _mxfp4_weight_dot_general(
    lhs: qarray.MaybeQArray,
    rhs: qarray.MaybeQArray,
    dimension_numbers: jax.lax.DotDimensionNumbers,
    preferred_element_type: jax.typing.DTypeLike | None,
) -> jax.Array | None:
  """Re-encodes MXFP4 weights to float8_e5m2 for FP8 matmul."""
  if not (isinstance(lhs, qarray.QArray) and isinstance(rhs, qarray.QArray)):
    return None
  (lhs_ca, rhs_ca), (lhs_ba, rhs_ba) = dimension_numbers
  # Re-encodes act[..., K] @ weight[K, ...]; einsum may swap them.
  swapped = lhs.qtype == "mxfp4"
  act, act_ca, w, w_ca = (
      (rhs, rhs_ca, lhs, lhs_ca) if swapped else (lhs, lhs_ca, rhs, rhs_ca)
  )
  c = len(w_ca)
  act_free, w_free = act.shape[:-c], w.shape[c:]
  m, k, n = math.prod(act_free), math.prod(w.shape[:c]), math.prod(w_free)
  if not (
      c > 0
      and 0 not in (*act.shape, *w.shape)
      and not (lhs_ba or rhs_ba)
      and w.qtype == "mxfp4"
      and act.qvalue.dtype in (jnp.float8_e4m3fn, jnp.float8_e5m2)
      and act.zero_point is None
      and w.zero_point is None
      and set(w_ca) == set(range(c))
      and tuple(act.ndim - c + a for a in w_ca) == tuple(act_ca)
      and all(s == 1 for s in act.scale.shape[-c:])
      and all(
          act.scale.shape[a] in (1, act.shape[a]) for a in range(act.ndim - c)
      )
      and all(
          w.scale.shape[a] in (1, w.shape[a])
          for a in range(w.ndim)
          if a != c - 1
      )
      and w.scale.shape[c - 1] * _MXFP4_BLOCK_SIZE == w.shape[c - 1]
  ):
    return None

  _, out_dtype = qarray.get_accumulator_and_result_type(
      lhs, rhs, preferred_element_type=preferred_element_type
  )
  act_scale = jnp.broadcast_to(act.scale, (*act_free, *(1,) * c)).reshape(m, 1)
  w_scale = jnp.broadcast_to(
      w.scale, (*w.shape[: c - 1], w.scale.shape[c - 1], *w_free)
  ).reshape(k // _MXFP4_BLOCK_SIZE, n)
  # Keep 2D operands behind a barrier so XLA does not hoist weight re-encoding,
  # fuse activation quantization into the dot, or relayout 3D weights.
  act_q, w_q, w_scale = jax.lax.optimization_barrier(
      (act.qvalue.reshape(m, k), w.qvalue.reshape(k, n), w_scale)
  )
  w_scale = w_scale.astype(jnp.bfloat16)
  # Shift each column's max scale to _E5M2_HEADROOM so w * ratio fits in e5m2
  # without rounding (dividing by a power of two only changes the exponent).
  s_max = jnp.max(w_scale, axis=0, keepdims=True)
  s_max = jnp.where(s_max == 0, 1.0, s_max)
  ratio = (w_scale * (_E5M2_HEADROOM / s_max)).astype(jnp.bfloat16)
  w_q = w_q.astype(jnp.bfloat16).reshape(
      k // _MXFP4_BLOCK_SIZE, _MXFP4_BLOCK_SIZE, n
  )
  w_e5m2 = (w_q * ratio[:, None, :]).reshape(k, n).astype(jnp.float8_e5m2)
  out = jnp.dot(
      act_q,
      w_e5m2,
      preferred_element_type=jnp.float32,
  )
  out = out * (s_max.astype(jnp.float32) / _E5M2_HEADROOM) * act_scale
  out = out.astype(out_dtype).reshape(*act_free, *w_free)
  if swapped:  # dot_general puts the lhs free axes first.
    out = jnp.moveaxis(out, range(len(act_free)), range(-len(act_free), 0))
  return out


def _is_mxfp(operand: Any) -> TypeGuard[qarray.QArray]:
  """Verifies whether the operand is an OCP/NVIDIA microscaled format.

  Note that `jax.lax.scaled_dot` only supports floating-point microscaled
  types, so integer microscaled formats like mxint8 are excluded.

  Args:
    operand: The operand to check.

  Returns:
    True if the operand is an OCP/NVIDIA microscaled format.
  """
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
  # Batch dimensions (dim 0), contracting dimensions (dim 2), and scale block
  # counts along the contracting dimension must align between LHS and RHS.
  # Note that free/output dimensions (dim 1) do not need to match.
  if not (
      lhs_val_3d.shape[0] == rhs_val_3d.shape[0]
      and lhs_val_3d.shape[2] == rhs_val_3d.shape[2]
      and lhs_scale_3d.shape[0] == rhs_scale_3d.shape[0]
      and lhs_scale_3d.shape[2] == rhs_scale_3d.shape[2]
  ):
    return False

  # Contracting dimension K must be an exact multiple of scale count K_scale so
  # elements are partitioned into uniform, integer-sized blocks.
  k = lhs_val_3d.shape[2]
  k_scale = lhs_scale_3d.shape[2]
  if k_scale <= 0 or k % k_scale != 0:
    return False

  # Hardware Tensor Cores (e.g. NVIDIA Blackwell via cuDNN) and microscaling
  # standards (OCP MXFP8, MXFP4, MXINT8, NVFP4) strictly require block sizes
  # of 32 or 16 along the reduction dimension. Incompatible tile sizes (e.g. 8,
  # 24, 64) fall back to Qwix's reference software emulation.
  block_size = k // k_scale
  return block_size in (16, 32)


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
