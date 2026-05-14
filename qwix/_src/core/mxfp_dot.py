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
"""MXFP dot_general dispatch for ZFC and Blackwell."""

from typing import Any
import jax
from jax import numpy as jnp
from qwix._src.core import qarray


def is_mxfp(operand: Any) -> bool:
  if isinstance(operand, qarray.QArray):
    return operand.qtype in ("mxfp8", "mxfp4", "nvfp4")
  return False


def mxfp_dot_general(
    lhs: qarray.MaybeQArray,
    rhs: qarray.MaybeQArray,
    dimension_numbers: jax.lax.DotDimensionNumbers,
    precision: jax.lax.PrecisionLike = None,
    preferred_element_type: jax.typing.DTypeLike | None = None,
    **kwargs,
) -> jax.Array | None:
  """Handles MXFP dot_general on ZFC and Blackwell GPUs.

  Args:
    lhs: Left hand side operand.
    rhs: Right hand side operand.
    dimension_numbers: Dot dimension numbers.
    precision: Precision for the operation.
    preferred_element_type: Preferred element type for output.
    **kwargs: Additional arguments.

  Returns:
    A jax.Array with the result, or None to fall back to emulation.
  """
  if not (is_mxfp(lhs) or is_mxfp(rhs)):
    return None

  del precision, kwargs

  # If we are on GPU
  if jax.devices()[0].platform == "gpu":
    return _gpu_mxfp_dot(lhs, rhs, dimension_numbers, preferred_element_type)

  return None


def _flatten_to_3d(val, scale, ca, ba):
  """Flattens an operand and its scale to 3D for scaled_matmul."""
  ndim = val.ndim
  free_axes = [i for i in range(ndim) if i not in ca and i not in ba]
  perm = list(ba) + free_axes + list(ca)

  val_t = jnp.transpose(val, perm)

  batch_size = 1
  for a in ba:
    batch_size *= val.shape[a]
  free_size = 1
  for a in free_axes:
    free_size *= val.shape[a]
  contracting_size = 1
  for a in ca:
    contracting_size *= val.shape[a]

  val_3d = jnp.reshape(val_t, (batch_size, free_size, contracting_size))

  scale_t = jnp.transpose(scale, perm)
  contracting_scale_size = 1
  for a in ca:
    contracting_scale_size *= scale.shape[a]
  scale_3d = jnp.reshape(
      scale_t, (batch_size, free_size, contracting_scale_size)
  )

  return val_3d, scale_3d


def _gpu_mxfp_dot(lhs, rhs, dimension_numbers, preferred_element_type):
  """GPU specific MXFP dot."""
  if not (isinstance(lhs, qarray.QArray) and isinstance(rhs, qarray.QArray)):
    return None

  (lhs_ca, rhs_ca), (lhs_ba, rhs_ba) = dimension_numbers

  # Flatten LHS
  lhs_val_3d, lhs_scale_3d = _flatten_to_3d(
      lhs.qvalue, lhs.scale, lhs_ca, lhs_ba
  )

  # Flatten RHS
  rhs_val_3d, rhs_scale_3d = _flatten_to_3d(
      rhs.qvalue, rhs.scale, rhs_ca, rhs_ba
  )

  if preferred_element_type is None:
    preferred_element_type = jnp.float32

  out_3d = jax.nn.scaled_matmul(
      lhs_val_3d,
      rhs_val_3d,
      lhs_scale_3d,
      rhs_scale_3d,
      preferred_element_type=preferred_element_type,
  )

  # Reshape result back to expected output shape.
  batch_shape = [lhs.shape[i] for i in lhs_ba]
  lhs_free_axes = [
      i for i in range(lhs.ndim) if i not in lhs_ca and i not in lhs_ba
  ]
  lhs_free_shape = [lhs.shape[i] for i in lhs_free_axes]
  rhs_free_axes = [
      i for i in range(rhs.ndim) if i not in rhs_ca and i not in rhs_ba
  ]
  rhs_free_shape = [rhs.shape[i] for i in rhs_free_axes]

  target_shape = batch_shape + lhs_free_shape + rhs_free_shape

  return jnp.reshape(out_3d, target_shape)
