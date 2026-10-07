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
"""1:4 structured sparse quantized dot_general lowering."""

from typing import Callable

import jax
from jax import numpy as jnp
from qwix._src.core import mxfp_dot
from qwix._src.core import qarray
from qwix._src.core import sparsity

_SUPPORTED_SPARSE_DTYPES = (
    'float8_e4m3fn',
    'float8_e5m2',
    'int4',
    'uint4',
    'int8',
    'bfloat16',
    'float32',
)

_NATIVE_SFS_SPARSE_DTYPES = (
    'float8_e4m3fn',
    'float8_e5m2',
    'int4',
    'uint4',
)


def can_use_sparse_dot(
    lhs: qarray.MaybeQArray,
    rhs: qarray.MaybeQArray,
    dimension_numbers: jax.lax.DotDimensionNumbers,
) -> bool:
  """Returns whether sparse_dot_general can be used for the given operands."""
  if not (isinstance(rhs, qarray.QArray) and rhs.sparsity_indices is not None):
    return False
  (lhs_ca, rhs_ca), _ = dimension_numbers
  if len(lhs_ca) != 1 or len(rhs_ca) != 1:
    return False
  rhs_sparse_axis = (rhs.sparse_axis or 0) % rhs.ndim
  if rhs_sparse_axis != (rhs_ca[0] % rhs.ndim):
    return False
  lhs_shape = lhs.logical_shape if isinstance(lhs, qarray.QArray) else lhs.shape
  if lhs_shape[lhs_ca[0]] != rhs.logical_shape[rhs_ca[0]]:
    return False
  return rhs.qvalue.dtype.name in _SUPPORTED_SPARSE_DTYPES


def can_emit_native_sparse_dot(
    lhs: qarray.MaybeQArray,
    rhs: qarray.MaybeQArray,
    dimension_numbers: jax.lax.DotDimensionNumbers,
) -> bool:
  """Returns whether the operands satisfy SFS native sparse MXU constraints.

  SFS sparse MXU lowering requires:
  - 1:4 compressed RHS along the single contracting dimension (`stride == 1`).
  - Narrow RHS element type in `{float8_e4m3fn, float8_e5m2, int4, uint4}`.
  - Uncompressed contracting size `K` divisible by 256 (`MxuContractingSize`).
  - Non-contracting LHS batch size divisible by 32 (`SublaneCount * 4`).

  Args:
    lhs: Left-hand side operand.
    rhs: Right-hand side operand.
    dimension_numbers: Dot dimension numbers.

  Returns:
    True if the shapes and dtypes are eligible for SFS hardware sparse MXU.
  """
  if not can_use_sparse_dot(lhs, rhs, dimension_numbers):
    return False
  assert isinstance(rhs, qarray.QArray)
  if rhs.qvalue.dtype.name not in _NATIVE_SFS_SPARSE_DTYPES:
    return False
  (lhs_ca, rhs_ca), (lhs_ba, _) = dimension_numbers
  uncompressed_k = rhs.logical_shape[rhs_ca[0]]
  if uncompressed_k % 256 != 0:
    return False
  lhs_shape = lhs.logical_shape if isinstance(lhs, qarray.QArray) else lhs.shape
  lhs_free_axes = [
      a for a in range(len(lhs_shape)) if a not in lhs_ca and a not in lhs_ba
  ]
  batch_size = 1
  for a in lhs_free_axes:
    batch_size *= lhs_shape[a]
  return batch_size % 32 == 0


def decompress_qarray(array: qarray.QArray) -> qarray.QArray:
  """Decompresses a 1:4 compressed QArray into a dense QArray."""
  qarray.validate_qarray(array)
  if array.sparsity_indices is None:
    return array
  axis = (array.sparse_axis or 0) % array.ndim
  decompressed_qvalue = sparsity.decompress_1_4(
      array.qvalue, array.sparsity_indices, axis=axis
  )
  if array.zero_point is not None:
    mask = sparsity.decompress_1_4(
        jnp.ones_like(array.qvalue, dtype=jnp.bool_),
        array.sparsity_indices,
        axis=axis,
    )
    decompressed_qvalue = qarray.call_with_generic_broadcast(
        lambda x, zp: jnp.where(mask, x, zp),
        decompressed_qvalue,
        array.zero_point,
    )
  return qarray.QArray(
      qvalue=decompressed_qvalue,
      scale=array.scale,
      zero_point=array.zero_point,
      qtype=array.qtype,
  )


def sparse_dot_general(
    lhs: qarray.MaybeQArray,
    rhs: qarray.MaybeQArray,
    dimension_numbers: jax.lax.DotDimensionNumbers,
    precision: jax.lax.PrecisionLike = None,
    preferred_element_type: jax.typing.DTypeLike | None = None,
    *,
    dot_general_fn: Callable[..., jax.Array] | None = None,
) -> jax.Array:
  """Performs dot_general when lhs or rhs is a 1:4 compressed sparse QArray."""
  if isinstance(lhs, qarray.QArray):
    qarray.validate_qarray(lhs)
    if lhs.sparsity_indices is not None:
      lhs = decompress_qarray(lhs)

  if isinstance(rhs, qarray.QArray):
    qarray.validate_qarray(rhs)
    if rhs.sparsity_indices is not None:
      rhs = decompress_qarray(rhs)

  (lhs_ca, rhs_ca), _ = dimension_numbers
  for l_ax, r_ax in zip(lhs_ca, rhs_ca):
    if lhs.shape[l_ax] != rhs.shape[r_ax]:
      raise ValueError(
          f'Contracting dimension mismatch: lhs.shape[{l_ax}] ='
          f' {lhs.shape[l_ax]} vs rhs.shape[{r_ax}] = {rhs.shape[r_ax]}.'
      )

  if dot_general_fn is not None:
    return dot_general_fn(
        lhs,
        rhs,
        dimension_numbers,
        precision=precision,
        preferred_element_type=preferred_element_type,
    )

  mxfp_res = mxfp_dot.mxfp_dot_general(
      lhs,
      rhs,
      dimension_numbers,
      preferred_element_type=preferred_element_type,
  )
  if mxfp_res is not None:
    return mxfp_res

  _, result_type = qarray.get_accumulator_and_result_type(
      lhs, rhs, preferred_element_type=preferred_element_type
  )
  lhs_val = qarray.dequantize(lhs) if isinstance(lhs, qarray.QArray) else lhs
  rhs_val = qarray.dequantize(rhs) if isinstance(rhs, qarray.QArray) else rhs
  return jax.lax.dot_general(
      lhs_val,
      rhs_val,
      dimension_numbers,
      precision=precision,
      preferred_element_type=result_type,
  )
