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


"""Conversion functions for HiQArray."""

# pyrefly: ignore-errors

import functools

import jax
import jax.experimental.hijax as hjx
import jax.numpy as jnp
import numpy as np
import qwix.contrib.hijax.hiqarray as hq


class AsHiQArray(hjx.VJPHiPrimitive):
  """Hijax primitive for packing inputs into a HiQArray."""

  def __init__(
      self,
      *,
      qvalue_ty: hjx.ShapedArray,
      scale_ty: hjx.ShapedArray,
      zero_point_ty: hjx.ShapedArray | None,
  ):
    self.in_avals = (qvalue_ty, scale_ty, zero_point_ty)
    self.out_aval = hq.HiQArrayTy(qvalue_ty, scale_ty, zero_point_ty)
    self.params = dict()
    super().__init__()

  def expand(self, data, scale, zero_point):
    return hq.HiQArray(data, scale, zero_point)


def as_hiqarray(
    data: jax.Array,
    scale: jax.Array,
    zero_point: jax.Array | None,
) -> hq.HiQArray:
  """Packs inputs into a HiQArray."""
  to_qarray_instance = AsHiQArray(
      qvalue_ty=jax.typeof(data),
      scale_ty=jax.typeof(scale),
      zero_point_ty=None if zero_point is None else jax.typeof(zero_point),
  )
  return to_qarray_instance(data, scale, zero_point)


class ToHiQArray(hjx.VJPHiPrimitive):
  """Hijax primitive for quantizing inputs to a HiQArray."""

  def __init__(
      self,
      qvalue_ty: hjx.ShapedArray,
      scale_ty: hjx.ShapedArray,
      zero_point_ty: hjx.ShapedArray | None,
      *,
      quantize_fn,
      **quantize_kwargs,
  ):
    in_avals = (qvalue_ty, scale_ty, zero_point_ty)
    self.in_avals = in_avals

    out_qvalue_ty = qvalue_ty.update(dtype=quantize_kwargs["qtype"])
    self.out_aval = hq.HiQArrayTy(out_qvalue_ty, scale_ty, zero_point_ty)
    self.params = dict(
        quantize_fn=quantize_fn,
        quantize_kwargs=quantize_kwargs,
    )
    # For type checking
    self.quantize_fn = quantize_fn
    self.quantize_kwargs = quantize_kwargs
    super().__init__()

  def expand(self, data, scale, zero_point):
    # assert False, f"{self.quantize_kwargs}"
    quantized_data = self.quantize_fn(
        data, scale, zero_point, **self.quantize_kwargs
    )
    return hq.HiQArray(quantized_data, scale, zero_point)

  # Reverse mode ad
  def vjp_fwd(
      self, nzs_in, data: jax.Array, scale: jax.Array, zero_point: jax.Array
  ):
    return self(data, scale, zero_point), None

  def vjp_bwd_retval(self, res, g, /):
    # Use Straight-Through Estimate (STE)
    return (g, None, None)


def to_hiqarray(
    data: jax.Array,
    scale: jax.Array,
    zero_point: jax.Array | None,
    *,
    quantize_fn,
    **quantize_kwargs,
) -> hq.HiQArray:
  """Converts from dequantized inputs to HiQArray."""
  to_qarray_instance = ToHiQArray(
      jax.typeof(data),
      jax.typeof(scale),
      None if zero_point is None else jax.typeof(zero_point),
      quantize_fn=quantize_fn,
      **quantize_kwargs,
  )
  return to_qarray_instance(data, scale, zero_point)


class FromHiQArray(hjx.VJPHiPrimitive):
  """Hijax primitive for dequantizing a HiQArray."""

  def __init__(
      self, in_aval: hq.HiQArrayTy, *, dequantize_fn, **dequantize_kwargs
  ):
    self.in_avals = (in_aval,)
    self.out_aval = in_aval.qvalue_ty.update(dtype=in_aval.dtype)
    self.params = dict(dequantize_fn=dequantize_fn, **dequantize_kwargs)
    # For type checking
    self.dequantize_fn = dequantize_fn
    self.dequantize_kwargs = dequantize_kwargs
    super().__init__()

  def expand(self, qarray: hq.HiQArray):
    dequantized_data = self.dequantize_fn(
        qarray.qvalue, qarray.scale, qarray.zero_point, **self.dequantize_kwargs
    )
    return dequantized_data

  # Reverse mode ad
  def vjp_fwd(self, nzs_in, qarray: hq.HiQArray):
    return self(qarray), None

  def vjp_bwd_retval(self, res, g, /):
    return (g,)


def from_hiqarray(
    qarray: hq.HiQArray, *, dequantize_fn, **dequantize_kwargs
) -> jax.Array:
  """Dequantizes a HiQArray."""
  ty = jax.typeof(qarray)
  from_qarray_instance = FromHiQArray(
      ty, dequantize_fn=dequantize_fn, **dequantize_kwargs
  )
  return from_qarray_instance(qarray)


class PermuteDims(hjx.VJPHiPrimitive):
  """Hijax primitive for permuting dimensions of a HiQArray."""

  def __init__(
      self,
      in_aval: hq.HiQArrayTy,
      axes: tuple[int, ...],
  ):
    self.in_avals = (in_aval,)
    self.out_aval = self._permute_dims_aval(in_aval, axes)
    self.params = dict(axes=axes)
    # For pytype warnings
    self.axes = axes
    super().__init__()

  # Private functions
  @staticmethod
  def _permute_dims_aval(
      in_aval: hq.HiQArrayTy, axes: tuple[int, ...]
  ) -> hq.HiQArrayTy:
    inner_fn = functools.partial(
        jax.eval_shape, lambda x: jnp.permute_dims(x, axes=axes)
    )

    def fn(x):
      sds = inner_fn(x)
      return jax._src.core._sds_aval_mapping(sds)  # pylint: disable=protected-access

    lo_avals_permuted = jax.tree_util.tree_map(fn, in_aval.lo_ty())
    return hq.HiQArrayTy.raise_ty(lo_avals_permuted)

  def expand(self, qarray: hq.HiQArray):
    return hq.HiQArray(
        jnp.permute_dims(qarray.qvalue, self.axes),
        jnp.permute_dims(qarray.scale, self.axes),
        (
            jnp.permute_dims(qarray.zero_point, self.axes)
            if qarray.zero_point is not None
            else None
        ),
    )

  # Reverse mode ad
  def vjp_fwd(self, nzs_in, qarray: hq.HiQArray):
    return permute_dims(qarray, self.axes), None

  def vjp_bwd_retval(self, res, g, /):
    inv_perm = tuple(np.argsort(self.axes))
    return (jnp.permute_dims(g, inv_perm),)


def permute_dims(qarray: hq.HiQArray, axes: tuple[int, ...]) -> hq.HiQArray:
  ty = jax.typeof(qarray)
  permute_axes_instance = PermuteDims(ty, axes)
  return permute_axes_instance(qarray)


def transpose(qarray: hq.HiQArray) -> hq.HiQArray:
  if qarray.ndim < 2:
    raise ValueError(f"Called transpose on HiQArray of shape {qarray.shape}")

  s = list(range(qarray.ndim))
  new_s = s[:-2] + [s[-1], s[-2]]
  return permute_dims(qarray, new_s)
