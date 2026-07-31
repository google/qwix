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

import jax
import jax.experimental.hijax as hjx
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
        **quantize_kwargs,
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


def from_hiqarray(
    qarray: hq.HiQArray, *, dequantize_fn, **dequantize_kwargs
) -> jax.Array:
  """Dequantizes a HiQArray."""
  ty = jax.typeof(qarray)
  from_qarray_instance = FromHiQArray(
      ty, dequantize_fn=dequantize_fn, **dequantize_kwargs
  )
  return from_qarray_instance(qarray)
