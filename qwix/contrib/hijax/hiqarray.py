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
"""HiQarray implementation using HiJAX."""

# pyrefly: ignore-errors

import dataclasses

import jax
import jax.experimental.hijax as hjx
import jax.numpy as jnp
from qwix.contrib.hijax import metadata as mtd


@dataclasses.dataclass(frozen=True)
class HiQArray:
  """HiJAX quantized array."""

  # Arrays
  qvalue: jax.Array
  scale: jax.Array
  zero_point: jax.Array | None

  metadata: mtd.QuantizationMetadata | None = None

  shape = property(lambda self: self.qvalue.shape)
  quant_shape = property(lambda self: self.scale.shape)
  dtype = property(lambda self: self.scale.dtype)
  qtype = property(lambda self: self.qvalue.dtype)
  ndim = property(lambda self: self.qvalue.ndim)

  def __post_init__(self):
    # Check that shapes have the same length
    assert self.qvalue.ndim == self.scale.ndim
    if self.zero_point is not None:
      assert self.zero_point.ndim == self.zero_point.ndim

    # Check that dimensions divide correctly
    assert all(d % s == 0 for d, s in zip(self.qvalue.shape, self.scale.shape))
    if self.zero_point is not None:
      assert all(
          d % s == 0
          for d, s in zip(self.zero_point.shape, self.zero_point.shape)
      )


# QArray type
@dataclasses.dataclass(frozen=True)
class HiQArrayTy(hjx.HiType):
  """The HiJAX type for a HiQArray."""

  qvalue_ty: hjx.ShapedArray
  scale_ty: hjx.ShapedArray
  zero_point_ty: hjx.ShapedArray | None

  metadata: mtd.QuantizationMetadata | None = None

  data_shape = property(lambda self: self.qvalue_ty.shape)
  quant_shape = property(lambda self: self.scale_ty.shape)
  dtype = property(lambda self: self.scale_ty.dtype)
  qtype = property(lambda self: self.qvalue_ty.dtype)
  use_zero_point = property(lambda self: self.zero_point_ty is not None)
  shape = property(lambda self: self.data_shape)
  ndim = property(lambda self: len(self.qvalue_ty.shape))
  array_ty = property(lambda self: self.qvalue_ty.update(dtype=self.dtype))

  def lo_ty(self) -> list[hjx.ShapedArray | None]:
    out = [self.qvalue_ty, self.scale_ty]
    if self.use_zero_point:
      out.append(self.zero_point_ty)
    return out

  # Functions for raising to hijax and lowering to lojax
  def lower_val(self, hi_val: HiQArray) -> list[jax.Array]:
    if hi_val.zero_point is None:
      return [hi_val.qvalue, hi_val.scale]
    else:
      return [hi_val.qvalue, hi_val.scale, hi_val.zero_point]

  def raise_val(
      self,
      data: jax.Array,
      scale: jax.Array,
      zero_point: jax.Array | None = None,
  ) -> HiQArray:
    return HiQArray(data, scale, zero_point)

  # Conveniece function for raising the type of inputs
  @staticmethod
  def raise_ty(avals: list[hjx.ShapedArray]):
    qvalue_aval, scale_aval = avals[0], avals[1]
    zero_point_aval = None if len(avals) < 3 else avals[2]
    return HiQArrayTy(qvalue_aval, scale_aval, zero_point_aval)

  # Autodiff functions
  def to_tangent_aval(self):
    return hjx.ShapedArray(self.data_shape, self.dtype)

  def to_ct_aval(self):
    return hjx.ShapedArray(self.data_shape, self.dtype)

  def vspace_zero(self):
    return jnp.zeros(self.data_shape, self.dtype)


hjx.register_hitype(
    HiQArray,
    lambda q: HiQArrayTy(
        jax.typeof(q.qvalue),
        jax.typeof(q.scale),
        None if q.zero_point is None else jax.typeof(q.zero_point),
        None,
    ),
)
