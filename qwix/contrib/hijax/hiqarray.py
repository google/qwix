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
"""Assumes that group size = -1 means full tensor quantization along that axis.

We assume that the cotangent type of a QArray is an Array.
We could always add a function that tells us how to do the backward stuff.
For example, if we want to quantize the backwards on a matmul we could do that.
"""

import dataclasses

import jax
import jax.experimental.hijax as hjx
import jax.experimental.pallas as pl
import jax.numpy as jnp
import numpy as np
import qwix.contrib.hijax.hiqarray_common as hqc

TransformedRef = jax._src.state.types.TransformedRef  # pylint: disable=protected-access
NDIndexer = jax._src.state.indexing.NDIndexer  # pylint: disable=protected-access


@dataclasses.dataclass(frozen=True)
class HiQArray:
  """HiQArray class."""

  # Arrays
  qvalue: jax.Array
  scale: jax.Array
  zero_point: jax.Array | None

  # Quantization metadata
  metadata: hqc.QuantizationMetadata

  @property
  def shape(self):
    return self.qvalue.shape

  @property
  def dtype(self):
    return self.scale.dtype

  @property
  def qtype(self):
    return self.qvalue.dtype


# Generic functions for QArrays
# These are utility functions and should not be called outside of this file
def _qarray_lo_ty(
    qmd: hqc.QuantizationMetadata, use_zero_point: bool
) -> list[hjx.ShapedArray]:
  """Returns the lo_ty of a QArray."""
  out = [
      hjx.ShapedArray(qmd.data_shape, qmd.qtype),
      hjx.ShapedArray(qmd.quant_shape, qmd.dtype),
  ]
  if use_zero_point:
    out.append(hjx.ShapedArray(qmd.quant_shape, qmd.qtype))
  return out


def _qarray_lower_val(hi_val: HiQArray) -> list[jax.Array]:
  """Returns the lo_val of a QArray."""
  if hi_val.zero_point is None:
    return [hi_val.qvalue, hi_val.scale]
  else:
    return [hi_val.qvalue, hi_val.scale, hi_val.zero_point]


def _qarray_lower_block_spec(
    metadata: hqc.QuantizationMetadata,
    use_zero_point: bool,
    block_spec: pl.BlockSpec,
) -> (
    tuple[pl.BlockSpec, pl.BlockSpec]
    | tuple[pl.BlockSpec, pl.BlockSpec, pl.BlockSpec]
):
  """Returns the block spec of a QArray."""
  data_block_spec = block_spec
  block_spec_size = tuple(
      map(lambda b: getattr(b, "block_size", b), block_spec.block_shape)  # pyrefly: ignore
  )  # pyrefly: ignore
  scale_block_shapes = hqc.map_ints_over_shapes(
      block_spec_size, metadata.data_shape, metadata.quant_shape  # pyrefly: ignore
  )  # pyrefly: ignore

  scale_block_spec = pl.BlockSpec(scale_block_shapes, block_spec.index_map)

  if use_zero_point:
    zero_point_block_spec = scale_block_spec
    return data_block_spec, scale_block_spec, zero_point_block_spec
  else:
    return data_block_spec, scale_block_spec


# QArray type
@dataclasses.dataclass(frozen=True)
class HiQArrayTy(hjx.HiType):
  """HiQArray type."""

  metadata: hqc.QuantizationMetadata
  use_zero_point: bool
  shape = property(lambda self: self.metadata.data_shape)
  dtype = property(lambda self: self.metadata.dtype)

  def update(self, *, shape: tuple[int, ...] | None = None, **kwargs):
    # Used for refs
    if len(kwargs) > 0:
      assert False, f"Unsupported kwargs: {kwargs=}"

    if shape is None:
      return self
    if len(shape) == len(self.metadata.data_shape) and all(
        s == q for s, q in zip(shape, self.metadata.data_shape)
    ):
      return self

    # Compute new metadata
    scale_shape = hqc.map_ints_over_shapes(
        shape, self.shape, self.metadata.quant_shape
    )
    data_shape_dtype_struct = jax.ShapeDtypeStruct(shape, self.metadata.qtype)
    scale_shape_dtype_struct = jax.ShapeDtypeStruct(
        scale_shape, self.metadata.dtype
    )
    new_qmd = hqc.QuantizationMetadata.init_from_qvalue_and_scales(
        data_shape_dtype_struct, scale_shape_dtype_struct  # pyrefly: ignore
    )  # pyrefly: ignore

    return HiQArrayTy(new_qmd, self.use_zero_point)

  def lo_ty(self) -> list[hjx.ShapedArray]:  # pyrefly: ignore
    return _qarray_lo_ty(self.metadata, self.use_zero_point)  # pyrefly: ignore

  # Functions for raising to hijax and lowering to lojax
  def lower_val(self, hi_val: HiQArray) -> list[jax.Array]:
    return _qarray_lower_val(hi_val)

  def raise_val(  # pyrefly: ignore
      self, data: jax.Array, scale: jax.Array, zero_point: jax.Array | None
  ) -> HiQArray:
    return HiQArray(data, scale, zero_point, self.metadata)  # pyrefly: ignore

  # Autodiff functions
  def to_tangent_aval(self):
    return hjx.ShapedArray(self.metadata.data_shape, self.metadata.dtype)

  def to_ct_aval(self):
    return hjx.ShapedArray(self.metadata.data_shape, self.metadata.dtype)

  def vspace_zero(self):
    return jnp.zeros(self.metadata.data_shape, self.metadata.dtype)

  # Lowering for block specs
  def lower_block_spec(self, block_spec: pl.BlockSpec):
    return _qarray_lower_block_spec(
        self.metadata, self.use_zero_point, block_spec
    )

  # Ref handling
  def ref_get_abstract_eval(self, ref_aval, *args, tree):
    arr_aval = jax.core.ShapedArray(self.shape, self.metadata.dtype)
    updated_ref = ref_aval.update(inner_aval=arr_aval)
    out, effects = jax._src.state.primitives.get_p.abstract_eval(  # pylint: disable=protected-access
        updated_ref, *args, tree=tree
    )
    assert isinstance(out, jax.core.ShapedArray)

    # Do some math to figure out the scale shapes
    scale_shape = hqc.map_ints_over_shapes(
        out.shape, self.shape, self.metadata.quant_shape
    )
    data_shape_dtype_struct = jax.ShapeDtypeStruct(
        out.shape, self.metadata.qtype
    )
    scale_shape_dtype_struct = jax.ShapeDtypeStruct(
        scale_shape, self.metadata.dtype
    )
    new_qmd = hqc.QuantizationMetadata.init_from_qvalue_and_scales(
        data_shape_dtype_struct, scale_shape_dtype_struct  # pyrefly: ignore
    )  # pyrefly: ignore
    return HiQArrayTy(new_qmd, self.use_zero_point), effects

  def ref_swap_abstract_eval(self, ref_aval, val_aval, *args, tree):
    arr_aval = jax.core.ShapedArray(self.shape, self.metadata.dtype)
    val_arr_aval = jax.core.ShapedArray(val_aval.shape, self.metadata.dtype)
    updated_ref = ref_aval.update(inner_aval=arr_aval)
    out_aval, effects = jax._src.state.primitives.swap_p.abstract_eval(  # pylint: disable=protected-access
        updated_ref, val_arr_aval, *args, tree=tree
    )
    assert isinstance(out_aval, jax.core.ShapedArray)

    # Do some math to figure out the scale shapes
    scale_shape = hqc.map_ints_over_shapes(
        out_aval.shape, self.shape, self.metadata.quant_shape
    )
    data_shape_dtype_struct = jax.ShapeDtypeStruct(
        out_aval.shape, self.metadata.qtype
    )
    scale_shape_dtype_struct = jax.ShapeDtypeStruct(
        scale_shape, self.metadata.dtype
    )
    new_qmd = hqc.QuantizationMetadata.init_from_qvalue_and_scales(
        data_shape_dtype_struct, scale_shape_dtype_struct  # pyrefly: ignore
    )  # pyrefly: ignore
    return HiQArrayTy(new_qmd, self.use_zero_point), effects

  def ref_get_to_lojax(self, ref: TransformedRef | jax.Ref, idx: NDIndexer):

    if isinstance(ref, TransformedRef):
      if ref.transforms:
        raise NotImplementedError(ref)
      ref = ref.ref
    # Unpack Ref type
    unpacked_ref = ref._refs  # pylint: disable=protected-access

    data_slices = idx.indices
    scale_slices = hqc.map_slices_over_shapes(
        idx.indices, self.shape, self.metadata.quant_shape  # pyrefly: ignore
    )  # pyrefly: ignore
    lovals = self.lower_val(unpacked_ref)
    slices = [data_slices, scale_slices, scale_slices]

    # If zero_point isn't used, then the zip won't hit it
    outs = [out.get()[*slcs] for out, slcs in zip(lovals, slices)]  # pyrefly: ignore
    return self.raise_val(*outs)

  def ref_swap_to_lojax(
      self, ref: TransformedRef | jax.Ref, val: jax.Array, idx: NDIndexer
  ):

    if isinstance(ref, TransformedRef):
      if ref.transforms:
        raise NotImplementedError(ref)
      ref = ref.ref
    # Unpack Ref type
    unpacked_ref = ref._refs  # pylint: disable=protected-access

    data_slices = idx.indices
    scale_slices = hqc.map_slices_over_shapes(
        idx.indices, self.shape, self.metadata.quant_shape  # pyrefly: ignore
    )  # pyrefly: ignore
    slices = [data_slices, scale_slices, scale_slices]

    outs = [
        out.swap(val)[*slcs]  # pytype: disable=attribute-error
        for out, val, slcs in zip(
            self.lower_val(unpacked_ref), self.lower_val(val), slices  # pytype: disable=wrong-arg-types
        )
    ]
    return self.raise_val(*outs)

  def __repr__(self):
    return f"HiQArrayTy({self.metadata.jaxpr_repr(self.use_zero_point)})"


hjx.register_hitype(
    HiQArray, lambda q: HiQArrayTy(q.metadata, q.zero_point is not None)
)


class ToHiQArray(hjx.VJPHiPrimitive):
  """Hijax primitive to convert data, scale, and zero_point to a HiQArray."""

  def __init__(
      self,
      metadata: hqc.QuantizationMetadata,
      key: jax.Array | None,
      use_zero_point: bool,
      use_lower: bool,
      use_upper: bool,
  ):
    data_scale_avals = (
        hjx.ShapedArray(metadata.data_shape, metadata.dtype),
        hjx.ShapedArray(metadata.quant_shape, metadata.dtype),
    )
    zp_aval = (
        hjx.ShapedArray(metadata.quant_shape, metadata.qtype)
        if use_zero_point
        else None
    )
    lower_aval = (
        hjx.ShapedArray(metadata.quant_shape, metadata.dtype)
        if use_lower
        else None
    )
    upper_aval = (
        hjx.ShapedArray(metadata.quant_shape, metadata.dtype)
        if use_upper
        else None
    )
    self.in_avals = (
        data_scale_avals + (zp_aval,) + (lower_aval,) + (upper_aval,)
    )
    self.out_aval = HiQArrayTy(metadata, use_zero_point)
    self.params = dict(
        metadata=metadata,
        key=key,
        use_zero_point=use_zero_point,
        use_lower=use_lower,
        use_upper=use_upper,
    )
    # For pytype warnings - redundant since super() will initialize these.
    self.metadata = metadata
    self.key = key
    self.use_zero_point = use_zero_point
    self.use_lower = use_lower
    self.use_upper = use_upper

    super().__init__()

  def expand(self, data, scale, zero_point, lower, upper):  # pyrefly: ignore
    quantized_data = hqc.scale_and_round(
        data,
        scale,
        zero_point,
        self.metadata,
        key=self.key,
        lower=lower,
        upper=upper,
    )
    return HiQArray(quantized_data, scale, zero_point, self.metadata)

  # Reverse mode ad
  def vjp_fwd(self, nzs_in, data, scale, zero_point, lower, upper):  # pyrefly: ignore
    return to_hiqarray(
        data,
        scale,
        zero_point,
        self.metadata,
        key=self.key,
        lower=lower,
        upper=upper,
    ), (data, scale, zero_point, lower, upper)

  def vjp_bwd_retval(self, res, outgrad, /):
    # Classic API: returns values instead of using accumulators
    fn = lambda *vals: hqc.scale_and_round(
        vals[0],
        vals[1],
        vals[2],
        self.metadata,
        lower=vals[3],
        upper=vals[4],
        key=self.key,
        differentiable=True,
    )
    _, vjp_fn = jax.vjp(fn, *res)
    vjp = vjp_fn(outgrad)
    return vjp

  def transpose(self, ct, *maybe_accums):
    pass

  def batch_dim_rule(self, axis_data, dims, /):
    raise NotImplementedError(
        f"for vmap support, subclass {type(self)} must "
        "implement `batch` or `batch_dim_rule`"
    )


def to_hiqarray(
    data: jax.Array,
    scale: jax.Array,
    zero_point: jax.Array | None,
    metadata: hqc.QuantizationMetadata,
    key: jax.Array | None = None,
    lower=None,
    upper=None,
) -> HiQArray:
  """Converts data, scale, and zero_point to a HiQArray."""
  to_qarray_instance = ToHiQArray(
      metadata,
      key,
      zero_point is not None,
      lower is not None,
      upper is not None,
  )
  return to_qarray_instance(data, scale, zero_point, lower, upper)


class FromHiQArray(hjx.VJPHiPrimitive):
  """Hijax primitive to convert a HiQArray to a dequantized array."""

  def __init__(self, metadata: hqc.QuantizationMetadata, use_zero_point: bool):
    self.in_avals = (HiQArrayTy(metadata, use_zero_point),)
    self.out_aval = hjx.ShapedArray(metadata.data_shape, metadata.dtype)
    self.params = dict(metadata=metadata, use_zero_point=use_zero_point)
    # For pytype warnings - redundant since super() will initialize these.
    self.metadata = metadata
    self.use_zero_point = use_zero_point
    super().__init__()

  def expand(self, qarray: HiQArray):  # pyrefly: ignore
    dequantized_data = hqc.scale_and_round_inverse(  # pyrefly: ignore
        qarray.qvalue, qarray.scale, qarray.zero_point, self.metadata
    )
    return dequantized_data

  # Reverse mode ad
  def vjp_fwd(self, nzs_in, qarray: HiQArray):  # pyrefly: ignore
    return from_hiqarray(qarray), (
        qarray.qvalue,
        qarray.scale,
        qarray.zero_point,
    )  # pyrefly: ignore

  def vjp_bwd_retval(self, res, outgrad, /):
    # Classic API: returns values instead of using accumulators
    fn = lambda x: hqc.scale_and_round_inverse(x, res[1], res[2], self.metadata)
    _, vjp_fn = jax.vjp(fn, res[0].astype(self.metadata.dtype))
    vjp = vjp_fn(outgrad)
    return vjp

  def transpose(self, ct, *maybe_accums):
    pass

  def batch_dim_rule(self, axis_data, dims, /):
    raise NotImplementedError(
        f"for vmap support, subclass {type(self)} must "
        "implement `batch` or `batch_dim_rule`"
    )


def from_hiqarray(qarray: HiQArray) -> jax.Array:
  """Converts a HiQArray to a dequantized array."""
  ty = jax.typeof(qarray)
  from_qarray_instance = FromHiQArray(qarray.metadata, ty.use_zero_point)
  return from_qarray_instance(qarray)


class PermuteAxes(hjx.VJPHiPrimitive):
  """Hijax primitive to permute the axes of a HiQArray."""

  def __init__(
      self,
      metadata: hqc.QuantizationMetadata,
      use_zero_point: bool,
      axes: tuple[int, ...],
  ):
    self.in_avals = (HiQArrayTy(metadata, use_zero_point),)
    self.out_aval = HiQArrayTy(
        PermuteAxes._permute_axes_metadata(metadata, axes), use_zero_point
    )
    self.params = dict(
        metadata=metadata,
        use_zero_point=use_zero_point,
        axes=axes,
    )
    # For pytype warnings - redundant since super() will initialize these.
    self.metadata = metadata
    self.use_zero_point = use_zero_point
    self.axes = axes
    super().__init__()

  # Private functions
  @staticmethod
  def _permute_axes_metadata(
      metadata: hqc.QuantizationMetadata, axes: tuple[int, ...]
  ) -> hqc.QuantizationMetadata:
    in_data_shape_dtype_struct = jax.ShapeDtypeStruct(
        metadata.data_shape, metadata.qtype
    )
    in_scale_shape_dtype_struct = jax.ShapeDtypeStruct(
        metadata.quant_shape, metadata.dtype
    )
    new_data_shape_dtype_struct = jax.eval_shape(
        lambda x: jnp.permute_dims(x, axes), in_data_shape_dtype_struct
    )
    new_scale_shape_dtype_struct = jax.eval_shape(
        lambda x: jnp.permute_dims(x, axes), in_scale_shape_dtype_struct
    )
    new_qmd = hqc.QuantizationMetadata.init_from_qvalue_and_scales(
        new_data_shape_dtype_struct, new_scale_shape_dtype_struct
    )
    return new_qmd

  def expand(self, qarray: HiQArray):  # pyrefly: ignore
    return HiQArray(
        jnp.permute_dims(qarray.qvalue, self.axes),
        jnp.permute_dims(qarray.scale, self.axes),
        jnp.permute_dims(qarray.zero_point, self.axes)
        if qarray.zero_point is not None
        else None,
        self.metadata,
    )  # pyrefly: ignore

  # Reverse mode ad
  def vjp_fwd(self, nzs_in, qarray: HiQArray):  # pyrefly: ignore
    return permute_axes(qarray, self.axes), tuple()  # pyrefly: ignore

  def vjp_bwd_retval(self, res, outgrad, /):
    # Classic API: returns values instead of using accumulators
    inv_perm = tuple(np.argsort(self.axes))
    return (jnp.permute_dims(outgrad, inv_perm),)

  def transpose(self, ct, *maybe_accums):
    pass

  def batch_dim_rule(self, axis_data, dims, /):
    raise NotImplementedError(
        f"for vmap support, subclass {type(self)} must "
        "implement `batch` or `batch_dim_rule`"
    )


def permute_axes(qarray: HiQArray, axes: tuple[int, ...]) -> HiQArray:
  """Permutes the axes of a HiQArray."""
  ty = jax.typeof(qarray)
  permute_axes_instance = PermuteAxes(qarray.metadata, ty.use_zero_point, axes)
  return permute_axes_instance(qarray)
