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
"""The code in this file implements functionality common to all HiQArrays.

This comes below the QArray abstraction and operates at the Array level.
"""

import dataclasses
from typing import overload

import jax
import jax.numpy as jnp
import qwix.contrib.hijax.hiquant_utils as hq_utils

JSlice = jax._src.indexing.Slice  # pylint: disable=protected-access


@dataclasses.dataclass(frozen=True)
class QuantizationMetadata:
  """This class contains information used to quantize and dequantize an array.

  If zero_point is not None, then we assume that the scale and zero_point have
  the same shape.
  """

  # Information for constructing the shape metadata
  quant_axes: tuple[int, ...]
  group_sizes: tuple[int, ...]

  # Shapes
  data_shape: tuple[int, ...]
  quant_shape: tuple[int, ...]

  # Shapes for performing reductions
  data_compatible_shape: tuple[int, ...]
  quant_compatible_shape: tuple[int, ...]

  # Axes for reductions. Used for internal calculations.
  _tiled_reduction_axes: tuple[int, ...]
  _full_reduction_axes: tuple[int, ...]

  # Dtypes
  dtype: jnp.dtype
  qtype: jnp.dtype

  @classmethod
  def init(
      cls,
      data_shape: tuple[int, ...],
      quant_info: dict[int, int],
      dtype: jnp.dtype,
      qtype: jnp.dtype,
  ):
    """Initializes the quantization metadata for an array.

    Args:
      data_shape: The shape of the original array.
      quant_info: A dictionary of quantization axes to group sizes.
      dtype: The dtype of the original array.
      qtype: The dtype of the quantized array.

    Returns:
      The quantization metadata for the array.
    """
    # Sort the quant_axes and make them non-negative
    l = len(quant_info)
    sorted_quant_info = sorted(
        [(q if q >= 0 else q + l, g) for q, g in quant_info.items()]
    )
    quant_axes = tuple([x[0] for x in sorted_quant_info])
    group_sizes = tuple([x[1] for x in sorted_quant_info])

    if len(quant_axes) != len(set(quant_axes)):
      raise ValueError("Quantization axes must be unique and non-negative.")

    # Construct information about the unquantized array
    (
        data_compatible_shape,
        quant_compatible_shape,
        tiled_reduction_axes,
        full_reduction_axes,
    ) = QuantizationMetadata._get_reduction_shape_and_axes(
        data_shape, quant_axes, group_sizes
    )
    quant_shape = QuantizationMetadata._get_quant_shape(
        data_compatible_shape, tiled_reduction_axes, full_reduction_axes
    )
    return cls(
        quant_axes,
        group_sizes,
        data_shape,
        quant_shape,
        data_compatible_shape,
        quant_compatible_shape,
        tiled_reduction_axes,
        full_reduction_axes,
        dtype,
        qtype,
    )

  @classmethod
  def init_from_qvalue_and_scales(cls, qvalue: jax.Array, scales: jax.Array):
    """Initializes the quantization metadata for an array from the quantized value and scales."""
    dtype = scales.dtype
    qtype = qvalue.dtype
    data_shape = qvalue.shape
    quant_axes = tuple([
        i for i, (x, y) in enumerate(zip(qvalue.shape, scales.shape)) if x != y
    ])
    group_sizes = tuple([
        x // y
        for i, (x, y) in enumerate(zip(qvalue.shape, scales.shape))
        if x != y
    ])
    quant_info = {k: v for k, v in zip(quant_axes, group_sizes)}
    return cls.init(
        data_shape,
        quant_info,
        dtype,
        qtype,
    )

  @staticmethod
  def _get_reduction_shape_and_axes(
      original_shape: tuple[int, ...],
      quant_axes: tuple[int, ...],
      group_sizes: tuple[int, ...],
  ) -> tuple[
      tuple[int, ...],
      tuple[int, ...],
      tuple[int, ...],
      tuple[int, ...],
  ]:
    """This function returns the intermediate shape needed for performing reductions as well as the axes along which to reduce.

    Assumes:
      - quant_axes is sorted and non-negative.
      - group_sizes has the same length as quant_axes.

    Args:
      original_shape: The shape of the original array.
      quant_axes: A tuple of axes to quantize.
      group_sizes: A tuple of group sizes for each quantization axis.

    Returns:
      A tuple containing:
        - data_compatible_shape: The shape of the data array compatible with
          the reduction operations.
        - quant_compatible_shape: The shape of the quantization parameters
          compatible with broadcasting against the data.
        - tiled_reduction_axes: Axes in `data_compatible_shape` that correspond
          to tiled groups and should be reduced.
        - full_reduction_axes: Axes in `data_compatible_shape` that correspond
          to full-dimension groups and should be reduced to size 1.

    Example:
      original_shape = (16, 32, 64)
      quant_axes = (0, 1)
      group_sizes = (2, -1)
      data_compatible_shape = (8, 2, 32, 64)
      quant_compatible_shape = (8, 1, 1, 64)
      tiled_reduction_axes = (1,)
      full_reduction_axes = (1,)
    """
    # intermediate shape used for later reductions
    data_compatible_shape = []
    quant_compatible_shape = []

    # axes which have been tiled and should be reduced away
    tiled_reduction_axes = []

    # axes which have not been tiled and should reduce to shape 1
    full_reduction_axes = tuple([
        q
        for i, (q, g) in enumerate(zip(quant_axes, group_sizes))
        if g == -1 or g == original_shape[i]
    ])

    qi = 0
    for i, xi in enumerate(original_shape):
      if qi >= len(quant_axes):
        data_compatible_shape.append(xi)
        quant_compatible_shape.append(xi)
        continue
      if i == quant_axes[qi]:
        gs = group_sizes[qi]
        if gs == -1 or gs == xi:
          data_compatible_shape.append(xi)
          quant_compatible_shape.append(1)
        else:
          assert xi % gs == 0, "Group size must divide dimension size"
          data_compatible_shape.append(xi // gs)
          data_compatible_shape.append(gs)
          quant_compatible_shape.append(xi // gs)
          quant_compatible_shape.append(1)
          tiled_reduction_axes.append(len(data_compatible_shape) - 1)
        qi += 1
      else:
        data_compatible_shape.append(xi)
        quant_compatible_shape.append(xi)

    return (
        tuple(data_compatible_shape),
        tuple(quant_compatible_shape),
        tuple(tiled_reduction_axes),
        full_reduction_axes,
    )

  @staticmethod
  def _get_quant_shape(
      intermediate_shape: tuple[int, ...],
      tiled_reduction_axes: tuple[int, ...],
      full_reduction_axes: tuple[int, ...],
  ) -> tuple[int, ...]:
    """Returns the shape of the tensor after quantization."""
    tmp_shape = []
    for i, xi in enumerate(intermediate_shape):
      if i in tiled_reduction_axes:
        continue
      else:
        tmp_shape.append(xi)
    out = []

    for i, xi in enumerate(tmp_shape):
      if i in full_reduction_axes:
        out.append(1)
      else:
        out.append(xi)
    return tuple(out)

  def detailed_repr(self):
    quant_axes = self.quant_axes
    group_sizes = self.group_sizes
    orig_dtype = self.dtype
    quant_dtype = self.qtype
    out = (
        f"QuantizationMetadata({quant_axes=}, {group_sizes=}, {orig_dtype=},"
        f" {quant_dtype=})"
    )
    return out

  def __repr__(self):
    quant_type_str = str(jax.core.ShapedArray(self.quant_shape, self.qtype))

    logical_type = jax.core.ShapedArray(self.data_shape, self.dtype)

    # find the type information
    i = quant_type_str.find("[")
    qtype_str = quant_type_str[:i]
    quant_shape_str = quant_type_str[i:]

    out = (
        f"QMD(logical_type={logical_type}, qtype={qtype_str},"
        f" qshape={quant_shape_str})"
    )
    return out

  def jaxpr_repr(self, use_zero_point: bool):
    data_type = jax.core.ShapedArray(self.data_shape, self.qtype)
    scale_type = jax.core.ShapedArray(self.quant_shape, self.dtype)
    zero_point_type = jax.core.ShapedArray(self.quant_shape, self.qtype)
    if use_zero_point:
      out = f"{data_type}, {scale_type}, {zero_point_type}"
    else:
      out = f"{data_type}, {scale_type}, None"
    return out


# Functions that operate on Arrays


def scale_and_round(
    data: jax.Array,
    scale: jax.Array,
    zero_point: jax.Array | None,
    metadata: QuantizationMetadata,
    *,
    lower: float | None = None,
    upper: float | None = None,
    key: jax.Array | None = None,
    differentiable: bool = False,
) -> jax.Array:
  """Quantization operation."""
  # e.g. q = clip(round(x/scale + zero_point), lower, upper)
  start_shape = data.shape
  uqd = data.reshape(metadata.data_compatible_shape)
  sc = scale.reshape(metadata.quant_compatible_shape)
  if zero_point is None:
    out = uqd / sc
  else:
    zp = zero_point.reshape(metadata.quant_compatible_shape)
    out = uqd / sc + zp

  # Stochastic rounding
  if key is not None:
    out += jax.random.uniform(key, shape=out.shape, minval=-0.5, maxval=0.5)

  out = jnp.clip(out, lower, upper)

  if not differentiable:
    if hq_utils.is_integer_dtype(metadata.qtype):
      out = jnp.round(out)
    out = out.astype(metadata.qtype)
  out = out.reshape(start_shape)

  return out


def scale_and_round_inverse(
    data: jax.Array,
    scale: jax.Array,
    zero_point: jax.Array | None,
    metadata: QuantizationMetadata,
) -> jax.Array:
  """Basic dequantization method."""
  # e.g. x = scale * (q - zero_point)
  qd = data.reshape(metadata.data_compatible_shape)
  sc = scale.reshape(metadata.quant_compatible_shape)
  if zero_point is None:
    out = sc * qd.astype(metadata.dtype)
  else:
    zp = zero_point.reshape(metadata.quant_compatible_shape)
    out = sc * (qd.astype(metadata.dtype) - zp)
  return out.reshape(metadata.data_shape)


@overload
def map_slice(slc: slice, big_shape: int, small_shape: int) -> slice:
  ...


@overload
def map_slice(slc: JSlice, big_shape: int, small_shape: int) -> JSlice:
  ...


def map_slice(
    slc: slice | JSlice, big_shape: int, small_shape: int
) -> slice | JSlice:
  """Maps a slice from a big shape to a small shape."""

  use_jax_slice = isinstance(slc, JSlice)

  if use_jax_slice:
    slc = slice(slc.start, slc.start + slc.size, slc.stride)
  assert slc.step is None or slc.step == 1, "Only unit strides supported"

  # We assume that big_shape is a multiple of small_shape
  ratio = big_shape // small_shape

  num_big_elements = slc.stop - slc.start
  if num_big_elements % ratio != 0:
    raise ValueError("Slice size must be a multiple of ratio")
  num_small_elements = num_big_elements // ratio

  new_start = slc.start // ratio
  new_stop = new_start + num_small_elements

  if use_jax_slice:
    return JSlice(new_start, new_stop, 1)
  return slice(new_start, new_stop, 1)


def map_int(i: int, big_shape: int, small_shape: int) -> int:
  """Maps an int from a big shape to a small shape."""
  ratio = big_shape // small_shape
  if i % ratio != 0:
    raise ValueError("Index must be a multiple of ratio")
  return i // ratio


def map_slices_over_shapes(
    slcs: tuple[slice | JSlice, ...],
    big_shapes: tuple[int, ...],
    small_shapes: tuple[int, ...],
) -> tuple[slice | JSlice, ...]:
  """Maps a slice from a big shape to a small shape."""
  assert len(big_shapes) == len(
      small_shapes
  ), "Big and small shapes must have same length"
  return tuple(map(map_slice, slcs, big_shapes, small_shapes))  # pyrefly: ignore


def map_ints_over_shapes(
    ints: tuple[int, ...],
    big_shapes: tuple[int, ...],
    small_shapes: tuple[int, ...],
) -> tuple[int, ...]:
  """Maps an int from a big shape to a small shape."""
  assert len(big_shapes) == len(
      small_shapes
  ), "Big and small shapes must have same length"
  return tuple(map(map_int, ints, big_shapes, small_shapes))  # pyrefly: ignore
