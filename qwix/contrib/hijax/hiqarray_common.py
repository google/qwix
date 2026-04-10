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

import jax.numpy as jnp


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
      original_dtype: jnp.dtype,
      quantized_dtype: jnp.dtype,
  ):
    """Initializes the quantization metadata for an array.

    Args:
      data_shape: The shape of the original array.
      quant_info: A dictionary of quantization axes to group sizes.
      original_dtype: The dtype of the original array.
      quantized_dtype: The dtype of the quantized array.

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
        original_dtype,
        quantized_dtype,
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

  def __repr__(self):
    quant_axes = self.quant_axes
    group_sizes = self.group_sizes
    orig_dtype = self.dtype
    quant_dtype = self.qtype
    out = (
        f"QuantizationMetadata({quant_axes=}, {group_sizes=}, {orig_dtype=},"
        f" {quant_dtype=})"
    )
    return out
