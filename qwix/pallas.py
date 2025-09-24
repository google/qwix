# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Qwix APIs for quantizing Pallas kernels.

This module is usually imported as `import qwix.pallas as qpl` in user code.

To add quantization support to a Pallas kernel, generally users need to

1) Quantize the inputs outside of the kernel via `qpl.quantize`.
2) Instead of calling `pl.pallas_call`, call `qpl.pallas_call`, which supports
   taking QArrays as inputs. `qpl.pallas_call` takes the same block specs as
   `pl.pallas_call` and adjusts them automatically for QArrays.
3) When inside the kernel, use `qpl.einsum` / `qpl.dot_general` / `qpl.dot`
   which take QArrays as inputs and return jax.Array. Alternatively, call
   `dequantize` on the QArray to obtain the dequantized jax.Array.

Note: dot_general / einsum / dot in this module should only be called inside
pallas kernels.
"""

__all__ = [
    'quantize',
    'dot',
    'dot_general',
    'einsum',
    'pallas_call',
    'transform_block_specs_for_tpu',
    'update_block_specs_for_qarray',
    'QArray',
    'dequantize',
    'get_current_rule',
]

# pylint: disable=g-multiple-import, g-importing-member

from collections.abc import Collection, Mapping
import dataclasses
import functools

import jax
from qwix._src.core import qarray
from qwix._src.core.dot_general import loop_dot_general as dot_general
from qwix._src.core.einsum import einsum as _einsum
from qwix._src.core.pallas import pallas_call, transform_block_specs_for_tpu, update_block_specs_for_qarray
from qwix._src.core.qarray import QArray, dequantize
from qwix._src.qconfig import get_current_rule

einsum = functools.partial(_einsum, _qwix_dot_general=dot_general)


def quantize(
    array: jax.Array,
    qtype: jax.typing.DTypeLike,
    *,
    channelwise_axes: Collection[int] = (),
    tiled_axes: Mapping[int, int | float] | None = None,
    calibration_method: str = 'absmax',
    scale_dtype: jax.typing.DTypeLike | None = None,
) -> QArray:
  """Quantize a Jax Array into QArray.

  Args:
    array: The array to quantize.
    qtype: The logical type of the quantized value, e.g. jnp.int8, jnp.int4,
      jnp.float8_e4m3fn, "nf4", etc.
    channelwise_axes: Channelwise axes have individual scales. This has the same
      effect as setting their tile sizes to 1 in tiled_axes.
    tiled_axes: Tiled axes have blockwise scales, aka subchannel quantization.
      The value is a mapping from the tiled axis to the tile size. If the tile
      size is a float, it will be interpreted as "1 / tile_count" and the actual
      tile size will be round(axis_size * tile_size).
    calibration_method: The calibration method to use. The format is
      "<method>[,<args>]", e.g. "absmax" or "fixed,-10,10".
    scale_dtype: The dtype of the scale. If not given, the dtype will be the
      same as the array's dtype.

  Returns:
    The quantized array.
  """
  # A stable API for qarray.quantize()
  how = qarray.HowToQuantize(
      qtype=qtype,
      channelwise_axes=channelwise_axes,
      tiled_axes=tiled_axes or {},
      calibration_method=calibration_method,
  )
  array = qarray.quantize(array, how)
  if scale_dtype is not None:
    array = dataclasses.replace(array, scale=array.scale.astype(scale_dtype))
  return array


def dot(
    a: jax.Array | QArray,
    b: jax.Array | QArray,
    precision: jax.lax.PrecisionLike = None,
    preferred_element_type: jax.typing.DTypeLike | None = None,
    out_sharding=None,
):
  """jnp.dot with QArray support."""
  a_ndim = a.ndim
  b_ndim = b.ndim
  if a_ndim == 0 or b_ndim == 0:
    contract_dims = ((), ())
  elif b_ndim == 1:
    contract_dims = ((a_ndim - 1,), (0,))
  else:
    contract_dims = ((a_ndim - 1,), (b_ndim - 2,))
  return dot_general(
      a,
      b,
      dimension_numbers=(contract_dims, ((), ())),
      precision=precision,
      preferred_element_type=preferred_element_type,
      out_sharding=out_sharding,
  )
