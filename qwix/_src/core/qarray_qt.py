# Copyright 2025 Google LLC
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
"""QArray with gradient for custom VJP."""

import dataclasses
from typing import Mapping
import flax.struct
import jax
from qwix._src.core import qarray


@flax.struct.dataclass
class QArrayWithGradient(qarray.QArray):
  """QArray with gradient.

  This dataclass allows us to associate a gradient with the QArray. It's
  achieved by defining an extra attribute `_grad` on the QArray, which has the
  same dtype and the same shape as the unquantized array. In forward pass, the
  `_grad` does nothing and should never be consumed. In backward pass, the
  `_grad` carries the gradient of the whole QArray.

  This approach overcomes the Jax limitation on the gradients, i.e., the
  gradient of a qvalue of int8[128,128] has to be float0[128,128], while the
  gradient of a scale of float32[1,1] has to be float32[1,1]. An alternative
  is to define the QArray as a new Hijax type, which is more complex.
  """

  _grad: jax.Array = flax.struct.field(kw_only=True)


def quantize_with_calibration(
    array: jax.Array,
    qtype: jax.typing.DTypeLike,
    calibration: Mapping[str, jax.Array],
    clip_gradient: bool = False,
) -> QArrayWithGradient:
  """Quantizes an array with calibration with backpropagation support.

  Args:
    array: The array to quantize.
    qtype: The quantized type.
    calibration: The calibration of the array.
    clip_gradient: Whether to clip the straight-through estimator to the
      calibration range, i.e., the gradient outside the calibration range is 0.

  Returns:
    The quantized array with backpropagation support.
  """
  scale, zero_point = qarray.compute_scale_zero_point(calibration, qtype)
  res = qarray.quantize_with_scale_zero_point(array, qtype, scale, zero_point)
  if clip_gradient:
    array = qarray.clip_to_calibration(
        array, calibration, qarray.get_tiled_axes(res)
    )
  # Do not allow gradients on the quantized array to flow back to the input.
  res = jax.lax.stop_gradient(res)
  return QArrayWithGradient(**dataclasses.asdict(res), _grad=array)


@jax.custom_jvp
def dequantize(array: QArrayWithGradient) -> jax.Array:
  """Dequantizes an array."""
  return qarray.dequantize(array)


@dequantize.defjvp
def _dequantize_jvp(primals, tangents):
  return dequantize(*primals), tangents[0]._grad  # pylint: disable=protected-access
