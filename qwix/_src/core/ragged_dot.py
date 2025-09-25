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

"""Quantized jax.lax.ragged_dot."""

import jax
from jax import numpy as jnp
from qwix._src.core import qarray


def _fast_ragged_dot(
    lhs: qarray.MaybeQArray,
    rhs: qarray.MaybeQArray,
    group_sizes: jax.Array,
    precision: jax.lax.PrecisionLike = None,
    preferred_element_type: jax.typing.DTypeLike | None = None,
    group_offset: jax.Array | None = None,
) -> jax.Array:
  """Quantized jax.lax.ragged_dot."""
  if isinstance(lhs, qarray.QArray):
    assert lhs.zero_point is None, 'not supported yet'
    lhs_value = lhs.qvalue
    lhs_scale = lhs.scale
  else:
    lhs_value = lhs
    lhs_scale = None
  if isinstance(rhs, qarray.QArray):
    assert rhs.zero_point is None, 'not supported yet'
    rhs_value = rhs.qvalue
    rhs_scale = rhs.scale
  else:
    rhs_value = rhs
    rhs_scale = None

  preferred_element_type, result_type = qarray.get_accumulator_and_result_type(
      lhs, rhs, preferred_element_type=preferred_element_type
  )

  out = jax.lax.ragged_dot(
      lhs_value,
      rhs_value,
      group_sizes,
      precision=precision,
      preferred_element_type=preferred_element_type,
      group_offset=group_offset,
  )

  # ragged_dot has fixed dimension numbers which makes implementation a lot
  # easier, i.e., lhs: [m, k], rhs: [g, k, n], res: [m, n].
  # TODO(dangyi): support arbitrary dimension numbers.
  if lhs_scale is not None:  # [m, 1]
    lhs_scale = qarray.transpose_array(lhs_scale, (0, None))
    out = qarray.call_with_generic_broadcast(jnp.multiply, out, lhs_scale)
  if rhs_scale is not None:  # [1, 1, n] or [g, 1, n]
    if rhs_scale.shape[0] == 1:
      # It's possible to apply the scale to the out directly.
      rhs_scale = qarray.transpose_array(rhs_scale, (None, 2))
    else:
      # We need another ragged_dot to apply the scale to the out.
      rhs_scale = jax.lax.ragged_dot(
          jnp.ones((out.shape[0], 1), rhs_scale.dtype),
          rhs_scale,
          group_sizes,
          group_offset=group_offset,
      )
    out = qarray.call_with_generic_broadcast(jnp.multiply, out, rhs_scale)

  return out.astype(result_type)


def _slow_ragged_dot(
    lhs: qarray.MaybeQArray,
    rhs: qarray.MaybeQArray,
    group_sizes: jax.Array,
    **kwargs,
) -> jax.Array:
  """Quantized jax.lax.ragged_dot which dequantizes first."""
  if isinstance(lhs, qarray.QArray):
    lhs = qarray.dequantize(lhs)
  if isinstance(rhs, qarray.QArray):
    rhs = qarray.dequantize(rhs)
  return jax.lax.ragged_dot(lhs, rhs, group_sizes, **kwargs)


def ragged_dot(
    lhs: qarray.MaybeQArray,
    rhs: qarray.MaybeQArray,
    group_sizes: jax.Array,
    precision: jax.lax.PrecisionLike = None,
    preferred_element_type: jax.typing.DTypeLike | None = None,
    group_offset: jax.Array | None = None,
) -> jax.Array:
  """Quantized jax.lax.ragged_dot."""
  use_fast_ragged_dot = True

  # fast_ragged_dot does't support channelwise scales on group axis, or tiled
  # scales on contracting axes, or zero_point.
  if isinstance(lhs, qarray.QArray):  # [m, k]
    if lhs.zero_point is not None or lhs.scale.shape[1] > 1:
      use_fast_ragged_dot = False
  if isinstance(rhs, qarray.QArray):  # [g, k, n]
    if (
        rhs.zero_point is not None
        or rhs.scale.shape[0] > 1
        or rhs.scale.shape[1] > 1
    ):
      use_fast_ragged_dot = False

  if use_fast_ragged_dot:
    return _fast_ragged_dot(
        lhs,
        rhs,
        group_sizes,
        precision=precision,
        preferred_element_type=preferred_element_type,
        group_offset=group_offset,
    )
  else:
    return _slow_ragged_dot(
        lhs,
        rhs,
        group_sizes,
        precision=precision,
        preferred_element_type=preferred_element_type,
        group_offset=group_offset,
    )
