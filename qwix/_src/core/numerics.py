# Copyright 2024 Google LLC
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
"""Numerics for quantization."""

from typing import Callable, Sequence
import jax
from jax import numpy as jnp

# A function that generates noise for stochastic rounding.
# args:
#   shape: The requested shape of the noise to generate.
# returns: Noise as a jax.Array whose shape is broadcastable to the requested
#   shape, and whose dtype can be promoted to fp32 implicitly.
NoiseFn = Callable[[Sequence[int]], jax.Array]
_QUANTIZE_DTYPES = (jnp.bfloat16, jnp.float16, jnp.float32, jnp.float64)


def should_quantize(dtype: jax.typing.DTypeLike) -> bool:
  """Returns True if the dtype should be quantized."""
  return jnp.dtype(dtype) in _QUANTIZE_DTYPES


def can_dequant_on_output(qtype: jax.typing.DTypeLike) -> bool:
  """qtypes that cannot be optimized by computing in quantized types first then dequantize."""
  return qtype not in ['nf4']


def get_asymmetric_bound(qtype: jax.typing.DTypeLike) -> tuple[float, float]:
  """Returns the bound of the given qtype used in asymmetric quantization."""
  try:
    dtype = jnp.dtype(qtype)
  except TypeError as e:
    raise ValueError(f"{qtype} doesn't support asymmetric quantization.") from e

  match dtype:
    case jnp.int8:
      return (-128.0, 127.0)
    case jnp.int4:
      return (-8.0, 7.0)
    case _:
      raise ValueError(f"{qtype} doesn't support asymmetric quantization.")


def get_symmetric_bound(qtype: jax.typing.DTypeLike) -> float:
  """Returns the bound of the given qtype used in symmetric quantization."""
  match qtype:
    case 'nf4':
      return 1.0
    case 'int2' | 'int3' | 'int5' | 'int6' | 'int7':
      # The bound is extended to qmax + 0.5 so that we have a better utilization
      # of the qmax bucket. This is more important for fewer bits of int.
      return 2 ** (int(qtype[3:]) - 1) - 0.5
    case 'mxfp8':
      qtype = jnp.float8_e4m3fn
    case 'mxfp4':
      qtype = jnp.float4_e2m1fn

  # Prevent common misconfigurations, e.g., use bf16 as qtype.
  if jnp.dtype(qtype).itemsize > 1:
    raise ValueError(f'Cannot use {qtype} as qtype.')
  try:
    # TODO(dangyi): Extend the finfo.max bucket for a better utilization.
    return float(jnp.finfo(qtype).max)
  except ValueError:
    # See the comment above for why we add 0.5.
    return jnp.iinfo(qtype).max + 0.5


def convert_to(
    x: jax.Array,
    qtype: jax.typing.DTypeLike,
    noise_fn: NoiseFn | None = None,
) -> jax.Array:
  """Rounds and converts x to the given qtype."""
  # Handles synthetic qtypes.
  match qtype:
    case 'nf4':
      return fp_to_nf4(x)
    case 'int2' | 'int3' | 'int5' | 'int6' | 'int7':
      bits = int(qtype[3:])
      qmin = -(2 ** (bits - 1))
      qmax = 2 ** (bits - 1) - 1
      if bits <= 4:
        qtype = jnp.int4
      elif bits <= 8:
        qtype = jnp.int8
      else:
        raise ValueError(f'Unsupported integer dtype: {qtype}')
      return jnp.round(x).clip(qmin, qmax).astype(qtype)
    case 'mxfp8':
      qtype = jnp.float8_e4m3fn
    case 'mxfp4':
      qtype = jnp.float4_e2m1fn

  # Handles builtin qtypes.
  try:
    finfo = jnp.finfo(qtype)
  except ValueError:
    pass
  else:
    # dtype is a floating point type. No rounding needed, but we need to clip to
    # the range to avoid inf or nan (e.g. for e4m3fn).
    return x.clip(float(finfo.min), float(finfo.max)).astype(qtype)

  # dtype is an integer type. We need to round manually but clipping can be
  # handled by "astype".
  if noise_fn is not None:
    # Stochastic rounding is done in fp32 to avoid bias from bf16, e.g.
    # round(bf16(41)-bf16(0.4)) ~= round(40.5) = 40, rather than
    # round(41-0.4) = round(40.6) = 41.
    x = x.astype(jnp.float32) + noise_fn(x.shape)
  return jnp.round(x).astype(qtype)


def convert_from(x: jax.Array, qtype: jax.typing.DTypeLike) -> jax.Array:
  """Converts x from the given qtype."""
  match qtype:
    case 'nf4':
      return nf4_to_fp(x)
    case _:
      # For native types, no extra conversion is needed. The dtype will be
      # converted during unquantization.
      return x


### NF4


# NB: to work around the issue of calling Jax functions in module-level context.
def get_nf4_buckets() -> jax.Array:
  """Returns the NF4 buckets defined in Appendix E in https://arxiv.org/pdf/2305.14314."""
  nf4_buckets = jnp.array([
      -1.0,
      -0.6961928009986877,
      -0.5250730514526367,
      -0.39491748809814453,
      -0.28444138169288635,
      -0.18477343022823334,
      -0.09105003625154495,
      0.0,
      0.07958029955625534,
      0.16093020141124725,
      0.24611230194568634,
      0.33791524171829224,
      0.44070982933044434,
      0.5626170039176941,
      0.7229568362236023,
      1.0,
  ])
  return nf4_buckets


def fp_to_nf4(array: jax.Array) -> jax.Array:
  """Quantizes an array to a 4-bit NormalFloat representation."""
  nf4_buckets = get_nf4_buckets()

  def bucketize(x):
    bucket = jnp.argmin(jnp.abs(nf4_buckets - x))
    return bucket

  buckets = jax.vmap(bucketize)(array.ravel())
  return buckets.astype(jnp.uint4).reshape(array.shape)  # stored as uint4.


def nf4_to_fp(array: jax.Array) -> jax.Array:
  """Dequantizes a NF4 array to original dtype."""
  nf4_buckets = get_nf4_buckets()

  def reverse_bucketize(x):
    return nf4_buckets[x]

  return jax.vmap(reverse_bucketize)(array.ravel()).reshape(array.shape)
