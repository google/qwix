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
"""Stochastic rounding utilities."""

from typing import Callable, Sequence
import jax
import jax.numpy as jnp


def low_bit_uniform_noise(
    key: jax.Array,
    shape: tuple[int, ...],
) -> jax.Array:
  """Random float32 numbers in nearly (-0.5, 0.5) of shape `shape`.

  Shifts the bits to the leading mantissa bit of the `f32` representation,
  and binary ORs with `1.0`, resulting in a normal number between
  `[1.0, 1.0 + (x - 1) / x]`, where `x = 2 ** rand_bits`.

  From there we adjust to `[-0.5 + 1/(2x), 0.5 - 1/(2x)]`, but don't bother
  with scaling up (converges fine for 8bits+).

  Args:
    key: Random key.
    shape: Shape of the noise.

  Returns:
    A random float32 array of shape `shape` in range (-0.5, 0.5).
  """
  nbits = 8
  rand_u8 = jax.random.bits(key, shape, dtype=jnp.uint8)

  dtype = jnp.float32
  nmant = jnp.finfo(dtype).nmant
  u_type = jnp.uint32

  rand = rand_u8.astype(u_type)
  if nmant > nbits:
    r_bitpattern = jnp.left_shift(rand, nmant - nbits)
  else:
    r_bitpattern = jnp.right_shift(rand, nbits - nmant)

  one_bitpattern = jax.lax.bitcast_convert_type(
      jnp.array(1.0, dtype=dtype), u_type
  )
  r_bitpattern = jnp.bitwise_or(one_bitpattern, r_bitpattern)

  rand_floats = jax.lax.bitcast_convert_type(r_bitpattern, dtype)

  shift = 2 ** (-1 - nbits)
  centered = rand_floats - (1.5 - shift)
  return centered


def uniform_noise(
    key: jax.Array,
    shape: tuple[int, ...],
) -> jax.Array:
  """Uniform noise."""
  return jax.random.uniform(key, shape) - 0.5


def get_noise_fn(
    method: str,
    key: jax.Array,
    channelwise_noise_axes: Sequence[int] = (0,),
) -> Callable[[tuple[int, ...]], jax.Array]:
  """Returns a noise function for stochastic rounding."""
  if method == 'uniform':
    fn = uniform_noise
  elif method == 'low_bit_uniform':
    fn = low_bit_uniform_noise
  else:
    raise ValueError(f'Unsupported stochastic rounding method: {method}')

  def noise_fn(shape: tuple[int, ...]) -> jax.Array:
    # Apply channelwise_noise_axes to get the noise shape. This significantly
    # reduces the overhead of creating a full noise array.
    noise_shape = tuple(
        dim if axis in channelwise_noise_axes else 1
        for axis, dim in enumerate(shape)
    )
    return fn(key, noise_shape)

  return noise_fn
