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
"""Functions for Hadamard Rotations.

The primary entry point is hadamard_rotate_inputs, which constructs a Hadamard
matrix and applies it to the inputs of the weight matrix. This is intended for
use with dot_general with a single reduction dimension.
e.g.
x = jnp.ones((batch_size, in_features))
w = jnp.ones((in_features, out_features))
key = jax.random.key(0)
x_rot, w_rot, _ = hadamard_rotate_inputs(
    x, w, key, row_sign_flip=True, col_sign_flip=True,
    lhs_reduction_dim=1, rhs_reduction_dim=0)
out = jnp.matmul(x_rot, w_rot)
"""

from collections.abc import Callable
import math

import jax
import jax.numpy as jnp


def _create_base_hadamard_matrix(power: int) -> jax.Array:
  """Returns a bfloat16 Hadamard matrix of size 2^power x 2^power."""
  if power == 0:
    return jnp.array([[1]], dtype=jnp.bfloat16)
  if power < 0:
    raise ValueError('Power must be non-negative.')
  had_block = _create_base_hadamard_matrix(power - 1)
  return jnp.block([[had_block, had_block], [had_block, -had_block]])


def _create_hadamard_matrix(
    power: int,
    key: jax.Array | None,
    *,
    row_sign_flip: bool,
    col_sign_flip: bool,
    dtype: jnp.dtype = jnp.bfloat16,  # pyrefly: ignore[bad-function-definition]
) -> tuple[jax.Array, jax.Array | None]:
  """Returns a Hadamard matrix of size 2^power x 2^power with random sign flips.

  Args:
    power: The power of 2 to generate the Hadamard matrix.
    key: The random key to use for sign flips.
    row_sign_flip: Whether to flip the signs of the rows.
    col_sign_flip: Whether to flip the signs of the columns.
    dtype: The dtype of the Hadamard matrix.

  Returns:
    A tuple of the Hadamard matrix and the next random key.
  """
  had = _create_base_hadamard_matrix(power).astype(dtype)
  num_keys = int(row_sign_flip) + int(col_sign_flip) + 1
  if num_keys == 1:
    return had, key
  if key is None:
    raise ValueError(
        'Key must be provided if row_sign_flip or col_sign_flip is True.'
    )
  keys = jax.random.split(key, num_keys)
  key_idx = 0
  had_size = had.shape[0]

  signs = jnp.array([-1, 1])

  if row_sign_flip:
    row_sign_vec = jax.random.choice(
        keys[key_idx],
        signs,
        shape=(had_size,),
        replace=True,
    ).astype(dtype)
    key_idx += 1
    had = had * row_sign_vec[:, None]
  if col_sign_flip:
    col_sign_vec = jax.random.choice(
        keys[key_idx],
        signs,
        shape=(had_size,),
        replace=True,
    ).astype(dtype)
    had = had * col_sign_vec[None, :]
  return had, keys[-1]


def _apply_hadamard_lhs(
    x: jax.Array, had: jax.Array, reduction_dim: int
) -> jax.Array:
  """Applies the lhs Hadamard transform to x along the reduction dimension."""
  lhs_str = ''.join([chr(ord('a') + i) for i in range(x.ndim)])
  rhs_str = f'{chr(ord("a") + reduction_dim)}z'
  out_str = lhs_str.replace(chr(ord('a') + reduction_dim), 'z')
  return jnp.einsum(f'{lhs_str},{rhs_str}->{out_str}', x, had)


def _apply_hadamard_rhs(
    x: jax.Array, had: jax.Array, reduction_dim: int
) -> jax.Array:
  """Applies the rhs Hadamard transform to x along the reduction dimension."""
  had_size = had.shape[0]
  lhs_str = f'{chr(ord("a") + reduction_dim)}z'
  rhs_str = ''.join([chr(ord('a') + i) for i in range(x.ndim)])
  out_str = rhs_str.replace(chr(ord('a') + reduction_dim), 'z')
  return jnp.einsum(f'{lhs_str},{rhs_str}->{out_str}', had, x) / had_size


def _apply_hadamard_lhs_rhs(
    act: jax.Array,
    weight: jax.Array,
    had: jax.Array,
    lhs_reduction_dim: int,
    rhs_reduction_dim: int,
) -> tuple[jax.Array, jax.Array]:
  """Applies the Hadamard matrix to the LHS and RHS of the weight matrix."""
  return _apply_hadamard_lhs(act, had, lhs_reduction_dim), _apply_hadamard_rhs(
      weight, had, rhs_reduction_dim
  )


def _apply_blocked(
    apply_fn: Callable[[jax.Array, jax.Array, int], jax.Array],
    x: jax.Array,
    had: jax.Array,
    reduction_dim: int,
) -> jax.Array:
  """Applies `apply_fn` blockwise along `reduction_dim`."""
  block_size = had.shape[0]
  axis = reduction_dim % x.ndim
  num_blocks = x.shape[axis] // block_size
  block_shape = x.shape[:axis] + (num_blocks, block_size) + x.shape[axis + 1 :]
  return apply_fn(x.reshape(block_shape), had, axis + 1).reshape(x.shape)


def _apply_block_hadamard_lhs_rhs(
    act: jax.Array,
    weight: jax.Array,
    had: jax.Array,
    lhs_reduction_dim: int,
    rhs_reduction_dim: int,
) -> tuple[jax.Array, jax.Array]:
  """Applies a block-diagonal Hadamard matrix along the reduction dimensions."""
  return (
      _apply_blocked(_apply_hadamard_lhs, act, had, lhs_reduction_dim),
      _apply_blocked(_apply_hadamard_rhs, weight, had, rhs_reduction_dim),
  )


def hadamard_rotate_inputs(
    act: jax.Array,
    weight: jax.Array,
    key: jax.Array | None,
    *,
    row_sign_flip: bool,
    col_sign_flip: bool,
    lhs_reduction_dim: int,
    rhs_reduction_dim: int,
    block_size: int | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array | None]:
  """Constructs a Hadamard matrix and applies it to activation and weight inputs.

  When block_size is specified (e.g. block_size=256 for a 256x256 TPU MXU tile),
  applies a block-diagonal Walsh-Hadamard transform of size
  block_size x block_size across the reduction dimension, supporting
  non-power-of-2 reduction dimensions (such as K in {1536, 1792, 3072, 3584,
  7168}) whenever K is divisible by block_size.

  Args:
    act: The activation tensor.
    weight: The weight tensor.
    key: The random key to use for sign flips, or None when sign flips are
      disabled.
    row_sign_flip: Whether to flip the signs of the rows.
    col_sign_flip: Whether to flip the signs of the columns.
    lhs_reduction_dim: The dimension of the activation matrix to apply the
      Hadamard matrix to.
    rhs_reduction_dim: The dimension of the weight matrix to apply the Hadamard
      matrix to.
    block_size: Optional power-of-2 block size (e.g. 256) for block-diagonal
      Hadamard rotation along non-power-of-2 reduction dimensions. If None, uses
      a single full K x K Hadamard matrix and requires the reduction dimension
      size K to be a power of 2.

  Returns:
    A tuple of the rotated activation matrix, the rotated weight matrix, and
    the next random key (or None if key was None).
  """
  if not isinstance(lhs_reduction_dim, int):
    raise ValueError(
        'Expected lhs_reduction_dim to be an integer, but got '
        f'{lhs_reduction_dim}'
    )
  if not isinstance(rhs_reduction_dim, int):
    raise ValueError(
        'Expected rhs_reduction_dim to be an integer, but got '
        f'{rhs_reduction_dim}'
    )
  if act.shape[lhs_reduction_dim] != weight.shape[rhs_reduction_dim]:
    raise ValueError(
        'Expected reduction dimension sizes to be the same, but got '
        f'{act.shape[lhs_reduction_dim]} and {weight.shape[rhs_reduction_dim]}'
    )
  reduction_size = act.shape[lhs_reduction_dim]

  if block_size is not None:
    if (
        not isinstance(block_size, int)
        or isinstance(block_size, bool)
        or block_size <= 0
        or (block_size & (block_size - 1)) != 0
    ):
      raise ValueError(
          'Expected block_size to be a positive power of 2, but got '
          f'{block_size}'
      )
    if reduction_size % block_size != 0:
      raise ValueError(
          f'Expected reduction dimension size {reduction_size} to be divisible '
          f'by block_size {block_size}'
      )
    power = int(math.log2(block_size))
    had, key = _create_hadamard_matrix(
        power, key, row_sign_flip=row_sign_flip, col_sign_flip=col_sign_flip
    )
    act, weight = _apply_block_hadamard_lhs_rhs(
        act, weight, had, lhs_reduction_dim, rhs_reduction_dim
    )
    return act, weight, key

  power = int(math.log2(reduction_size))
  had, key = _create_hadamard_matrix(
      power, key, row_sign_flip=row_sign_flip, col_sign_flip=col_sign_flip
  )
  had_size = had.shape[0]
  if had_size != reduction_size:
    raise ValueError(
        'Expected reduction dimension size to be a power of 2, but got '
        f'{reduction_size}'
    )

  act, weight = _apply_hadamard_lhs_rhs(
      act, weight, had, lhs_reduction_dim, rhs_reduction_dim
  )
  return act, weight, key
