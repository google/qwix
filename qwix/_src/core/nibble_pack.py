"""Nibble-packing for int4 QArrays — stores two 4-bit values per uint8 byte.

Without packing, jnp.int4 arrays consume 1.0 bytes/element (JAX/XLA byte-pads sub-byte dtypes).
This module provides pack/unpack that achieve the true 0.5 bytes/element memory reduction —
the intended benefit of int4 quantization.

Usage:
  packed = nibble_pack(q_int4_or_int8)  # [N, K] -> [N, K/2] uint8
  unpacked = nibble_unpack(packed)       # [N, K/2] uint8 -> [N, K] int8 in [-8, 7]
"""

import jax.numpy as jnp


def nibble_pack(q):
  """Pack int4/int8 values (in [-8, 7]) into uint8 (two nibbles per byte).

  Args:
    q: Array with values in [-8, 7], any shape with even last dimension.
       dtype can be int4, int8, or int32.

  Returns:
    uint8 array with last dimension halved. Each byte stores two 4-bit values:
    low nibble = q[..., 0::2], high nibble = q[..., 1::2].
  """
  n = (q.astype(jnp.int32) & 0xF).astype(jnp.uint8)
  lo = n[..., 0::2]
  hi = n[..., 1::2]
  return (lo | (hi << 4)).astype(jnp.uint8)


def nibble_unpack(packed):
  """Unpack uint8 nibble-packed array back to int8 values in [-8, 7].

  Args:
    packed: uint8 array from nibble_pack.

  Returns:
    int8 array with last dimension doubled. Values are sign-extended 4-bit
    (range [-8, 7]).
  """
  lo = packed & 0xF
  hi = (packed >> 4) & 0xF
  # Sign-extend from 4-bit two's complement
  lo = jnp.where(lo >= 8, lo.astype(jnp.int32) - 16, lo.astype(jnp.int32)).astype(jnp.int8)
  hi = jnp.where(hi >= 8, hi.astype(jnp.int32) - 16, hi.astype(jnp.int32)).astype(jnp.int8)
  return jnp.stack([lo, hi], axis=-1).reshape(*packed.shape[:-1], packed.shape[-1] * 2)
