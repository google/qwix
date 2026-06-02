"""Utility functions for testing hiqarrays."""

import jax
import jax.numpy as jnp
from qwix.contrib.hijax import hiqarray as hq
from qwix.contrib.hijax import hiqarray_common as hqc


def create_random_qarray(
    orig_shape: tuple[int, ...],
    quant_axes: tuple[int, ...],
    group_sizes: tuple[int, ...],
    key: jax.Array | None = None,
) -> hq.HiQArray:
  """Create a HiQArray with random values."""
  if key is None:
    key = jax.random.key(0)
  k1, k2, k3 = jax.random.split(key, 3)
  data = jax.random.randint(
      k1, orig_shape, minval=-128, maxval=128, dtype=jnp.int8
  )
  quant_info = {k: v for k, v in zip(quant_axes, group_sizes)}
  metadata = hqc.QuantizationMetadata.init(
      orig_shape, quant_info, jnp.float32, jnp.int8
  )
  scale = jax.random.normal(k2, metadata.quant_shape, dtype=jnp.float32)
  zero_point = jax.random.randint(
      k3, metadata.quant_shape, minval=-128, maxval=128, dtype=jnp.int8
  )
  return hq.HiQArray(data, scale, zero_point, metadata)


def create_invertible_qarray(
    orig_shape: tuple[int, ...],
    quant_axes: tuple[int, ...],
    group_sizes: tuple[int, ...],
    use_zero_point: bool,
    key: jax.Array | None = None,
) -> hq.HiQArray:
  """Create a HiQArray that can be inverted exactly with to_qarray and from_qarray."""
  if key is None:
    key = jax.random.key(0)
  data = jax.random.normal(key, orig_shape, dtype=jnp.float32)
  quant_info = {k: v for k, v in zip(quant_axes, group_sizes)}
  metadata = hqc.QuantizationMetadata.init(
      orig_shape, quant_info, jnp.float32, jnp.float32
  )
  scale = jnp.ones(metadata.quant_shape, dtype=jnp.float32)
  zero_point = (
      jnp.zeros(metadata.quant_shape, dtype=jnp.float32)
      if use_zero_point
      else None
  )
  return hq.HiQArray(data, scale, zero_point, metadata)
