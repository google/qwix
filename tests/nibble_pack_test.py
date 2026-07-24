"""Tests for nibble_pack/unpack correctness and memory savings."""

import numpy as np
from absl.testing import absltest
import jax.numpy as jnp

from qwix._src.core.nibble_pack import nibble_pack, nibble_unpack


class NibblePackTest(absltest.TestCase):

  def test_roundtrip_exact(self):
    """Pack then unpack recovers original values exactly."""
    rng = np.random.default_rng(42)
    q = jnp.asarray(rng.integers(-8, 8, (64, 128)), jnp.int8)
    packed = nibble_pack(q)
    unpacked = nibble_unpack(packed)
    np.testing.assert_array_equal(np.asarray(unpacked), np.asarray(q))

  def test_memory_reduction(self):
    """Packed array is exactly half the elements (0.5 B/elem vs 1.0)."""
    q = jnp.zeros((1024, 4096), jnp.int8)
    packed = nibble_pack(q)
    self.assertEqual(packed.shape, (1024, 2048))
    self.assertEqual(packed.dtype, jnp.uint8)
    # True memory: packed is half the bytes of the original
    self.assertEqual(packed.nbytes, q.nbytes // 2)

  def test_boundary_values(self):
    """Correctly handles the full int4 range [-8, 7]."""
    vals = jnp.asarray(list(range(-8, 8)) * 2, jnp.int8).reshape(1, 32)
    packed = nibble_pack(vals)
    unpacked = nibble_unpack(packed)
    np.testing.assert_array_equal(np.asarray(unpacked), np.asarray(vals))


if __name__ == '__main__':
  absltest.main()
