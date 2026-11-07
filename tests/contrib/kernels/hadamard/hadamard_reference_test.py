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

"""Unit tests for `qwix.contrib.kernels.hadamard.reference`."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np
from qwix._src.core import qarray
from qwix.contrib.kernels.hadamard import reference


class HadamardConstructionTest(parameterized.TestCase):

  def test_sylvester_hadamard_matrix_hardcoded_4x4(self):
    expected = np.array(
        [
            [1, 1, 1, 1],
            [1, -1, 1, -1],
            [1, 1, -1, -1],
            [1, -1, -1, 1],
        ],
        dtype=np.int8,
    )
    np.testing.assert_array_equal(reference.hadamard_matrix(4), expected)

  def test_paley_hadamard_hardcoded_order_4(self):
    # q = 3 == 3 (mod 4): skew-symmetric off-diagonal Paley-I matrix of order 4.
    expected = np.array(
        [
            [1, 1, 1, 1],
            [-1, 1, 1, -1],
            [-1, -1, 1, 1],
            [-1, 1, -1, 1],
        ],
        dtype=np.int8,
    )
    actual = reference.paley_hadamard(3)
    np.testing.assert_array_equal(actual, expected)
    self.assertFalse(np.array_equal(actual, actual.T))
    np.testing.assert_array_equal(
        actual.astype(np.int32) @ actual.T.astype(np.int32),
        4 * np.eye(4, dtype=np.int32),
    )

  @parameterized.parameters(
      128, 256, 512, 768, 1024, 1152, 1536, 2048, 2688, 3072, 3200, 4096
  )
  def test_hadamard_factors_are_orthogonal_pm1(self, n):
    self.assertTrue(reference.has_hadamard_factors(n))
    f1, f2 = reference.hadamard_factors(n)
    self.assertEqual(f1.shape[0] * f2.shape[0], n)
    for f in (f1, f2):
      self.assertEqual(f.dtype, np.int8)
      np.testing.assert_array_equal(np.abs(f), np.ones_like(f))
      gram = f.astype(np.int32) @ f.T.astype(np.int32)
      np.testing.assert_array_equal(
          gram, f.shape[0] * np.eye(f.shape[0], dtype=np.int32)
      )

  @parameterized.parameters(128, 768)
  def test_hadamard_matrix_is_readonly_and_orthogonal(self, n):
    h = reference.hadamard_matrix(n)
    self.assertEqual(h.shape, (n, n))
    self.assertEqual(h.dtype, np.int8)
    self.assertFalse(h.flags.writeable)
    gram = h.astype(np.int32) @ h.T.astype(np.int32)
    np.testing.assert_array_equal(gram, n * np.eye(n, dtype=np.int32))

  @parameterized.parameters(0, -8, 3, 36, 100)
  def test_invalid_hadamard_orders(self, n):
    self.assertFalse(reference.has_hadamard_factors(n))
    with self.assertRaises(ValueError):
      reference.hadamard_factors(n)

  @parameterized.parameters(2, 5, 9, 15)
  def test_paley_rejects_invalid_q(self, q):
    with self.assertRaises(ValueError):
      reference.paley_hadamard(q)


class ReferenceNumericsTest(parameterized.TestCase):

  @parameterized.product(
      dtype=[jnp.bfloat16, jnp.float32],
      qtype=[
          jnp.int8,
          "int8",
          np.dtype("int8"),
          jnp.float8_e4m3fn,
          "float8_e4m3fn",
          jnp.float8_e5m2,
      ],
  )
  def test_quantize_rows_2d_bit_exact_with_qarray_quantize_api(
      self, dtype, qtype
  ):
    x = jax.random.normal(jax.random.key(0), (32, 256), dtype=dtype) * 7.5
    # Include an all-zero row and an extreme-outlier row to exercise tiny_sqrt
    # clamping and integer/fp8 saturation boundaries.
    x = x.at[0].set(0.0)
    x = x.at[1, 0].set(jnp.asarray(127.5, dtype=dtype))

    xq, scale_f32 = reference.quantize_rows_2d(x, qtype)
    expected = qarray.quantize_api(x, qtype, channelwise_axes=(0,))

    np.testing.assert_array_equal(np.asarray(xq), np.asarray(expected.qvalue))
    np.testing.assert_array_equal(
        np.asarray(scale_f32.astype(dtype)), np.asarray(expected.scale)
    )

  @parameterized.parameters(
      (128, None),
      (256, None),
      (768, None),  # Paley-I order 12 * Sylvester 64 (skew-symmetric factor)
      (1152, None),  # Paley-I order 72 * Sylvester 16
      (3200, None),  # Paley-I order 200 * Sylvester 16
      (768, 128),  # Block-Hadamard
      (1024, 256),  # Block-Hadamard
  )
  def test_rht_roundtrip_and_inner_product_invariance(self, k, block_size):
    k0, k1, k2 = jax.random.split(jax.random.key(42), 3)
    x = jax.random.normal(k0, (6, 4, k), dtype=jnp.float32)
    w = jax.random.normal(k1, (6, 4, k), dtype=jnp.float32)
    signs = reference.random_signs(k2, k, dtype=jnp.float32)

    x_rot = reference.hadamard_rotate_reference(
        x, signs=signs, block_size=block_size, axis=-1
    )
    w_rot = reference.hadamard_rotate_reference(
        w, signs=signs, block_size=block_size, axis=-1
    )
    x_rec = reference.hadamard_rotate_reference(
        x_rot, signs=signs, block_size=block_size, axis=-1, inverse=True
    )

    np.testing.assert_allclose(x_rec, x, rtol=1e-5, atol=1e-5)
    dots_orig = jnp.sum(x * w, axis=-1)
    dots_rot = jnp.sum(x_rot * w_rot, axis=-1)
    np.testing.assert_allclose(dots_rot, dots_orig, rtol=1e-5, atol=1e-4)

  def test_rht_middle_axis_matches_last_axis(self):
    k0, k1 = jax.random.split(jax.random.key(7))
    x = jax.random.normal(k0, (3, 768, 5), dtype=jnp.float32)
    signs = reference.random_signs(k1, 768, dtype=jnp.bfloat16)

    actual = reference.hadamard_rotate_reference(x, signs=signs, axis=1)
    expected = jnp.moveaxis(
        reference.hadamard_rotate_reference(
            jnp.moveaxis(x, 1, -1), signs=signs, axis=-1
        ),
        -1,
        1,
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)


if __name__ == "__main__":
  absltest.main()
