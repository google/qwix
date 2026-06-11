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
import itertools
from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax import numpy as jnp
from qwix.contrib import hadamard_rot


class HadamardRotTest(parameterized.TestCase):

  @parameterized.parameters(
      itertools.product(
          list(range(5, 8)),
          [False, True],
          [False, True],
      ),
  )
  def test_hadamard(self, power, row_sign_flip, col_sign_flip):
    """Tests that HH^T = dI."""
    key = jax.random.key(0)
    had = hadamard_rot.hadamard(
        power, key, row_sign_flip=row_sign_flip, col_sign_flip=col_sign_flip
    )[0]

    had_size = had.shape[0]
    identity = jnp.eye(had_size)
    product = jnp.dot(had, had.T)
    self.assertTrue(jnp.allclose(product, had_size * identity))

  @parameterized.parameters(0, 1, 2)
  def test_apply_hadamard_lhs(self, idx):
    """Tests that apply_hadamard_lhs works as expected."""
    powers = (2, 3, 4)
    lhs = jnp.ones(tuple(map(lambda x: 2**x, powers)))

    had = hadamard_rot.hadamard(
        powers[idx],
        jax.random.key(0),
        row_sign_flip=True,
        col_sign_flip=False,
    )[0]
    lhs_rotated = hadamard_rot.apply_hadamard_lhs(lhs, had, idx)

    self.assertEqual(lhs.shape, lhs_rotated.shape)

  @parameterized.parameters(0, 1, 2)
  def test_apply_hadamard_rhs(self, idx):
    """Tests that apply_hadamard_rhs works as expected."""
    powers = (2, 3, 4)
    rhs = jnp.ones(tuple(map(lambda x: 2**x, powers)))

    had = hadamard_rot.hadamard(
        powers[idx],
        jax.random.key(0),
        row_sign_flip=False,
        col_sign_flip=True,
    )[0]
    rhs_rotated = hadamard_rot.apply_hadamard_rhs(rhs, had, idx)

    self.assertEqual(rhs.shape, rhs_rotated.shape)

  def test_hadamard_rotate_multiply_identity(self):
    """Tests that (IH)(H^T I / d) = I."""
    lhs = jnp.eye(16)
    rhs = jnp.eye(16)

    lhs_rotated, rhs_rotated, _ = hadamard_rot.hadamard_rotate_inputs(
        lhs,
        rhs,
        jax.random.key(0),
        row_sign_flip=True,
        col_sign_flip=True,
        lhs_reduction_dim=1,
        rhs_reduction_dim=0,
    )
    out = jnp.matmul(lhs_rotated, rhs_rotated)
    self.assertTrue(jnp.allclose(out, jnp.eye(16), rtol=1e-4, atol=1e-4))

  def test_hadamard_rotate_matmul(self):
    """Tests that (xH)(H^T w) = x w."""
    key = jax.random.key(0)
    k1, k2, k3 = jax.random.split(key, 3)
    lhs = jax.random.normal(k1, (4, 8))
    rhs = jax.random.normal(k2, (8, 16))

    lhs_rotated, rhs_rotated, _ = hadamard_rot.hadamard_rotate_inputs(
        lhs,
        rhs,
        k3,
        row_sign_flip=True,
        col_sign_flip=True,
        lhs_reduction_dim=1,
        rhs_reduction_dim=0,
    )
    out_rotated = jnp.matmul(lhs_rotated, rhs_rotated)
    out = jnp.matmul(lhs, rhs)
    self.assertTrue(jnp.allclose(out, out_rotated, rtol=1e-4, atol=1e-4))

  @parameterized.parameters(
      itertools.product(
          (1, 2),
          [False, True],
          [False, True],
      )
  )
  def test_hadamard_rotate_multiply(self, idx, row_sign_flip, col_sign_flip):
    """Tests that (xH)(H^T w / d) = x w along different reduction axes."""
    key = jax.random.key(0)
    k1, k2, k3 = jax.random.split(key, 3)
    lhs = jax.random.normal(k1, (4, 8, 16))
    rhs = jax.random.normal(k2, (8, 16, 32))

    def op(x, y):
      if idx == 1:
        return jnp.einsum("arb,rcd->abcd", x, y)
      if idx == 2:
        return jnp.einsum("abr,crd->abcd", x, y)
      else:
        raise ValueError(f"Unsupported index: {idx}")

    lhs_rotated, rhs_rotated, _ = hadamard_rot.hadamard_rotate_inputs(
        lhs,
        rhs,
        k3,
        row_sign_flip=row_sign_flip,
        col_sign_flip=col_sign_flip,
        lhs_reduction_dim=idx,
        rhs_reduction_dim=idx - 1,
    )
    out_rotated = op(lhs_rotated, rhs_rotated)
    out = op(lhs, rhs)
    self.assertTrue(jnp.allclose(out, out_rotated, rtol=1e-4, atol=1e-4))

  def test_hadmard_matrix_construction_errors(self):
    # negative power
    with self.assertRaises(ValueError):
      hadamard_rot.hadamard(
          -1,
          None,
          row_sign_flip=False,
          col_sign_flip=False,
      )

    # no key but row_sign_flip is True
    with self.assertRaises(ValueError):
      hadamard_rot.hadamard(
          2,
          None,
          row_sign_flip=True,
          col_sign_flip=False,
      )

    # no key but col_sign_flip is True
    with self.assertRaises(ValueError):
      hadamard_rot.hadamard(
          2,
          None,
          row_sign_flip=False,
          col_sign_flip=True,
      )

    # no key but both row_sign_flip and col_sign_flip are True
    with self.assertRaises(ValueError):
      hadamard_rot.hadamard(
          2,
          None,
          row_sign_flip=True,
          col_sign_flip=True,
      )

  def test_error_on_multiple_reduction_axes(self):
    # multiple lhs reduction axes
    with self.assertRaises(ValueError):
      hadamard_rot.hadamard_rotate_inputs(
          jnp.ones((2, 2, 2)),
          jnp.ones((2, 2, 2)),
          jax.random.key(0),
          row_sign_flip=True,
          col_sign_flip=True,
          lhs_reduction_dim=(1, 2),
          rhs_reduction_dim=2,
      )

    # multiple rhs reduction axes
    with self.assertRaises(ValueError):
      hadamard_rot.hadamard_rotate_inputs(
          jnp.ones((2, 2, 2)),
          jnp.ones((2, 2, 2)),
          jax.random.key(0),
          row_sign_flip=True,
          col_sign_flip=True,
          lhs_reduction_dim=1,
          rhs_reduction_dim=(1, 2),
      )

    # invalid reduction axis shapes between inputs
    with self.assertRaises(ValueError):
      hadamard_rot.hadamard_rotate_inputs(
          jnp.ones((2, 4)),
          jnp.ones((2, 3)),
          jax.random.key(0),
          row_sign_flip=True,
          col_sign_flip=True,
          lhs_reduction_dim=1,
          rhs_reduction_dim=0,
      )

    # reduction axis is not a power of 2
    with self.assertRaises(ValueError):
      hadamard_rot.hadamard_rotate_inputs(
          jnp.ones((2, 3)),
          jnp.ones((2, 3)),
          jax.random.key(0),
          row_sign_flip=True,
          col_sign_flip=True,
          lhs_reduction_dim=1,
          rhs_reduction_dim=1,
      )


if __name__ == "__main__":
  absltest.main()
