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

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
from qwix._src.core import dot_general
from qwix._src.core import dot_general_qt
from qwix._src.core import multipass
from qwix._src.core import qarray


class MultiPassTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.rng = jax.random.PRNGKey(42)

  def test_residual_decompose(self):
    key = self.rng
    x = jax.random.normal(key, (4, 32), dtype=jnp.float32)
    how = dot_general.get_how_to_quantize(
        dimension_numbers=(((1,), (0,)), ((), ())),
        ndims=(2, 2),
        for_lhs=True,
        qtype='mxfp8_16',
        tile_size=16,
    )
    passes = multipass.residual_decompose(x, how, n_passes=2)
    self.assertLen(passes, 2)
    q0, q1 = passes
    self.assertIsInstance(q0, qarray.QArray)
    self.assertIsInstance(q1, qarray.QArray)

    # Dequantized values
    x0 = qarray.dequantize(q0)
    x1 = qarray.dequantize(q1)
    reconstructed = x0 + x1

    err_single = multipass.compute_relative_error(x, x0)
    err_double = multipass.compute_relative_error(x, reconstructed)
    self.assertLess(err_double, err_single)

    snr_single = multipass.compute_snr_db(x, x0)
    snr_double = multipass.compute_snr_db(x, reconstructed)
    self.assertGreater(snr_double, snr_single)

  @parameterized.parameters(
      'triangular',
      'full_cross',
      'lhs_high_precision',
      'rhs_high_precision',
  )
  def test_multipass_modes(self, mode):
    k1, k2 = jax.random.split(self.rng)
    lhs = jax.random.normal(k1, (8, 32), dtype=jnp.float32)
    rhs = jax.random.normal(k2, (32, 8), dtype=jnp.float32)
    dnums = (((1,), (0,)), ((), ()))

    true_res = jax.lax.dot_general(lhs, rhs, dnums)
    res = multipass.multipass_dot(
        lhs,
        rhs,
        dimension_numbers=dnums,
        mode=mode,
        lhs_qtype='mxfp8_16',
        rhs_qtype='mxfp8_16',
        tile_size=16,
    )

    self.assertEqual(res.shape, (8, 8))
    self.assertFalse(jnp.isnan(res).any())

    # Multi-pass should achieve strong SNR with true result (> 25 dB)
    snr = multipass.compute_snr_db(true_res, res)
    self.assertGreater(snr, 25.0)

  def test_multipass_triangular_vs_full_cross(self):
    k1, k2 = jax.random.split(self.rng)
    lhs = jax.random.normal(k1, (8, 32), dtype=jnp.float32)
    rhs = jax.random.normal(k2, (32, 8), dtype=jnp.float32)
    dnums = (((1,), (0,)), ((), ()))

    res_tri = multipass.multipass_dot(
        lhs,
        rhs,
        dnums,
        mode='triangular',
        lhs_qtype='mxfp8_16',
        rhs_qtype='mxfp8_16',
        tile_size=16,
    )
    res_full = multipass.multipass_dot(
        lhs,
        rhs,
        dnums,
        mode='full_cross',
        lhs_qtype='mxfp8_16',
        rhs_qtype='mxfp8_16',
        tile_size=16,
    )

    # Triangular drops second-order cross term A_1 B_1, so diff should be
    # very small.
    rel_diff = multipass.compute_relative_error(res_full, res_tri)
    self.assertLess(rel_diff, 0.05)

  @parameterized.parameters(False, True)
  def test_asymmetric_mxint_dot(self, accumulate_blocks):
    k1, k2 = jax.random.split(self.rng)
    lhs = jax.random.normal(k1, (8, 64), dtype=jnp.float32)
    rhs = jax.random.normal(k2, (64, 8), dtype=jnp.float32)
    dnums = (((1,), (0,)), ((), ()))

    # mxint4 x mxint8
    res48 = multipass.asymmetric_mxint_dot(
        lhs,
        rhs,
        dnums,
        lhs_qtype='mxint4',
        rhs_qtype='mxint8',
        tile_size=32,
        accumulate_blocks=accumulate_blocks,
    )
    self.assertEqual(res48.shape, (8, 8))
    self.assertFalse(jnp.isnan(res48).any())

    # mxint8 x mxint4
    res84 = multipass.asymmetric_mxint_dot(
        lhs,
        rhs,
        dnums,
        lhs_qtype='mxint8',
        rhs_qtype='mxint4',
        tile_size=32,
        accumulate_blocks=accumulate_blocks,
    )
    self.assertEqual(res84.shape, (8, 8))
    self.assertFalse(jnp.isnan(res84).any())

  def test_asymmetric_mxint_dot_accumulate_consistency(self):
    k1, k2 = jax.random.split(self.rng)
    lhs = jax.random.normal(k1, (4, 64), dtype=jnp.float32)
    rhs = jax.random.normal(k2, (64, 4), dtype=jnp.float32)
    dnums = (((1,), (0,)), ((), ()))

    res_deq = multipass.asymmetric_mxint_dot(
        lhs,
        rhs,
        dnums,
        lhs_qtype='mxint4',
        rhs_qtype='mxint8',
        tile_size=32,
        accumulate_blocks=False,
    )
    res_block = multipass.asymmetric_mxint_dot(
        lhs,
        rhs,
        dnums,
        lhs_qtype='mxint4',
        rhs_qtype='mxint8',
        tile_size=32,
        accumulate_blocks=True,
    )
    self.assertTrue(jnp.allclose(res_deq, res_block, rtol=1e-4, atol=1e-4))

  @parameterized.parameters(
      'triangular',
      'full_cross',
      'lhs_high_precision',
      'rhs_high_precision',
  )
  def test_dot_general_qt_multipass_fwd(self, mode):
    k1, k2 = jax.random.split(self.rng)
    lhs = jax.random.normal(k1, (4, 32), dtype=jnp.float32)
    rhs = jax.random.normal(k2, (32, 4), dtype=jnp.float32)
    dnums = (((1,), (0,)), ((), ()))

    config = dot_general_qt.DotGeneralQtConfig(
        lhs_qtype='mxfp8_16',
        rhs_qtype='mxfp8_16',
        tile_size=16,
        multipass_mode=mode,
    )
    res = dot_general_qt.dot_general_qt(lhs, rhs, dnums, config)
    self.assertEqual(res.shape, (4, 4))
    self.assertFalse(jnp.isnan(res).any())

  def test_dot_general_qt_multipass_bwd(self):
    k1, k2 = jax.random.split(self.rng)
    lhs = jax.random.normal(k1, (16, 32), dtype=jnp.float32)
    rhs = jax.random.normal(k2, (32, 16), dtype=jnp.float32)
    dnums = (((1,), (0,)), ((), ()))

    config = dot_general_qt.DotGeneralQtConfig(
        lhs_qtype='mxfp8_16',
        rhs_qtype='mxfp8_16',
        tile_size=16,
        multipass_mode='triangular',
        dlhs_multipass_mode='triangular',
        drhs_multipass_mode='triangular',
    )

    def loss_fn(a, b):
      c = dot_general_qt.dot_general_qt(a, b, dnums, config)
      return jnp.sum(c)

    grad_a, grad_b = jax.grad(loss_fn, argnums=(0, 1))(lhs, rhs)
    self.assertEqual(grad_a.shape, lhs.shape)
    self.assertEqual(grad_b.shape, rhs.shape)
    self.assertFalse(jnp.isnan(grad_a).any())
    self.assertFalse(jnp.isnan(grad_b).any())

  def test_dot_general_qt_asymmetric_mxint4_mxint8(self):
    k1, k2 = jax.random.split(self.rng)
    lhs = jax.random.normal(k1, (4, 32), dtype=jnp.float32)
    rhs = jax.random.normal(k2, (32, 4), dtype=jnp.float32)
    dnums = (((1,), (0,)), ((), ()))

    config = dot_general_qt.DotGeneralQtConfig(
        lhs_qtype='mxint4',
        rhs_qtype='mxint8',
        tile_size=32,
    )

    def loss_fn(a, b):
      c = dot_general_qt.dot_general_qt(a, b, dnums, config)
      return jnp.sum(c)

    val, (grad_a, grad_b) = jax.value_and_grad(loss_fn, argnums=(0, 1))(
        lhs, rhs
    )
    self.assertFalse(jnp.isnan(val))
    self.assertEqual(grad_a.shape, lhs.shape)
    self.assertEqual(grad_b.shape, rhs.shape)
    self.assertFalse(jnp.isnan(grad_a).any())
    self.assertFalse(jnp.isnan(grad_b).any())


if __name__ == '__main__':
  absltest.main()
