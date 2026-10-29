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

"""Tests for the Pallas forward kernel and its tile heuristics."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np
from qwix.contrib.kernels.fused_fp8_matmul import forward_kernel
from qwix.contrib.kernels.fused_fp8_matmul import reference
from qwix.contrib.kernels.fused_fp8_matmul import test_utils


def _skip_unless_fp8_tpu(test: absltest.TestCase):
  device = jax.devices()[0]
  if not forward_kernel.fp8_matmul_supported_on(device):
    test.skipTest(f"FP8 matmul is not supported on {device.device_kind}.")


class ForwardKernelTest(parameterized.TestCase):
  """Runs the kernel with explicit tilings against the XLA reference."""

  def setUp(self):
    super().setUp()
    _skip_unless_fp8_tpu(self)

  @parameterized.parameters(
      # (m, k, n, bm, bk, bn)
      (1024, 3072, 4096, 512, 3072, 4096),  # single k block, bn == n
      (1024, 3072, 4096, 256, 1024, 1024),  # multi-k with an n grid
      (1024, 3072, 4096, 1024, 3072, 512),  # bm == m
      (512, 1024, 384, 256, 1024, 384),  # n not a power of two
  )
  def test_matches_reference(self, m, k, n, bm, bk, bn):
    lhs, rhs = test_utils.random_inputs(m, k, n, jnp.bfloat16)
    rhs_q, rhs_scale = reference.rhs_as_fp8(rhs)
    run = jax.jit(
        lambda x, w, sw: forward_kernel.fused_fp8_matmul_pallas(
            x, w, sw, bm=bm, bk=bk, bn=bn
        )
    )
    actual = run(lhs, rhs_q, rhs_scale)
    expected = jax.jit(test_utils.reference_matmul)(lhs, rhs)
    self.assertEqual((actual.shape, actual.dtype), ((m, n), jnp.bfloat16))
    # With one k block both sides quantize each row against one scale; with
    # several, the kernel quantizes per block and only agrees up to FP8 noise.
    if bk == k:
      tol = test_utils.SAME_QUANTIZATION_TOL
    else:
      tol = test_utils.FP8_NOISE_TOL
    self.assertLess(test_utils.relative_error(actual, expected), tol)

  def test_residuals_have_documented_layout_and_reproduce_lhs(self):
    m, k, n = 1024, 3072, 4096
    lhs, rhs = test_utils.random_inputs(m, k, n, jnp.bfloat16)
    rhs_q, rhs_scale = reference.rhs_as_fp8(rhs)
    run = jax.jit(
        lambda x, w, sw: forward_kernel.fused_fp8_matmul_pallas(
            x, w, sw, bm=512, bk=k, bn=n, return_xq=True
        )
    )
    out, lhs_q, lhs_scale = run(lhs, rhs_q, rhs_scale)
    self.assertEqual((lhs_q.shape, lhs_q.dtype), ((m, k), jnp.float8_e4m3fn))
    self.assertEqual((lhs_scale.shape, lhs_scale.dtype), ((m, 1), jnp.float32))
    np.testing.assert_allclose(
        np.asarray(lhs_q.astype(jnp.float32) * lhs_scale),
        np.asarray(lhs, np.float32),
        rtol=test_utils.FP8_RECONSTRUCTION_RTOL,
        atol=1e-3,
    )
    # Asking for the residuals must not change the output. Both calls go
    # through the same compiler, so this holds bit-for-bit.
    out_only = jax.jit(
        lambda x, w, sw: forward_kernel.fused_fp8_matmul_pallas(
            x, w, sw, bm=512, bk=k, bn=n
        )
    )(lhs, rhs_q, rhs_scale)
    np.testing.assert_array_equal(
        np.asarray(out, np.float32), np.asarray(out_only, np.float32)
    )

  def test_rejects_bad_tiling(self):
    lhs, rhs = test_utils.random_inputs(512, 1024, 1024, jnp.bfloat16)
    rhs_q, rhs_scale = reference.rhs_as_fp8(rhs)
    with self.assertRaisesRegex(ValueError, "divisible by bm"):
      forward_kernel.fused_fp8_matmul_pallas(
          lhs, rhs_q, rhs_scale, bm=384, bk=1024, bn=1024
      )
    with self.assertRaisesRegex(ValueError, "single k block"):
      forward_kernel.fused_fp8_matmul_pallas(
          lhs, rhs_q, rhs_scale, bm=256, bk=512, bn=1024, return_xq=True
      )
    with self.assertRaisesRegex(ValueError, "w must be"):
      forward_kernel.fused_fp8_matmul_pallas(
          lhs, rhs, rhs_scale, bm=256, bk=1024, bn=1024
      )


class HeuristicsTest(parameterized.TestCase):
  """Tile heuristics are pure Python and run anywhere."""

  @parameterized.parameters(*test_utils.SHAPES, (32768, 3072, 4096))
  def test_heuristics_divide_shape_and_fit(self, m, k, n):
    bk = forward_kernel.default_block_k(k)
    bm, bn = forward_kernel.fit_forward_tiles(m, k, n, bk, lhs_itemsize=2)
    self.assertEqual((m % bm, k % bk, n % bn), (0, 0, 0))
    self.assertLessEqual(
        forward_kernel.forward_vmem_bytes(bm, bk, bn, k, n, lhs_itemsize=2),
        forward_kernel.VMEM_BUDGET_BYTES,
    )

  def test_heuristics_prefer_single_k_and_full_n(self):
    # The production shape: the measured best tiling is bk == k and bn == n.
    m, k, n = 32768, 3072, 4096
    bk = forward_kernel.default_block_k(k)
    _, bn = forward_kernel.fit_forward_tiles(m, k, n, bk, lhs_itemsize=2)
    self.assertEqual((bk, bn), (k, n))

  def test_large_k_is_split(self):
    self.assertEqual(forward_kernel.default_block_k(12288), 1024)

  def test_f32_activations_get_a_smaller_row_block(self):
    # At this shape bm = 1024 fits for bf16 but not for f32, whose activation
    # and output blocks are twice as large (measured: 65 MiB, over the 64 MiB
    # of VMEM, when the kernel also emits residuals).
    m, k, n = 2048, 3072, 3072
    bk = forward_kernel.default_block_k(k)
    bm_bf16, _ = forward_kernel.fit_forward_tiles(m, k, n, bk, lhs_itemsize=2)
    bm_f32, bn_f32 = forward_kernel.fit_forward_tiles(
        m, k, n, bk, lhs_itemsize=4
    )
    self.assertEqual(bm_bf16, 1024)
    self.assertEqual((bm_f32, bn_f32), (512, n))

  @parameterized.parameters(
      (4096, (1024, 512), 1024),
      (1536, (1024, 512), 512),  # 1024 does not divide 1536; 512 does.
      (384, (1024, 512), 128),  # No candidate divides 384; falls back.
  )
  def test_pick_tile(self, dim, candidates, expected):
    self.assertEqual(forward_kernel.pick_tile(dim, candidates), expected)


if __name__ == "__main__":
  absltest.main()
