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

"""Tests for the Pallas dLHS kernel and its tile heuristics."""

import functools

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
from qwix.contrib.kernels.fused_fp8_matmul import dlhs_kernel
from qwix.contrib.kernels.fused_fp8_matmul import forward_kernel
from qwix.contrib.kernels.fused_fp8_matmul import reference
from qwix.contrib.kernels.fused_fp8_matmul import test_utils

# The production shape the heuristics were tuned on.
_WI_1 = (32768, 3072, 4096)


def _inputs(m, k, n, dtype=jnp.bfloat16):
  """Returns `(dout, rhs_q, rhs_scale)` as the backward pass receives them."""
  _, rhs = test_utils.random_inputs(m, k, n, dtype)
  dout = jax.random.normal(jax.random.PRNGKey(3), (m, n), jnp.float32)
  rhs_q, rhs_scale = reference.rhs_as_fp8(rhs)
  return dout.astype(dtype), rhs_q, rhs_scale


class DlhsKernelTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    device = jax.devices()[0]
    if not forward_kernel.fp8_matmul_supported_on(device):
      self.skipTest(f"FP8 matmul is not supported on {device.device_kind}.")

  def _dlhs(self, dout, rhs_q, rhs_scale, **tiles):
    fn = functools.partial(dlhs_kernel.fused_fp8_dlhs_pallas, **tiles)
    return jax.jit(fn)(dout, rhs_q, rhs_scale)

  @parameterized.named_parameters(
      dict(
          testcase_name="full_k_full_n_no_scratch",
          shape=(1024, 3072, 4096),
          tiles=dict(bm=512, bk=3072, bn=4096),
      ),
      dict(
          testcase_name="single_m_block",
          shape=(1024, 3072, 4096),
          tiles=dict(bm=1024, bk=3072, bn=4096),
      ),
      dict(
          testcase_name="partial_k_cotangent_cache",
          shape=(1024, 3072, 4096),
          tiles=dict(bm=256, bk=1024, bn=4096),
      ),
      dict(
          testcase_name="partial_n_accumulator",
          shape=(1024, 3072, 4096),
          tiles=dict(bm=256, bk=3072, bn=1024),
      ),
      dict(
          testcase_name="partial_k_and_n",
          shape=(512, 2048, 2048),
          tiles=dict(bm=256, bk=512, bn=512),
      ),
      dict(
          testcase_name="smallest_aligned_blocks",
          shape=(256, 384, 512),
          tiles=dict(bm=128, bk=128, bn=128),
      ),
  )
  def test_matches_reference(self, shape, tiles):
    m, k, n = shape
    dout, rhs_q, rhs_scale = _inputs(m, k, n)
    actual = self._dlhs(dout, rhs_q, rhs_scale, **tiles)
    expected = jax.jit(reference.fused_fp8_dlhs_reference)(
        dout, rhs_q, rhs_scale
    )
    self.assertEqual((actual.shape, actual.dtype), ((m, k), dout.dtype))
    # With `bn == n` the kernel quantizes each cotangent row against one scale,
    # exactly like the reference; with `bn < n` each n block gets its own
    # scale, which only agrees up to FP8 quantization noise.
    if tiles["bn"] == n:
      tol = test_utils.SAME_QUANTIZATION_TOL
    else:
      tol = test_utils.FP8_NOISE_TOL
    self.assertLess(test_utils.relative_error(actual, expected), tol)

  def test_float32_cotangent_and_out_dtype(self):
    m, k, n = 512, 1024, 1024
    dout, rhs_q, rhs_scale = _inputs(m, k, n, jnp.float32)
    tiles = dict(bm=512, bk=1024, bn=1024)
    actual = self._dlhs(dout, rhs_q, rhs_scale, **tiles)
    expected = jax.jit(reference.fused_fp8_dlhs_reference)(
        dout, rhs_q, rhs_scale
    )
    self.assertEqual(actual.dtype, jnp.float32)
    self.assertLess(
        test_utils.relative_error(actual, expected),
        test_utils.SAME_QUANTIZATION_TOL,
    )
    narrow = self._dlhs(dout, rhs_q, rhs_scale, out_dtype=jnp.bfloat16, **tiles)
    self.assertEqual(narrow.dtype, jnp.bfloat16)
    self.assertLess(
        test_utils.relative_error(narrow, expected),
        test_utils.SAME_QUANTIZATION_TOL,
    )

  @parameterized.product(
      shape=test_utils.SHAPES + (_WI_1,), dtype=(jnp.bfloat16, jnp.float32)
  )
  def test_heuristic_tiling_compiles_and_matches_reference(self, shape, dtype):
    self._check_heuristic_tiling(shape, dtype)

  @parameterized.parameters(
      # (shape, dtype): heuristic tilings that the VMEM model puts at the top of
      # the budget, including ones on the cotangent-cache path (`bk < k`).
      ((2048, 7168, 7168), jnp.bfloat16),  # (128, k, n): 60.4 MiB
      ((2048, 4096, 16384), jnp.bfloat16),  # (256, 1024, n): 58.0 MiB, cache
      # Without the folded tile in the model this picked (512, 1024, n) at a
      # modeled 62.0 MiB, which ran out of VMEM at 72.3 MiB.
      ((4096, 8192, 8192), jnp.float32),  # (256, 1024, n), cache
  )
  def test_heuristic_tiling_at_the_vmem_budget(self, shape, dtype):
    self._check_heuristic_tiling(shape, dtype)

  def _check_heuristic_tiling(self, shape, dtype):
    # The tilings the heuristics pick sit close to the VMEM budget for some
    # shapes; this is where `VMEM_BUDGET_BYTES` is actually exercised.
    m, k, n = shape
    dout, rhs_q, rhs_scale = _inputs(m, k, n, dtype)
    bm, bk, bn = dlhs_kernel.fit_dlhs_tiles(
        m, k, n, cotangent_itemsize=jnp.dtype(dtype).itemsize
    )
    actual = self._dlhs(dout, rhs_q, rhs_scale, bm=bm, bk=bk, bn=bn)
    expected = jax.jit(reference.fused_fp8_dlhs_reference)(
        dout, rhs_q, rhs_scale
    )
    tol = (
        test_utils.SAME_QUANTIZATION_TOL
        if bn == n
        else test_utils.FP8_NOISE_TOL
    )
    self.assertLess(test_utils.relative_error(actual, expected), tol)

  @parameterized.named_parameters(
      dict(
          testcase_name="contracting_dims_differ",
          corrupt=lambda dout, rhs_q, rhs_scale: (
              dout,
              rhs_q[:, :256],
              rhs_scale,
          ),
          error="Contracting dims differ",
      ),
      dict(
          testcase_name="rhs_scale_wrong_rank",
          corrupt=lambda dout, rhs_q, rhs_scale: (dout, rhs_q, rhs_scale[None]),
          error="rhs_scale must have shape",
      ),
      dict(
          testcase_name="rhs_q_not_fp8",
          corrupt=lambda dout, rhs_q, rhs_scale: (
              dout,
              rhs_q.astype(jnp.bfloat16),
              rhs_scale,
          ),
          error="rhs_q must be",
      ),
      dict(
          testcase_name="bm_does_not_divide_m",
          tiles=dict(bm=100),
          error="m .* must be divisible",
      ),
      dict(
          testcase_name="bk_does_not_divide_k",
          tiles=dict(bk=100),
          error="k .* must be divisible",
      ),
      dict(
          testcase_name="bn_does_not_divide_n",
          tiles=dict(bn=100),
          error="n .* must be divisible",
      ),
  )
  def test_rejects_bad_inputs(
      self, error, corrupt=lambda *args: args, tiles=()
  ):
    args = corrupt(*_inputs(256, 512, 384))
    with self.assertRaisesRegex(ValueError, error):
      self._dlhs(*args, **(dict(bm=128, bk=128, bn=128) | dict(tiles)))


class HeuristicsTest(parameterized.TestCase):
  """Tile heuristics are pure Python and run anywhere."""

  @parameterized.parameters(*test_utils.SHAPES, _WI_1)
  def test_heuristics_divide_shape_and_fit(self, m, k, n):
    bm, bk, bn = dlhs_kernel.fit_dlhs_tiles(m, k, n, cotangent_itemsize=2)
    self.assertEqual((m % bm, k % bk, n % bn), (0, 0, 0))
    self.assertLessEqual(
        dlhs_kernel.dlhs_vmem_bytes(bm, bk, bn, k, n, cotangent_itemsize=2),
        dlhs_kernel.VMEM_BUDGET_BYTES,
    )

  def test_heuristics_pick_the_measured_production_tiling(self):
    # Full-k, full-n, and the largest m block that fits: the tiling behind the
    # number reported for this shape.
    m, k, n = _WI_1
    self.assertEqual(
        dlhs_kernel.fit_dlhs_tiles(m, k, n, cotangent_itemsize=2), (1024, k, n)
    )

  def test_f32_cotangent_gets_a_smaller_row_block(self):
    # A float32 cotangent doubles the in and out blocks, so bm = 1024 no longer
    # fits at the production shape; bk == k and bn == n are kept.
    m, k, n = _WI_1
    self.assertEqual(
        dlhs_kernel.fit_dlhs_tiles(m, k, n, cotangent_itemsize=4), (512, k, n)
    )

  def test_heuristics_shrink_k_before_n_when_full_k_does_not_fit(self):
    # A `(k, n)` weight block of 64 MiB cannot fit at any `bm`, so `bk` must
    # shrink; `bn == n` is kept because it is the stronger preference.
    m, k, n = 2048, 4096, 16384
    bm, bk, bn = dlhs_kernel.fit_dlhs_tiles(m, k, n, cotangent_itemsize=2)
    self.assertEqual(bn, n)
    self.assertLess(bk, k)
    self.assertEqual((m % bm, k % bk), (0, 0))
    self.assertLessEqual(
        dlhs_kernel.dlhs_vmem_bytes(bm, bk, bn, k, n, cotangent_itemsize=2),
        dlhs_kernel.VMEM_BUDGET_BYTES,
    )


if __name__ == "__main__":
  absltest.main()
