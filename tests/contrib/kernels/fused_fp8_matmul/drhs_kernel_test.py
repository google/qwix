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

"""Tests for the Pallas dRHS kernel and its tile heuristics."""

import functools

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
from qwix.contrib.kernels.fused_fp8_matmul import drhs_kernel
from qwix.contrib.kernels.fused_fp8_matmul import forward_kernel
from qwix.contrib.kernels.fused_fp8_matmul import reference
from qwix.contrib.kernels.fused_fp8_matmul import test_utils

# Production shapes the heuristics were tuned on: the first, and the one whose
# `k` admits no 1024 block.
_WI_1 = (32768, 3072, 4096)
_O_PROJ = (32768, 1792, 3072)


def _inputs(
    m: int, k: int, n: int, dtype: jax.typing.DTypeLike = jnp.bfloat16
) -> tuple[jax.Array, jax.Array, jax.Array]:
  """Returns `(lhs_q, lhs_scale, dout)` as the backward pass receives them."""
  lhs, _ = test_utils.random_inputs(m, k, n, dtype)
  dout = jax.random.normal(jax.random.PRNGKey(3), (m, n), jnp.float32)
  lhs_q, lhs_scale = reference.quantize_rows_fp8(lhs)
  return lhs_q, lhs_scale, dout.astype(dtype)


class DrhsKernelTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    device = jax.devices()[0]
    if not forward_kernel.fp8_matmul_supported_on(device):
      self.skipTest(f"FP8 matmul is not supported on {device.device_kind}.")

  def _drhs(self, lhs_q, lhs_scale, dout, **tiles):
    fn = functools.partial(drhs_kernel.fused_fp8_drhs_pallas, **tiles)
    return jax.jit(fn)(lhs_q, lhs_scale, dout)

  @parameterized.named_parameters(
      dict(
          testcase_name="single_m_block",
          shape=(1024, 1024, 2048),
          tiles=dict(bm=1024, bk=1024, bn=2048),
      ),
      dict(
          testcase_name="single_m_block_tiled_k_and_n",
          shape=(1024, 2048, 2048),
          tiles=dict(bm=1024, bk=512, bn=1024),
      ),
      dict(
          testcase_name="multi_m_accumulator",
          shape=(2048, 1024, 1024),
          tiles=dict(bm=512, bk=1024, bn=1024),
      ),
      dict(
          testcase_name="production_tiling",
          shape=(8192, 3072, 4096),
          tiles=dict(bm=4096, bk=1024, bn=1024),
      ),
      dict(
          testcase_name="smallest_aligned_blocks",
          shape=(256, 384, 512),
          tiles=dict(bm=128, bk=128, bn=128),
      ),
  )
  def test_matches_reference(self, shape, tiles):
    m, k, n = shape
    lhs_q, lhs_scale, dout = _inputs(m, k, n)
    actual = self._drhs(lhs_q, lhs_scale, dout, **tiles)
    expected = jax.jit(reference.fused_fp8_drhs_reference)(
        lhs_q, lhs_scale, dout
    )
    self.assertEqual((actual.shape, actual.dtype), ((k, n), dout.dtype))
    # With `bm == m` the kernel quantizes each cotangent column against one
    # scale, exactly like the reference; with `bm < m` each m block gets its
    # own scale, which only agrees up to FP8 quantization noise.
    if tiles["bm"] == m:
      tol = test_utils.SAME_QUANTIZATION_TOL
    else:
      tol = test_utils.FP8_NOISE_TOL
    self.assertLess(test_utils.relative_error(actual, expected), tol)

  def test_float32_cotangent_and_out_dtype(self):
    m, k, n = 512, 1024, 1024
    lhs_q, lhs_scale, dout = _inputs(m, k, n, jnp.float32)
    tiles = dict(bm=512, bk=1024, bn=1024)
    actual = self._drhs(lhs_q, lhs_scale, dout, **tiles)
    expected = jax.jit(reference.fused_fp8_drhs_reference)(
        lhs_q, lhs_scale, dout
    )
    self.assertEqual(actual.dtype, jnp.float32)
    self.assertLess(
        test_utils.relative_error(actual, expected),
        test_utils.SAME_QUANTIZATION_TOL,
    )
    narrow = self._drhs(lhs_q, lhs_scale, dout, out_dtype=jnp.bfloat16, **tiles)
    self.assertEqual(narrow.dtype, jnp.bfloat16)
    self.assertLess(
        test_utils.relative_error(narrow, expected),
        test_utils.SAME_QUANTIZATION_TOL,
    )

  @parameterized.product(
      shape=test_utils.SHAPES + (_WI_1, _O_PROJ),
      dtype=(jnp.bfloat16, jnp.float32),
  )
  def test_heuristic_tiling_compiles_and_matches_reference(self, shape, dtype):
    # The heuristics were measured with a bfloat16 cotangent; this is where a
    # float32 one is shown to fit too (it does not with the bfloat16 tiling at
    # the first production shape).
    m, k, n = shape
    lhs_q, lhs_scale, dout = _inputs(m, k, n, dtype)
    bm, bk, bn = drhs_kernel.fit_drhs_tiles(
        m, k, n, cotangent_itemsize=jnp.dtype(dtype).itemsize
    )
    actual = self._drhs(lhs_q, lhs_scale, dout, bm=bm, bk=bk, bn=bn)
    expected = jax.jit(reference.fused_fp8_drhs_reference)(
        lhs_q, lhs_scale, dout
    )
    tol = (
        test_utils.SAME_QUANTIZATION_TOL
        if bm == m
        else test_utils.FP8_NOISE_TOL
    )
    self.assertLess(test_utils.relative_error(actual, expected), tol)

  def test_float32_weight_output_with_production_tiling(self):
    # bf16 activations with float32 master weights: the cotangent block is
    # bf16, only the output block doubles.
    m, k, n = _WI_1
    lhs_q, lhs_scale, dout = _inputs(m, k, n, jnp.bfloat16)
    bm, bk, bn = drhs_kernel.fit_drhs_tiles(m, k, n, cotangent_itemsize=2)
    actual = self._drhs(
        lhs_q, lhs_scale, dout, bm=bm, bk=bk, bn=bn, out_dtype=jnp.float32
    )
    self.assertEqual(actual.dtype, jnp.float32)

  @parameterized.named_parameters(
      dict(
          testcase_name="contracting_dims_differ",
          corrupt=lambda lhs_q, lhs_scale, dout: (
              lhs_q[:128],
              lhs_scale[:128],
              dout,
          ),
          error="Contracting dims differ",
      ),
      dict(
          testcase_name="lhs_scale_wrong_shape",
          corrupt=lambda lhs_q, lhs_scale, dout: (
              lhs_q,
              lhs_scale.reshape(-1),
              dout,
          ),
          error="lhs_scale must have shape",
      ),
      dict(
          testcase_name="lhs_q_not_fp8",
          corrupt=lambda lhs_q, lhs_scale, dout: (
              lhs_q.astype(jnp.bfloat16),
              lhs_scale,
              dout,
          ),
          error="lhs_q must be",
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
      dict(
          testcase_name="bk_not_a_multiple_of_128",
          tiles=dict(bk=192),
          error="bk .* must be a multiple of 128",
      ),
  )
  def test_rejects_bad_inputs(
      self, error, corrupt=lambda *args: args, tiles=()
  ):
    args = corrupt(*_inputs(256, 384, 512))
    with self.assertRaisesRegex(ValueError, error):
      self._drhs(*args, **(dict(bm=128, bk=128, bn=128) | dict(tiles)))


class HeuristicsTest(parameterized.TestCase):
  """Tile heuristics are pure Python and run anywhere."""

  @parameterized.parameters(*test_utils.SHAPES, _WI_1, _O_PROJ)
  def test_heuristics_divide_shape(self, m, k, n):
    bm, bk, bn = drhs_kernel.fit_drhs_tiles(m, k, n, cotangent_itemsize=2)
    self.assertEqual((m % bm, k % bk, n % bn), (0, 0, 0))
    self.assertEqual(bk % 128, 0)

  def test_heuristics_pick_the_measured_production_tilings(self):
    # The measured table's first entry, and the k = 1792 entry where bk = 1024
    # does not divide. These are the tilings behind the reported numbers.
    self.assertEqual(
        drhs_kernel.fit_drhs_tiles(*_WI_1, cotangent_itemsize=2),
        (4096, 1024, 1024),
    )
    self.assertEqual(
        drhs_kernel.fit_drhs_tiles(*_O_PROJ, cotangent_itemsize=2),
        (4096, 1792, 512),
    )

  def test_f32_cotangent_gets_a_smaller_m_block(self):
    # (4096, 1024, 1024) overflows VMEM with a float32 cotangent; the next
    # table entry that divides the shape is taken instead.
    m, k, n = _WI_1
    self.assertEqual(
        drhs_kernel.fit_drhs_tiles(m, k, n, cotangent_itemsize=4),
        (2048, 1024, 1024),
    )
    # A narrower n block keeps the full m block even in float32.
    m, k, n = _O_PROJ
    self.assertEqual(
        drhs_kernel.fit_drhs_tiles(m, k, n, cotangent_itemsize=4),
        (4096, 1792, 512),
    )

  @parameterized.parameters((512, 1024, 2048), (1024, 3072, 4096))
  def test_heuristics_take_one_m_block_when_it_fits(self, m, k, n):
    # One m block means one scale per cotangent column, the reference's
    # numerics; the API-level tests rely on this for their tight tolerance.
    self.assertEqual(
        drhs_kernel.fit_drhs_tiles(m, k, n, cotangent_itemsize=2)[0], m
    )


if __name__ == "__main__":
  absltest.main()
