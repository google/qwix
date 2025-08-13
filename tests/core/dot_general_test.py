# Copyright 2025 Google LLC
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

import functools
import logging
import time

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax import numpy as jnp
from qwix._src.core import dot_general
from qwix._src.core import einsum
from qwix._src.core import qarray


def time_it(f, *args):
  start = time.time()
  res = jax.block_until_ready(f(*args))
  end = time.time()
  return res, (end - start) * 1000


class DotGeneralTest(parameterized.TestCase):

  def _make_array(self, shape, asymmetric=False):
    zero_point = 1 if asymmetric else 0
    return (
        jax.random.normal(jax.random.key(42), shape, jnp.bfloat16) + zero_point
    )

  @parameterized.named_parameters(
      dict(
          testcase_name='int8_int4_sc',
          lhs_shape=(128, 512, 4),
          lhs_tile_sizes=(1, 128, 1),
          lhs_qtype=jnp.int8,
          rhs_shape=(512, 256, 4),
          rhs_tile_sizes=(128, 1, 1),
          rhs_qtype=jnp.int4,
          dimension_numbers=(([1], [0]), ([2], [2])),
          equivalent_einsum_str='abn,bcn->nac',
          expected_mae=0.107422,
      ),
      dict(
          testcase_name='fp8_tiled_ra',
          lhs_shape=(4, 128, 256),
          lhs_tile_sizes=(1, 1, 128),
          lhs_qtype=jnp.float8_e4m3fn,
          rhs_shape=(256, 256),
          rhs_tile_sizes=(128, 128),  # deepseek style.
          rhs_qtype=jnp.float8_e4m3fn,
          dimension_numbers=(([2], [0]), ([], [])),
          expected_mae=0.0361328,
      ),
      dict(
          testcase_name='nf4',
          lhs_shape=(128, 512, 4),
          lhs_tile_sizes=(1, None, 1),
          lhs_qtype=None,
          rhs_shape=(512, 256, 4),
          rhs_tile_sizes=(None, 1, 1),
          rhs_qtype='nf4',
          dimension_numbers=(([1], [0]), ([2], [2])),
          equivalent_einsum_str='abn,bcn->nac',
          expected_mae=0.0966797,
      ),
      dict(
          testcase_name='lhs_asymmetric',
          lhs_shape=(128, 512, 3),
          lhs_qtype=jnp.int8,
          lhs_asymmetric=True,
          rhs_shape=(512, 256, 3),
          rhs_qtype=jnp.int4,
          dimension_numbers=(([1], [0]), ([2], [2])),
          equivalent_einsum_str='abn,bcn->nac',
          expected_mae=0.154297,
      ),
      dict(
          testcase_name='lhs_asymmetric_subchannel',
          lhs_shape=(128, 512, 3),
          lhs_tile_sizes=(1, 1 / 4, 1),
          lhs_qtype=jnp.int8,
          lhs_asymmetric=True,
          rhs_shape=(512, 256, 3),
          rhs_tile_sizes=(1 / 4, 1, 1),
          rhs_qtype=jnp.int4,
          dimension_numbers=(([1], [0]), ([2], [2])),
          equivalent_einsum_str='abn,bcn->nac',
          expected_mae=0.128906,
      ),
      dict(
          testcase_name='two_contractions',
          lhs_shape=(128, 2, 128, 32, 64),
          lhs_qtype=None,
          rhs_shape=(64, 32, 128, 128),
          rhs_qtype=jnp.int8,
          dimension_numbers=(([3, 4], [1, 0]), ([0], [3])),
          equivalent_einsum_str='abcde,edfa->abcf',
          expected_mae=0.00671387,
      ),
      dict(
          testcase_name='two_contractions_two_sc',
          lhs_shape=(128, 2, 128, 32, 64),
          lhs_tile_sizes=(1, 1, 1, 8, 8),
          lhs_qtype=jnp.int8,
          rhs_shape=(64, 32, 128, 128),
          rhs_tile_sizes=(8, 8, 1, 1),
          rhs_qtype=jnp.int8,
          dimension_numbers=(([3, 4], [1, 0]), ([0], [3])),
          equivalent_einsum_str='abcde,edfa->abcf',
          expected_mae=0.00842285,
      ),
      dict(
          testcase_name='two_contractions_one_sc',
          lhs_shape=(128, 2, 128, 32, 64),
          lhs_tile_sizes=(1, 1, 1, None, 16),
          lhs_qtype=jnp.int8,
          rhs_shape=(64, 32, 128, 128),
          rhs_tile_sizes=(16, None, 1, 1),
          rhs_qtype=jnp.int8,
          dimension_numbers=(([3, 4], [1, 0]), ([0], [3])),
          equivalent_einsum_str='abcde,edfa->abcf',
          expected_mae=0.00958252,
      ),
      dict(
          testcase_name='dequant_on_input',
          lhs_shape=(16, 128, 128),
          lhs_tile_sizes=(1, 1, None),
          lhs_qtype=jnp.int8,
          rhs_shape=(128, 128, 16),
          rhs_tile_sizes=(1, None, 1),
          rhs_qtype=jnp.int8,
          dimension_numbers=(([1, 2], [0, 1]), ([], [])),
          expected_mae=0.0090332,
      ),
  )
  def test_dot_general(
      self,
      *,
      lhs_shape: tuple[int, ...],
      lhs_qtype: jax.typing.DTypeLike | None,
      lhs_tile_sizes: tuple[int | None, ...] = (),
      lhs_asymmetric: bool = False,
      rhs_shape: tuple[int, ...],
      rhs_qtype: jax.typing.DTypeLike | None,
      rhs_tile_sizes: tuple[int | None, ...] = (),
      rhs_asymmetric: bool = False,
      dimension_numbers: jax.lax.DotDimensionNumbers,
      expected_mae: float,
      equivalent_einsum_str: str | None = None,
  ):
    # Use a small tile size limit during testing.
    dot_general.MIN_TILE_SIZE_TO_DEQUANT_ON_OUTPUT = 16

    lhs = self._make_array(lhs_shape, lhs_asymmetric)
    rhs = self._make_array(rhs_shape, rhs_asymmetric)

    @functools.partial(jax.jit, static_argnums=(1,))
    def quantize(array, for_lhs):
      calibration_method = 'minmax' if for_lhs and lhs_asymmetric else 'absmax'
      tile_sizes = lhs_tile_sizes if for_lhs else rhs_tile_sizes
      how = qarray.HowToQuantize(
          qtype=lhs_qtype if for_lhs else rhs_qtype,
          channelwise_axes=(),
          tiled_axes={a: s for a, s in enumerate(tile_sizes) if s is not None},
          calibration_method=calibration_method,
      )
      return qarray.quantize(array, how)

    @jax.jit
    def _dot_general(lhs, rhs):
      if lhs_qtype is not None:
        lhs = quantize(lhs, for_lhs=True)
      return dot_general.dot_general(lhs, rhs, dimension_numbers)

    @jax.jit
    def _einsum(lhs, rhs):
      if lhs_qtype is not None:
        lhs = quantize(lhs, for_lhs=True)
      return einsum.einsum(equivalent_einsum_str, lhs, rhs)

    q_rhs = quantize(rhs, for_lhs=False)

    # Prewarm jit cache.
    jax.lax.dot_general(lhs, rhs, dimension_numbers).block_until_ready()
    _dot_general(lhs, q_rhs).block_until_ready()

    if equivalent_einsum_str is not None:
      _einsum(lhs, q_rhs).block_until_ready()

    fp_res, fp_time = time_it(jax.lax.dot_general, lhs, rhs, dimension_numbers)
    q_res, q_time = time_it(_dot_general, lhs, q_rhs)
    einsum_res = None
    einsum_time = None
    if equivalent_einsum_str is not None:
      einsum_res, einsum_time = time_it(_einsum, lhs, q_rhs)
    self.assertEqual(fp_res.dtype, q_res.dtype)
    self.assertEqual(fp_res.shape, q_res.shape)
    if einsum_res is not None:
      self.assertTrue(jnp.array_equal(q_res, einsum_res))
    mae = jnp.abs(fp_res - q_res).mean() / jnp.abs(fp_res).mean()
    logging.info(
        'Performance: lhs_qtype=%s rhs_qtype=%s mae=%s fp_time=%s q_time=%s'
        ' einsum_time=%s',
        lhs_qtype,
        rhs_qtype,
        mae,
        fp_time,
        q_time,
        einsum_time,
    )
    self.assertAlmostEqual(mae, expected_mae)


if __name__ == '__main__':
  absltest.main()
