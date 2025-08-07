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
from qwix._src.core import einsum
from qwix._src.core import qarray


def time_it(f, *args):
  start = time.time()
  res = jax.block_until_ready(f(*args))
  end = time.time()
  return res, (end - start) * 1000


class EinsumTest(parameterized.TestCase):

  def _make_array(self, shape, dtype=jnp.bfloat16, asymmetric=False):
    zero_point = 1 if asymmetric else 0
    return jax.random.normal(jax.random.key(42), shape, dtype) + zero_point

  @parameterized.named_parameters(
      dict(
          testcase_name='w8a16',
          rhs_qtype=jnp.int8,
          expected_mae=0.00674438,
      ),
      dict(
          testcase_name='w8a16_sc',
          rhs_qtype=jnp.int8,
          tile_size=128,
          expected_mae=0.00646973,
      ),
      dict(
          testcase_name='w8a8',
          rhs_qtype=jnp.int8,
          lhs_qtype=jnp.int8,
          expected_mae=0.00952148,
      ),
      dict(
          testcase_name='w8a8_sc',
          rhs_qtype=jnp.int8,
          lhs_qtype=jnp.int8,
          tile_size=128,
          expected_mae=0.0090332,
      ),
      dict(
          testcase_name='w4a16',
          rhs_qtype=jnp.int4,
          expected_mae=0.11377,
      ),
      dict(
          testcase_name='w4a16_sc',
          rhs_qtype=jnp.int4,
          tile_size=128,
          expected_mae=0.106934,
      ),
      dict(
          testcase_name='w4a8',
          rhs_qtype=jnp.int4,
          lhs_qtype=jnp.int8,
          expected_mae=0.11377,
      ),
      dict(
          testcase_name='w4a8_sc',
          rhs_qtype=jnp.int4,
          lhs_qtype=jnp.int8,
          tile_size=128,
          expected_mae=0.10791,
      ),
      dict(
          testcase_name='w4a4',
          rhs_qtype=jnp.int4,
          lhs_qtype=jnp.int4,
          expected_mae=0.161133,
      ),
      # TODO(dangyi): Re-enable once b/433799925 is fixed.
      # dict(
      #     testcase_name='w4a4_sc',
      #     rhs_qtype=jnp.int4,
      #     lhs_qtype=jnp.int4,
      #     tile_size=128,
      #     expected_mae=0.152344,
      # ),
      dict(
          testcase_name='fp8',
          rhs_qtype=jnp.float8_e4m3fn,
          expected_mae=0.0258789,
      ),
      dict(
          testcase_name='fp8_act',
          rhs_qtype=jnp.float8_e4m3fn,
          lhs_qtype=jnp.float8_e4m3fn,
          expected_mae=0.0371094,
      ),
      dict(
          testcase_name='fp8_act_sc',
          rhs_qtype=jnp.float8_e4m3fn,
          lhs_qtype=jnp.float8_e4m3fn,
          tile_size=1 / 32,
          expected_mae=0.0358887,
      ),
      dict(
          testcase_name='nf4',
          rhs_qtype='nf4',
          lhs_qtype='nf4',
          tile_size=128,
          expected_mae=0.133789,
      ),
  )
  def test_einsum_and_benchmark(
      self,
      rhs_qtype: jax.typing.DTypeLike,
      expected_mae: float,
      lhs_qtype: jax.typing.DTypeLike | None = None,
      tile_size: int | None = None,
  ):
    einsum_str = 'ab,bc->ac'
    lhs = self._make_array((1024, 4096))
    rhs = self._make_array((4096, 32768))

    @functools.partial(jax.jit, static_argnums=(1,))
    def quantize(array, for_lhs):
      how = einsum.get_how_to_quantize(
          einsum_str=einsum_str,
          ndims=[len(lhs.shape), len(rhs.shape)],
          for_lhs=for_lhs,
          qtype=lhs_qtype if for_lhs else rhs_qtype,
          tile_size=tile_size,
          calibration_method='absmax',
      )
      return qarray.quantize(array, how)

    q_rhs = quantize(rhs, for_lhs=False)

    @jax.jit
    def _einsum(lhs, rhs):
      if lhs_qtype is not None:
        lhs = quantize(lhs, for_lhs=True)
      return einsum.einsum(einsum_str, lhs, rhs)

    # Initialize jit cache.
    jnp.einsum(einsum_str, lhs, rhs).block_until_ready()
    _einsum(lhs, q_rhs).block_until_ready()

    fp_res, fp_time = time_it(jnp.einsum, einsum_str, lhs, rhs)
    q_res, q_time = time_it(_einsum, lhs, q_rhs)
    self.assertEqual(fp_res.dtype, q_res.dtype)
    self.assertEqual(fp_res.shape, q_res.shape)
    mae = jnp.abs(fp_res - q_res).mean() / jnp.abs(fp_res).mean()
    logging.info('mae=%s fp_time=%s q_time=%s', mae, fp_time, q_time)
    self.assertAlmostEqual(mae, expected_mae)

  @parameterized.named_parameters(
      dict(
          testcase_name='multi_contraction',
          einsum_str='a b c, b c d -> a d',
          qtype=jnp.int8,
          lhs_shape=(10, 256, 16),
          rhs_shape=(256, 16, 128),
          expected_rel_mae=0.0109253,
      ),
      dict(
          testcase_name='ellipsis_batch',
          einsum_str='...ab,...bc->...ac',
          qtype=jnp.int8,
          lhs_shape=(10, 256, 16),
          rhs_shape=(10, 16, 128),
          expected_rel_mae=0.00823975,
      ),
      dict(
          testcase_name='ellipsis_lhs',
          einsum_str='...ab,bc->...ac',
          qtype=jnp.int8,
          lhs_shape=(10, 256, 16),
          rhs_shape=(16, 128),
          expected_rel_mae=0.00817871,
      ),
      dict(
          testcase_name='lhs_asymmetric',
          einsum_str='abc,bcd->acd',
          qtype=jnp.int8,
          lhs_shape=(10, 256, 16),
          rhs_shape=(256, 16, 128),
          lhs_asymmetric=True,
          expected_rel_mae=0.0129395,
      ),
      dict(
          testcase_name='lhs_asymmetric_subchannel',
          einsum_str='abc,bcd->acd',
          qtype=jnp.int8,
          lhs_shape=(10, 256, 16),
          rhs_shape=(256, 16, 128),
          tile_size=1 / 4,
          lhs_asymmetric=True,
          expected_rel_mae=0.00982666,
      ),
      dict(
          testcase_name='symmetric_subchannel_nf4',
          einsum_str='abc,bcd->acd',
          qtype='nf4',
          lhs_shape=(10, 256, 16),
          rhs_shape=(256, 16, 128),
          tile_size=1 / 4,
          expected_rel_mae=0.130859,
      ),
  )
  def test_einsum(
      self,
      *,
      einsum_str,
      qtype,
      lhs_shape,
      rhs_shape,
      tile_size=None,
      lhs_asymmetric=False,
      expected_rel_mae,
  ):
    lhs = self._make_array(lhs_shape, asymmetric=lhs_asymmetric)
    rhs = self._make_array(rhs_shape)
    fp_res = jnp.einsum(einsum_str, lhs, rhs)
    q_rhs = qarray.quantize(
        rhs,
        einsum.get_how_to_quantize(
            einsum_str=einsum_str,
            ndims=[len(lhs_shape), len(rhs_shape)],
            for_lhs=False,
            qtype=qtype,
            tile_size=tile_size,
            calibration_method='absmax',
        ),
    )
    q_lhs = qarray.quantize(
        lhs,
        einsum.get_how_to_quantize(
            einsum_str=einsum_str,
            ndims=[len(lhs_shape), len(rhs_shape)],
            for_lhs=True,
            qtype=qtype,
            tile_size=tile_size,
            calibration_method='minmax' if lhs_asymmetric else 'absmax',
        ),
    )
    q_res = einsum.einsum(einsum_str, q_lhs, q_rhs)
    self.assertEqual(fp_res.dtype, q_res.dtype)
    self.assertEqual(fp_res.shape, q_res.shape)
    rel_mae = jnp.abs(fp_res - q_res).mean() / jnp.abs(fp_res).mean()
    self.assertAlmostEqual(rel_mae, expected_rel_mae)

  def test_fake_quantization(self):
    # This test case shows that, any changes, including switch jnp.float32
    # to jnp.bfloat16, or change einsum precision from HIGHEST to DEFAULT,
    # will cause the fq_q_mae to be larger than 0.002.  It also shows that
    # the relation between fp_fq_mae and fp_q_mae is not fixed.
    einsum_str = 'ab,bc->ac'
    lhs = self._make_array((512, 512), jnp.float32)
    rhs = self._make_array((512, 512), jnp.float32)
    fp_res = jnp.einsum(
        einsum_str, lhs, rhs, precision=jax.lax.Precision.HIGHEST
    )
    how = qarray.HowToQuantize(
        qtype=jnp.int8,
        channelwise_axes=(),
        tiled_axes={},
        calibration_method='absmax',
    )
    lhs = qarray.quantize(lhs, how)
    rhs = qarray.quantize(rhs, how)

    @jax.jit
    def fq_einsum(lhs, rhs):
      return jnp.einsum(
          einsum_str,
          lhs.qvalue.astype(lhs.scale.dtype) * lhs.scale,
          rhs.qvalue.astype(rhs.scale.dtype) * rhs.scale,
          precision=jax.lax.Precision.HIGHEST,
      )

    @jax.jit
    def q_einsum(lhs, rhs):
      return (
          jnp.einsum(
              einsum_str,
              lhs.qvalue,
              rhs.qvalue,
              preferred_element_type=jnp.int32,
          )
          * lhs.scale
          * rhs.scale
      )

    fq_res = fq_einsum(lhs, rhs)
    q_res = q_einsum(lhs, rhs)
    fp_fq_mae = jnp.abs(fp_res - fq_res).mean() / jnp.abs(fp_res).mean()
    fp_q_mae = jnp.abs(fp_res - q_res).mean() / jnp.abs(fp_res).mean()
    fq_q_mae = jnp.abs(fq_res - q_res).mean() / jnp.abs(fq_res).mean()
    logging.info(
        'fp_fq_mae=%s fp_q_mae=%s fq_q_mae=%s', fp_fq_mae, fp_q_mae, fq_q_mae
    )
    self.assertLess(fq_q_mae, 1e-6)

  def test_einsum_with_preferred_element_type(self):
    lhs = self._make_array((512, 512), jnp.bfloat16)
    rhs = self._make_array((512, 512), jnp.bfloat16)
    how = qarray.HowToQuantize(
        qtype=jnp.int8,
        channelwise_axes=(),
        tiled_axes={},
        calibration_method='absmax',
    )
    lhs = qarray.quantize(lhs, how)
    rhs = qarray.quantize(rhs, how)
    self.assertEqual(einsum.einsum('ab,bc->ac', lhs, rhs).dtype, jnp.bfloat16)
    self.assertEqual(
        einsum.einsum(
            'ab,bc->ac', lhs, rhs, preferred_element_type=jnp.float32
        ).dtype,
        jnp.float32,
    )


if __name__ == '__main__':
  absltest.main()
