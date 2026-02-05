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
          expected_mae=0.007,
      ),
      dict(
          testcase_name='w8a16_sc',
          rhs_qtype=jnp.int8,
          tile_size=128,
          expected_mae=0.007,
      ),
      dict(
          testcase_name='w8a8',
          rhs_qtype=jnp.int8,
          lhs_qtype=jnp.int8,
          expected_mae=0.01,
      ),
      dict(
          testcase_name='w8a8_sc',
          rhs_qtype=jnp.int8,
          lhs_qtype=jnp.int8,
          tile_size=128,
          expected_mae=0.009,
      ),
      dict(
          testcase_name='w4a16',
          rhs_qtype=jnp.int4,
          expected_mae=0.12,
      ),
      dict(
          testcase_name='w4a16_sc',
          rhs_qtype=jnp.int4,
          tile_size=128,
          expected_mae=0.11,
      ),
      dict(
          testcase_name='w4a8',
          rhs_qtype=jnp.int4,
          lhs_qtype=jnp.int8,
          expected_mae=0.12,
      ),
      dict(
          testcase_name='w4a8_sc',
          rhs_qtype=jnp.int4,
          lhs_qtype=jnp.int8,
          tile_size=128,
          expected_mae=0.11,
      ),
      dict(
          testcase_name='w4a4',
          rhs_qtype=jnp.int4,
          lhs_qtype=jnp.int4,
          expected_mae=0.17,
      ),
      dict(
          testcase_name='w4a4_sc',
          rhs_qtype=jnp.int4,
          lhs_qtype=jnp.int4,
          tile_size=128,
          expected_mae=0.16,
      ),
      dict(
          testcase_name='fp8',
          rhs_qtype=jnp.float8_e4m3fn,
          expected_mae=0.03,
      ),
      dict(
          testcase_name='fp8_act',
          rhs_qtype=jnp.float8_e4m3fn,
          lhs_qtype=jnp.float8_e4m3fn,
          expected_mae=0.038,
      ),
      dict(
          testcase_name='fp8_act_sc',
          rhs_qtype=jnp.float8_e4m3fn,
          lhs_qtype=jnp.float8_e4m3fn,
          tile_size=1 / 32,
          expected_mae=0.036,
      ),
      dict(
          testcase_name='nf4',
          rhs_qtype='nf4',
          lhs_qtype='nf4',
          tile_size=128,
          expected_mae=0.14,
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
    self.assertLessEqual(mae, expected_mae)

  @parameterized.named_parameters(
      dict(
          testcase_name='multi_contraction',
          einsum_str='a b c, b c d -> a d',
          qtype=jnp.int8,
          lhs_shape=(10, 256, 16),
          rhs_shape=(256, 16, 128),
          expected_rel_mae=0.011,
      ),
      dict(
          testcase_name='ellipsis_batch',
          einsum_str='...ab,...bc->...ac',
          qtype=jnp.int8,
          lhs_shape=(10, 256, 16),
          rhs_shape=(10, 16, 128),
          expected_rel_mae=0.009,
      ),
      dict(
          testcase_name='ellipsis_lhs',
          einsum_str='...ab,bc->...ac',
          qtype=jnp.int8,
          lhs_shape=(10, 256, 16),
          rhs_shape=(16, 128),
          expected_rel_mae=0.009,
      ),
      dict(
          testcase_name='lhs_asymmetric',
          einsum_str='abc,bcd->acd',
          qtype=jnp.int8,
          lhs_shape=(10, 256, 16),
          rhs_shape=(256, 16, 128),
          lhs_asymmetric=True,
          expected_rel_mae=0.013,
      ),
      dict(
          testcase_name='lhs_asymmetric_subchannel',
          einsum_str='abc,bcd->acd',
          qtype=jnp.int8,
          lhs_shape=(10, 256, 16),
          rhs_shape=(256, 16, 128),
          tile_size=1 / 4,
          lhs_asymmetric=True,
          expected_rel_mae=0.010,
      ),
      dict(
          testcase_name='symmetric_subchannel_nf4',
          einsum_str='abc,bcd->acd',
          qtype='nf4',
          lhs_shape=(10, 256, 16),
          rhs_shape=(256, 16, 128),
          tile_size=1 / 4,
          expected_rel_mae=0.14,
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
    self.assertLessEqual(rel_mae, expected_rel_mae)

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
    how = qarray.HowToQuantize(qtype=jnp.int8)
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
    self.assertLessEqual(fq_q_mae, 1e-6)

  def test_dequant_on_inputs(self):
    lhs = self._make_array((16, 128, 128), jnp.bfloat16)
    rhs = self._make_array((128, 128, 16), jnp.bfloat16)
    lhs = qarray.quantize(
        lhs,
        qarray.HowToQuantize(qtype=jnp.int8, channelwise_axes=(0, 1)),
    )
    rhs = qarray.quantize(
        rhs,
        qarray.HowToQuantize(qtype=jnp.int8, channelwise_axes=(0, 2)),
    )
    out = einsum.einsum('TNH,NHD -> TD', lhs, rhs)
    self.assertEqual(out.shape, (16, 16))
    self.assertEqual(out.dtype, jnp.bfloat16)

  @parameterized.named_parameters(
      dict(
          testcase_name='broadcasting_issue',
          einsum_str='BTNH,BSNH->BTNS',
          lhs_shape=(1, 2, 4, 8),
          rhs_shape=(1, 3, 1, 8),
          expected_shape=(1, 2, 4, 3),
      ),
      dict(
          testcase_name='generic_broadcasting_4_vs_8',
          einsum_str='BTNH,BSNH->BTNS',
          lhs_shape=(1, 2, 4, 8),
          rhs_shape=(1, 3, 8, 8),
          expected_shape=(1, 2, 8, 3),
          skip_reference_check=True,
      ),
      dict(
          testcase_name='generic_broadcasting_8_vs_4',
          einsum_str='BTNH,BSNH->BTNS',
          lhs_shape=(1, 2, 8, 8),
          rhs_shape=(1, 3, 4, 8),
          expected_shape=(1, 2, 8, 3),
          skip_reference_check=True,
      ),
      dict(
          testcase_name='new_batch_broadcasting_bth_thk',
          einsum_str='bth,thk->btk',
          lhs_shape=(2, 3, 4),
          rhs_shape=(3, 4, 5),
          expected_shape=(2, 3, 5),
      ),
      dict(
          testcase_name='mixed_type_inputs_lhs_quantized',
          einsum_str='ij,jk->ik',
          lhs_shape=(128, 128),
          rhs_shape=(128, 32),
          expected_shape=(128, 32),
          lhs_qtype=jnp.int8,
          expected_rel_error=0.02,
      ),
      dict(
          testcase_name='scalar_broadcasting',
          einsum_str='ij,->ij',
          lhs_shape=(128, 128),
          rhs_shape=(),
          expected_shape=(128, 128),
          lhs_qtype=jnp.int8,
          rhs_is_scalar=True,
          expected_rel_error=0.02,
      ),
  )
  def test_broadcasting_and_mixed_types(
      self,
      einsum_str,
      lhs_shape,
      rhs_shape,
      expected_shape,
      lhs_qtype=None,
      rhs_qtype=None,
      rhs_is_scalar=False,
      expected_rel_error=None,
      skip_reference_check=False,
  ):
    if rhs_is_scalar:
      lhs = self._make_array(lhs_shape, jnp.float32)
      rhs = jnp.array(2.0, dtype=jnp.float32)
      fp_res = lhs * 2.0
    elif not skip_reference_check:
      lhs = (
          self._make_array(lhs_shape, jnp.float32)
          if lhs_shape
          else jnp.array(2.0, dtype=jnp.float32)
      )
      rhs = (
          self._make_array(rhs_shape, jnp.float32)
          if rhs_shape
          else jnp.array(2.0, dtype=jnp.float32)
      )
      fp_res = jnp.einsum(einsum_str, lhs, rhs)
    else:
      # Create inputs for qarray execution even if we skip reference check
      lhs = (
          self._make_array(lhs_shape, jnp.float32)
          if lhs_shape
          else jnp.array(2.0, dtype=jnp.float32)
      )
      rhs = (
          self._make_array(rhs_shape, jnp.float32)
          if rhs_shape
          else jnp.array(2.0, dtype=jnp.float32)
      )
      fp_res = None

    q_lhs = lhs
    if lhs_qtype is not None:
      q_lhs = qarray.quantize(lhs, qarray.HowToQuantize(qtype=lhs_qtype))

    q_rhs = rhs
    if rhs_qtype is not None:
      q_rhs = qarray.quantize(rhs, qarray.HowToQuantize(qtype=rhs_qtype))

    q_res = einsum.einsum(einsum_str, q_lhs, q_rhs)

    self.assertEqual(q_res.shape, expected_shape)

    _ = q_res.block_until_ready()
    # Basic check ensuring it runs. Correctness check below.

    if expected_rel_error is not None and not skip_reference_check:
      rel_error = jnp.abs(q_res - fp_res).mean() / jnp.abs(fp_res).mean()
      self.assertLess(rel_error, expected_rel_error)

  def test_generic_broadcasting_incompatible(self):
    lhs = jnp.ones((1, 2, 3, 8))
    rhs = jnp.ones((1, 3, 4, 8))
    with self.assertRaisesRegex(ValueError, 'Cannot broadcast'):
      einsum.einsum('BTNH,BSNH->BTNS', lhs, rhs)


if __name__ == '__main__':
  absltest.main()
