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

import re
from unittest import mock

from absl import logging
from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax import numpy as jnp
from qwix._src.core import dot_general
from qwix._src.core import qarray


def rel_mae(x, y):
  assert x.dtype == y.dtype and x.shape == y.shape
  return jnp.abs(x - y).mean() / jnp.abs(x).mean()


_HLO_COMPUTATION = re.compile(r'^(?:ENTRY )?%([\w.-]+) .*\{$')
_HLO_INSTRUCTION = re.compile(
    r'^\s*(?:ROOT )?%([\w.-]+) = (\w+)\[([\d,]*)\]\S* ([\w-]+)\(([^)]*)\)(.*)$'
)
_INT_DTYPES = ('s4', 's8', 'u4', 'u8')
_FLOAT_DTYPES = ('bf16', 'f16', 'f32')


def parse_hlo(
    hlo: str,
) -> tuple[
    dict[str, tuple[str, str, str, list[str], str | None]],
    dict[str, list[str]],
]:
  """Parses the instructions and computations of an HLO text.

  Args:
    hlo: The compiled (optimized) HLO text.

  Returns:
    A tuple of instructions, a dict from instruction name to (dtype, shape, op,
    operand names, called computation name), and computations, a dict from
    computation name to the names of its instructions.
  """
  instructions = {}  # name -> (dtype, shape, op, operands, calls)
  computations = {}  # name -> [instruction names]
  computation = None
  for line in hlo.splitlines():
    if m := _HLO_COMPUTATION.match(line):
      computation = computations.setdefault(m.group(1), [])
    elif m := _HLO_INSTRUCTION.match(line):
      name, dtype, shape, op, operands, rest = m.groups()
      calls = re.search(r'calls=%([\w.-]+)', rest)
      instructions[name] = (
          dtype,
          shape,
          op,
          re.findall(r'%([\w.-]+)', operands),
          calls and calls.group(1),
      )
      computation.append(name)
  return instructions, computations


def _is_permutation_of(shape: str, dims: tuple[int, ...]) -> bool:
  """Returns whether the HLO shape string has the given dims in any order."""
  return sorted(int(d) for d in shape.split(',') if d) == sorted(dims)


def matmul_weight_feed(hlo: str, weight_shape: tuple[int, ...]) -> str:
  """Returns how an integer weight of `weight_shape` reaches the matmul.

  Args:
    hlo: The compiled (optimized) HLO text.
    weight_shape: The shape of the weight operand of the matmul. The matmul may
      consume it with its axes in any order, e.g. after relayouting batch axes.

  Returns:
    'int' if the matmul consumes the integer weight directly (XLA upcasts it
    inside the matmul), 'convert' if the weight is only converted (and possibly
    bitcast or copied) from an integer array before the matmul, or 'other' if
    other ops are applied to it, e.g. multiplying it by the scale.
  """
  instructions, computations = parse_hlo(hlo)
  feeds = set()
  for _, _, op, operands, _ in instructions.values():
    if op not in ('convolution', 'dot'):
      continue
    for operand in operands:
      w_dtype, w_shape, w_op, _, w_calls = instructions[operand]
      if not _is_permutation_of(w_shape, weight_shape):
        continue
      if w_dtype in _INT_DTYPES:
        feeds.add('int')
        continue
      body = [instructions[i] for i in computations.get(w_calls, ())]
      ops = {i[2] for i in body} - {'parameter'}
      params = [i[0] for i in body if i[2] == 'parameter']
      if (
          w_op == 'fusion'
          and ops <= {'convert', 'bitcast', 'copy'}
          and params
          and all(p in _INT_DTYPES for p in params)
      ):
        feeds.add('convert')
      else:
        feeds.add('other')
  if len(feeds) != 1:
    raise ValueError(f'Expected one matmul weight feed, got {feeds}.')
  return feeds.pop()


def dequantized_copies(hlo: str, shape: tuple[int, ...]) -> list[str]:
  """Returns the floating-point `multiply` instructions with `shape`.

  Dequantizing a QArray multiplies its (converted) qvalue by the scale, so a
  float multiply whose shape has the same dims as the QArray in any order is a
  dequantized copy of it, whether XLA fuses it into the matmul or materializes
  it.

  Args:
    hlo: The compiled (optimized) HLO text.
    shape: The shape of the QArray.

  Returns:
    The names of the matching instructions.
  """
  instructions, _ = parse_hlo(hlo)
  return [
      name
      for name, (dtype, i_shape, op, _, _) in instructions.items()
      if op == 'multiply'
      and dtype in _FLOAT_DTYPES
      and _is_permutation_of(i_shape, shape)
  ]


class DotGeneralTest(parameterized.TestCase):
  """More expensive TPU tests for dot_general, mainly on numerics."""

  def setUp(self):
    super().setUp()
    self._random_key = jax.random.key(42)

  def _make_array(self, shape, asymmetric=False):
    self._random_key, key = jax.random.split(self._random_key)
    if asymmetric:
      return jax.random.uniform(key, shape, jnp.float32)
    return jax.random.normal(key, shape, jnp.float32) / 10

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
          expected_mae=0.12,
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
          expected_mae=0.04,
          # fast/loop vs. slow is ~0.0045 on TPUs with native fp8 matmuls (e.g.
          # TPU7x), ~0.0025 on TPU v5e/v6e.
          max_impl_mae=0.005,
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
          expected_mae=0.11,
          # Only slow_dot_general supports nf4.
          disable_fast_dot_general=True,
          disable_loop_dot_general=True,
      ),
      dict(
          testcase_name='lhs_asymmetric',
          lhs_shape=(128, 512, 3),
          lhs_qtype=jnp.int8,
          lhs_asymmetric=True,
          rhs_shape=(512, 256, 3),
          rhs_qtype=jnp.int4,
          dimension_numbers=(([1], [0]), ([2], [2])),
          expected_mae=0.19,
          disable_loop_dot_general=True,
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
          expected_mae=0.128906,
          disable_loop_dot_general=True,
      ),
      dict(
          testcase_name='two_contractions',
          lhs_shape=(128, 2, 128, 32, 64),
          lhs_qtype=None,
          rhs_shape=(64, 32, 128, 128),
          rhs_qtype=jnp.int8,
          dimension_numbers=(([3, 4], [1, 0]), ([0], [3])),
          expected_mae=0.013,
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
          expected_mae=0.01,
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
          expected_mae=0.011,
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
          expected_mae=0.01,
          # Only slow_dot_general supports dequant_on_input.
          disable_fast_dot_general=True,
          disable_loop_dot_general=True,
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
      max_impl_mae: float = 0.003,
      disable_fast_dot_general: bool = False,
      disable_loop_dot_general: bool = False,
  ):
    if jax.devices()[0].platform != 'tpu':
      self.skipTest('Only run on TPU.')
    lhs = self._make_array(lhs_shape, lhs_asymmetric)
    rhs = self._make_array(rhs_shape, rhs_asymmetric)

    if lhs_qtype:
      lhs_how = qarray.HowToQuantize(
          qtype=lhs_qtype,
          channelwise_axes=(),
          tiled_axes={a: s for a, s in enumerate(lhs_tile_sizes) if s},
          calibration_method='minmax' if lhs_asymmetric else 'absmax',
      )
      q_lhs = qarray.quantize(lhs, lhs_how)
    else:
      q_lhs = lhs

    if rhs_qtype:
      rhs_how = qarray.HowToQuantize(
          qtype=rhs_qtype,
          channelwise_axes=(),
          tiled_axes={a: s for a, s in enumerate(rhs_tile_sizes) if s},
          calibration_method='minmax' if rhs_asymmetric else 'absmax',
      )
      q_rhs = qarray.quantize(rhs, rhs_how)
    else:
      q_rhs = rhs

    @jax.jit
    def _multi_dot_general(lhs, rhs, fp_res):
      slow_res = dot_general._slow_dot_general(lhs, rhs, dimension_numbers)
      if disable_fast_dot_general:
        fast_res = slow_res
      else:
        fast_res = dot_general._fast_dot_general(lhs, rhs, dimension_numbers)
      if disable_loop_dot_general:
        loop_res = slow_res
      else:
        loop_res = dot_general.loop_dot_general(lhs, rhs, dimension_numbers)
      return (
          rel_mae(slow_res, fp_res),
          rel_mae(slow_res, fast_res),
          rel_mae(slow_res, loop_res),
      )

    fp_res = jax.lax.dot_general(lhs, rhs, dimension_numbers)
    fp_mae, fast_mae, loop_mae = _multi_dot_general(q_lhs, q_rhs, fp_res)

    logging.info(
        'fp_mae=%s fast_mae=%s loop_mae=%s', fp_mae, fast_mae, loop_mae
    )
    self.assertLessEqual(fp_mae, expected_mae)
    # The error between slow vs fast, or slow vs loop should be purely due to
    # floating point imprecision, and should be small.
    self.assertLessEqual(fast_mae, max_impl_mae)
    self.assertLessEqual(loop_mae, max_impl_mae)

  @parameterized.parameters(128, 256, 512)
  def test_dot_general_subchannel_tile_sizes(self, tile_size: int):
    if jax.devices()[0].platform != 'tpu':
      self.skipTest('Only run on TPU.')
    lhs = self._make_array((128, 1024))
    rhs = self._make_array((1024, 256))
    q_lhs = qarray.quantize(
        lhs,
        qarray.HowToQuantize(
            qtype=jnp.int8,
            channelwise_axes=(0,),
            tiled_axes={1: tile_size},
        ),
    )
    q_rhs = qarray.quantize(
        rhs,
        qarray.HowToQuantize(
            qtype=jnp.int4,
            channelwise_axes=(1,),
            tiled_axes={0: tile_size},
        ),
    )
    dnums = (((1,), (0,)), ((), ()))
    fp_res = jax.lax.dot_general(lhs, rhs, dnums)
    res = jax.jit(dot_general.dot_general, static_argnums=(2,))(
        q_lhs, q_rhs, dnums
    )
    slow_res = jax.jit(dot_general._slow_dot_general, static_argnums=(2,))(
        q_lhs, q_rhs, dnums
    )
    self.assertLessEqual(rel_mae(slow_res, fp_res), 0.13)
    self.assertLessEqual(rel_mae(slow_res, res), 0.003)

  def test_get_min_tile_size_to_dequant_on_output_live_tpu(self):
    if jax.devices()[0].platform != 'tpu':
      self.skipTest('Only run on TPU.')
    expected_by_kind = {
        'TPU v5 lite': 128,
        'TPU v5p': 128,
        'TPU v6 lite': 256,
        'TPU v6e': 256,
        'TPU7': 512,
        'TPU7x': 512,
        'TPU8i': 256,
        'TPU8t': 256,
    }
    device_kind = jax.devices()[0].device_kind.split('\n', 1)[0]
    self.assertIn(device_kind, expected_by_kind)
    expected_threshold = expected_by_kind[device_kind]
    self.assertEqual(
        dot_general.get_min_tile_size_to_dequant_on_output(),
        expected_threshold,
    )

    k = 1024
    dnums = (([1], [0]), ([], []))
    for tile_size in (128, 256, 512):
      num_tiles = k // tile_size
      lhs = qarray.QArray(
          jnp.ones((16, k), jnp.int8),
          jnp.ones((16, num_tiles), jnp.bfloat16),
      )
      rhs = qarray.QArray(
          jnp.ones((k, 16), jnp.int4),
          jnp.ones((num_tiles, 16), jnp.bfloat16),
      )
      with (
          mock.patch.object(
              dot_general,
              '_fast_dot_general',
              wraps=dot_general._fast_dot_general,
          ) as mock_fast,
          mock.patch.object(
              dot_general,
              '_slow_dot_general',
              wraps=dot_general._slow_dot_general,
          ) as mock_slow,
      ):
        res = dot_general.dot_general(lhs, rhs, dnums)
        self.assertEqual(res.shape, (16, 16))
        if tile_size >= expected_threshold:
          mock_fast.assert_called_once()
          mock_slow.assert_not_called()
        else:
          mock_slow.assert_called_once()
          mock_fast.assert_not_called()

  @parameterized.named_parameters(
      dict(testcase_name='int8', qtype=jnp.int8),
      dict(testcase_name='int4', qtype=jnp.int4),
  )
  def test_unquantized_x_channelwise_upcasts_weight_in_xla(self, qtype):
    """Checks that XLA, not Qwix, upcasts the int weight for the matmul."""
    if jax.devices()[0].platform != 'tpu':
      self.skipTest('Only run on TPU.')
    m, k, n = 128, 1024, 512
    lhs = jnp.zeros((m, k), jnp.bfloat16)
    q_rhs = qarray.quantize(
        jnp.ones((k, n), jnp.bfloat16),
        qarray.HowToQuantize(qtype=qtype, channelwise_axes=(1,), tiled_axes={}),
    )
    dimension_numbers = (((1,), (0,)), ((), ()))

    def weight_feed(fn):
      hlo = (
          jax.jit(fn, static_argnums=2)
          .lower(lhs, q_rhs, dimension_numbers)
          .compile()
          .as_text()
      )
      return matmul_weight_feed(hlo, (k, n))

    # The matmul either takes the int weight directly (e.g. TPU v5e/v6e) or a
    # plain convert of it (e.g. int4 on TPU7x), with the scale on the output.
    self.assertIn(weight_feed(dot_general.dot_general), ('int', 'convert'))
    # Sanity check: the slow path multiplies the weight by the scale first.
    self.assertEqual(weight_feed(dot_general._slow_dot_general), 'other')

  @parameterized.named_parameters(
      dict(testcase_name='int8', qtype=jnp.int8),
      dict(testcase_name='int4', qtype=jnp.int4),
  )
  def test_unquantized_x_contracting_channelwise_folds_scale(self, qtype):
    """Checks that the fold path never dequantizes the QArray in XLA."""
    if jax.devices()[0].platform != 'tpu':
      self.skipTest('Only run on TPU.')
    # Attention P[b,k,g,t,s] . V[b,s,k,d] at decode (t=1) with per-token,
    # per-head scales [b,s,k,1], which vary along the contracting axis s.
    b, s, k, g, d = 32, 2048, 8, 4, 128
    p = jax.nn.softmax(self._make_array((b, k, g, 1, s)) * 30, axis=-1).astype(
        jnp.bfloat16
    )
    v = self._make_array((b, s, k, d)).astype(jnp.bfloat16)
    q_v = qarray.quantize(
        v,
        qarray.HowToQuantize(
            qtype=qtype, channelwise_axes=(0, 1, 2), tiled_axes={}
        ),
    )
    dimension_numbers = (((4,), (1,)), ((0, 1), (0, 2)))

    with mock.patch.object(
        dot_general,
        '_fold_scale_dot_general',
        wraps=dot_general._fold_scale_dot_general,
    ) as mock_fold:
      res = jax.jit(dot_general.dot_general, static_argnums=2)(
          p, q_v, dimension_numbers
      )
    mock_fold.assert_called_once()
    slow_res = jax.jit(dot_general._slow_dot_general, static_argnums=2)(
        p, q_v, dimension_numbers
    )
    fp_res = jax.lax.dot_general(
        p.astype(jnp.float32),
        qarray.dequantize(q_v.astype(jnp.float32)),
        dimension_numbers,
        precision=jax.lax.Precision.HIGHEST,
    )
    slow_mae = rel_mae(slow_res, res)
    fp_mae = rel_mae(fp_res, res.astype(jnp.float32))
    logging.info('slow_mae=%s fp_mae=%s', slow_mae, fp_mae)
    self.assertEqual(res.dtype, slow_res.dtype)
    # The fold path rounds P * scale instead of each dequantized value of V,
    # so it differs from the slow path by bf16 rounding only. Observed on TPU
    # v5e/v6e/v7x: slow_mae 0.0018-0.0021 and fp_mae 0.0021 for int8 and int4.
    self.assertLessEqual(slow_mae, 0.003)
    self.assertLessEqual(fp_mae, 0.003)

    def compile_fn(fn):
      return (
          jax.jit(fn, static_argnums=2)
          .lower(p, q_v, dimension_numbers)
          .compile()
      )

    compiled = compile_fn(dot_general.dot_general)
    hlo = compiled.as_text()
    slow_compiled = compile_fn(dot_general._slow_dot_general)
    slow_hlo = slow_compiled.as_text()
    # The matmul consumes the int values of V directly or via a plain convert,
    # and no dequantized copy of V exists, fused or not: the scale went to P.
    feed = matmul_weight_feed(hlo, (b, s, k, d))
    self.assertIn(feed, ('int', 'convert'))
    self.assertEmpty(dequantized_copies(hlo, (b, s, k, d)))
    # Sanity check: the slow path dequantizes V before the matmul.
    self.assertEqual(matmul_weight_feed(slow_hlo, (b, s, k, d)), 'other')
    self.assertNotEmpty(dequantized_copies(slow_hlo, (b, s, k, d)))
    # XLA fuses the relayout of int V into the matmul, whereas on the slow path
    # it materializes the dequantized bf16 V in HBM (temp_size_in_bytes ==
    # v_bf16_bytes). Verify that the fold path allocates no large temporary
    # (0 bytes on TPU v5e/v6e/v7x for both 'int' and 'convert' feeds).
    v_bf16_bytes = v.size * 2
    temp_bytes = compiled.memory_analysis().temp_size_in_bytes
    slow_temp_bytes = slow_compiled.memory_analysis().temp_size_in_bytes
    logging.info(
        'feed=%s temp_bytes=%s slow_temp_bytes=%s v_bf16_bytes=%s',
        feed,
        temp_bytes,
        slow_temp_bytes,
        v_bf16_bytes,
    )
    self.assertLess(temp_bytes, v_bf16_bytes // 4)


if __name__ == '__main__':
  absltest.main()
