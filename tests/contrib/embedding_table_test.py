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

import dataclasses
import functools
from typing import cast

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax import numpy as jnp
import numpy as np
from qwix._src.core import qarray
from qwix.contrib import embedding_table

_SCHEMES = (
    ('fixed_range', embedding_table.FixedRangeScheme(-1.0, 127.0 / 128.0)),
    ('row_absmax', embedding_table.RowAbsmaxScheme()),
)


def _make_qtable(
    *,
    shape: tuple[int, ...] = (32, 16),
    qtype: jax.typing.DTypeLike = jnp.int8,
    channelwise_axes: tuple[int, ...] = (0,),
    tiled_axes: dict[int, int] | None = None,
    calibration_method: str = 'absmax',
    scale_dtype: jax.typing.DTypeLike = jnp.float32,
    seed: int = 0,
) -> qarray.QArray:
  rng = np.random.default_rng(seed)
  table = rng.uniform(-1.5, 1.5, size=shape).astype(np.float32)
  return qarray.quantize_api(
      jnp.asarray(table),
      qtype,
      channelwise_axes=channelwise_axes,
      tiled_axes=tiled_axes,
      calibration_method=calibration_method,
      scale_dtype=scale_dtype,
  )


class EmbeddingTableTest(parameterized.TestCase):

  def test_validation_errors(self):
    for bad_bounds, pattern in (
        ((-np.inf, 1.0), 'Bounds must be finite'),
        ((-1.0, np.nan), 'Bounds must be finite'),
        ((1.0, -1.0), 'Expected lower < upper'),
        ((0.0, 0.0), 'Expected lower < upper'),
        ((-1.0, 1.0), 'zero-aligned bounds'),
        ((-127.0 / 128.0, 1.0), 'zero-aligned bounds'),
        ((-1.0, 2.0), 'zero-aligned bounds'),
        ((-1e308, 1e308), 'not cleanly representable in float32'),
        ((1e39, 2e39), 'not cleanly representable in float32'),
        ((-2e38, 2e38), 'not cleanly representable in float32'),
        ((-1e-38, 127.0 / 128.0 * 1e-38), 'not cleanly representable'),
    ):
      with self.subTest(bad_bounds=bad_bounds):
        with self.assertRaisesRegex(ValueError, pattern):
          embedding_table.FixedRangeScheme(*bad_bounds)

    for good_bounds in (
        (-5.12, 5.08),
        (-128.0 / 127.0, 1.0),
        (-1.00787, 1.0),
    ):
      with self.subTest(good_bounds=good_bounds):
        embedding_table.FixedRangeScheme(*good_bounds)

    scheme = embedding_table.RowAbsmaxScheme()
    table = np.ones((4, 8), dtype=np.float32)
    qtable = embedding_table.quantize_table(table, scheme)

    z = jnp.zeros((4, 8), jnp.int8)
    s = jnp.ones((4, 1), jnp.float32)
    for idx, (bad_qt, pattern) in enumerate((
        (
            qarray.QArray(z, s, zero_point=jnp.zeros((4, 1), jnp.int8)),
            'zero_point',
        ),
        (
            qarray.QArray(jnp.zeros((4, 8), jnp.uint4), s, qtype='nf4'),
            'Unsupported qtype',
        ),
        (
            qarray.QArray(jnp.zeros((4, 8), jnp.uint8), s, qtype=jnp.int8),
            'Unsupported qtype',
        ),
        (
            qarray.QArray(jnp.zeros((4, 8), jnp.uint8), s, qtype=jnp.uint8),
            'Unsupported qtype',
        ),
        (
            qarray.QArray(z, jnp.ones((4, 2), jnp.float32)),
            'Expected scale shape',
        ),
        (
            qarray.QArray(z, jnp.ones((4, 1), jnp.int32)),
            'Unsupported scale dtype',
        ),
    )):
      with self.subTest(idx=idx, pattern=pattern):
        with self.assertRaisesRegex(ValueError, pattern):
          embedding_table.dequantize_table(bad_qt)

    with self.assertRaisesRegex(ValueError, 'Unsupported scheme'):
      embedding_table.quantize_table(table, object())  # type: ignore[arg-type]
    with self.assertRaisesRegex(ValueError, 'chunk_bytes must be positive'):
      embedding_table.quantize_table(table, scheme, chunk_bytes=0)
    with self.assertRaisesRegex(ValueError, 'Expected 2D table'):
      embedding_table.quantize_table(np.ones((4,), np.float32), scheme)
    with self.assertRaisesRegex(ValueError, 'Expected 2D table'):
      embedding_table.quantize_table(np.zeros((0, 8), np.float32), scheme)
    with self.assertRaisesRegex(ValueError, 'Expected 2D table'):
      embedding_table.quantize_table(np.zeros((4, 0), np.float32), scheme)
    with self.assertRaisesRegex(ValueError, 'floating-point'):
      embedding_table.quantize_table(np.ones((4, 8), np.int32), scheme)
    with self.assertRaisesRegex(ValueError, 'chunk_bytes must be positive'):
      embedding_table.dequantize_table(qtable, chunk_bytes=0)

    for _, test_scheme in _SCHEMES:
      for bad_val, dtype in (
          (np.nan, np.float32),
          (np.inf, np.float32),
          (-np.inf, np.float32),
          (1e39, np.float64),
      ):
        bad_table = np.zeros((4, 4), dtype=dtype)
        bad_table[3, 2] = bad_val
        with self.subTest(scheme=test_scheme, bad_val=bad_val, dtype=dtype):
          with self.assertRaisesRegex(ValueError, 'finite'):
            embedding_table.quantize_table(
                bad_table, test_scheme, chunk_bytes=1
            )

  @parameterized.named_parameters(*_SCHEMES)
  def test_quantize_and_dequantize_table(self, scheme):
    rng = np.random.default_rng(0)
    table = rng.uniform(-1.5, 1.5, size=(17, 16)).astype(np.float32)
    table[0] = 0.0  # All-zero row tests absmax epsilon floor and zero tie.

    full = embedding_table.quantize_table(table, scheme)
    self.assertIsInstance(full.qvalue, np.ndarray)
    self.assertIsInstance(full.scale, np.ndarray)
    self.assertIsNone(full.zero_point)

    # Chunk-size invariance on both quantize_table and dequantize_table.
    full_dq = embedding_table.dequantize_table(full)
    for chunk_bytes in (1, 16 * 4 * 3):
      chunked = embedding_table.quantize_table(
          table, scheme, chunk_bytes=chunk_bytes
      )
      np.testing.assert_array_equal(chunked.qvalue, full.qvalue)
      np.testing.assert_array_equal(chunked.scale, full.scale)
      np.testing.assert_array_equal(
          embedding_table.dequantize_table(full, chunk_bytes=chunk_bytes),
          full_dq,
      )

    # JAX array inputs (float32 and bfloat16) slice per chunk and still return
    # host NumPy int8 qvalue + float32 scale leaves.
    from_jax = embedding_table.quantize_table(
        jnp.asarray(table), scheme, chunk_bytes=16 * 4 * 3
    )
    self.assertIsInstance(from_jax.qvalue, np.ndarray)
    self.assertIsInstance(from_jax.scale, np.ndarray)
    np.testing.assert_array_equal(from_jax.qvalue, full.qvalue)
    np.testing.assert_array_equal(from_jax.scale, full.scale)

    bf16_table = jnp.asarray(table, dtype=jnp.bfloat16)
    from_bf16 = embedding_table.quantize_table(
        bf16_table, scheme, chunk_bytes=16 * 4 * 3
    )
    self.assertIsInstance(from_bf16.qvalue, np.ndarray)
    self.assertIsInstance(from_bf16.scale, np.ndarray)
    self.assertEqual(from_bf16.qvalue.dtype, np.int8)
    self.assertEqual(from_bf16.scale.dtype, np.float32)
    from_bf16_f32 = embedding_table.quantize_table(
        np.asarray(bf16_table, dtype=np.float32), scheme
    )
    np.testing.assert_array_equal(from_bf16.qvalue, from_bf16_f32.qvalue)
    np.testing.assert_array_equal(from_bf16.scale, from_bf16_f32.scale)

  def test_dequantize_table_bfloat16_scale(self):
    # dequantize_table multiplies in bfloat16 before upcasting to float32:
    # 127 * 1.0078125 = 127.9921875 -> 128.0 in bf16, and
    # 3 * 1.0078125 = 3.0234375 -> 3.03125 in bf16 (ties to even).
    qvalue = np.array([[-128, 0, 3, 127], [-128, 0, 3, 127]], dtype=np.int8)
    per_tensor = qarray.QArray(
        cast(jax.Array, qvalue),
        cast(jax.Array, np.array([[1.0078125]], dtype=jnp.bfloat16)),
        qtype=jnp.int8,
    )
    dq_per_tensor = embedding_table.dequantize_table(per_tensor, chunk_bytes=1)
    self.assertEqual(dq_per_tensor.dtype, np.float32)
    np.testing.assert_array_equal(
        dq_per_tensor,
        np.array(
            [[-129.0, 0.0, 3.03125, 128.0], [-129.0, 0.0, 3.03125, 128.0]],
            dtype=np.float32,
        ),
    )

    per_row = qarray.QArray(
        cast(jax.Array, qvalue),
        cast(
            jax.Array,
            np.array([[1.0078125], [0.50390625]], dtype=jnp.bfloat16),
        ),
        qtype=jnp.int8,
    )
    dq_per_row = embedding_table.dequantize_table(per_row, chunk_bytes=1)
    self.assertEqual(dq_per_row.dtype, np.float32)
    np.testing.assert_array_equal(
        dq_per_row,
        np.array(
            [[-129.0, 0.0, 3.03125, 128.0], [-64.5, 0.0, 1.515625, 64.0]],
            dtype=np.float32,
        ),
    )

  def test_quantize_table_exact_grids(self):
    # FixedRangeScheme: exact zero-aligned 256-bucket int8 grid inversion
    # (including -128, 0, 127), half-up rounding, and out-of-range clipping.
    grid = np.arange(-128, 128, dtype=np.float32).reshape(2, 128) / np.float32(
        128.0
    )
    q_fixed = embedding_table.quantize_table(
        grid, embedding_table.FixedRangeScheme(-1.0, 127.0 / 128.0)
    )
    self.assertEqual(q_fixed.qtype, jnp.int8)
    self.assertEqual(q_fixed.scale.shape, (1, 1))
    np.testing.assert_array_equal(
        q_fixed.qvalue, np.arange(-128, 128, dtype=np.int8).reshape(2, 128)
    )
    np.testing.assert_array_equal(
        q_fixed.scale, np.array([[0.0078125]], dtype=np.float32)
    )
    np.testing.assert_array_equal(
        embedding_table.dequantize_table(q_fixed), grid
    )
    q_edges = embedding_table.quantize_table(
        np.array(
            [[-2.0, -0.5 / 128.0, 0.0, 0.5 / 128.0, 2.0]], dtype=np.float32
        ),
        embedding_table.FixedRangeScheme(-1.0, 127.0 / 128.0),
    )
    np.testing.assert_array_equal(
        q_edges.qvalue, np.array([[-128, 0, 0, 1, 127]], dtype=np.int8)
    )
    # Non-power-of-two range pins float32 `(x - lower) * inv_step` order
    # (`-5.1 -> -128` for [-5.12, 5.08], vs. `-127` for naive `x / step`).
    q_non_unit = embedding_table.quantize_table(
        np.array([[-5.12, -5.1, 0.0, 5.08]], dtype=np.float32),
        embedding_table.FixedRangeScheme(-5.12, 5.08),
    )
    np.testing.assert_array_equal(
        q_non_unit.qvalue, np.array([[-128, -128, 0, 127]], dtype=np.int8)
    )

    # RowAbsmaxScheme: exact [-127, 127] grid inversion and 1e-7 floor.
    absmax_rows = np.array(
        [
            np.arange(-127, 128, dtype=np.float32),
            np.zeros(255, dtype=np.float32),
        ],
        dtype=np.float32,
    )
    q_abs = embedding_table.quantize_table(
        absmax_rows, embedding_table.RowAbsmaxScheme()
    )
    self.assertEqual(q_abs.qtype, jnp.int8)
    self.assertEqual(q_abs.scale.shape, (2, 1))
    np.testing.assert_array_equal(
        q_abs.qvalue[0], np.arange(-127, 128, dtype=np.int8)
    )
    np.testing.assert_array_equal(q_abs.qvalue[1], np.zeros(255, dtype=np.int8))
    np.testing.assert_array_equal(
        q_abs.scale,
        np.array([[1.0], [7.874016e-10]], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        embedding_table.dequantize_table(q_abs)[0], absmax_rows[0]
    )
    for alt_qtype in ('int8', np.dtype('int8')):
      np.testing.assert_array_equal(
          embedding_table.dequantize_table(
              dataclasses.replace(q_abs, qtype=alt_qtype)
          )[0],
          absmax_rows[0],
      )
    # Non-unit absmax (4.5) pins both `absmax * (1 / 127)` scale (0.03543307 vs.
    # 0.035433073 for `/ 127`) and `x * (127 / absmax)` op order (-2.25 -> -63
    # vs. -64 for `/ scale`; 2.25 -> 64 vs. 63 for `/ (absmax / 127)`), for both
    # float32 and bfloat16 inputs.
    for table_dtype in (np.float32, jnp.bfloat16):
      q_abs_non_unit = embedding_table.quantize_table(
          np.array([[-4.5, -2.25, 0.0, 2.25, 4.5]], dtype=table_dtype),
          embedding_table.RowAbsmaxScheme(),
      )
      np.testing.assert_array_equal(
          q_abs_non_unit.qvalue,
          np.array([[-127, -63, 0, 64, 127]], dtype=np.int8),
      )
      np.testing.assert_array_equal(
          q_abs_non_unit.scale,
          np.array([[0.03543307]], dtype=np.float32),
      )
    q_one = embedding_table.quantize_table(
        absmax_rows[:1], embedding_table.RowAbsmaxScheme()
    )
    np.testing.assert_array_equal(
        embedding_table.dequantize_table(q_one), absmax_rows[:1]
    )
    # Half-up ties on half-integers (half-to-even maps -1.5, 0.5, 2.5 to
    # -2, 0, 2 instead of -1, 1, 3).
    q_ties = embedding_table.quantize_table(
        np.array([[-2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 127.0]], dtype=np.float32),
        embedding_table.RowAbsmaxScheme(),
    )
    np.testing.assert_array_equal(
        q_ties.qvalue, np.array([[-2, -1, 0, 1, 2, 3, 127]], dtype=np.int8)
    )

  def test_sparsecore_take_validation_errors(self):
    qtable = _make_qtable()
    ids = jnp.array([0, 1], dtype=jnp.int32)
    qtable_3d = _make_qtable(shape=(4, 4, 4))
    asym_qt = _make_qtable(calibration_method='minmax')
    bf16_qt = _make_qtable(scale_dtype=jnp.bfloat16)
    subch_qt = _make_qtable(channelwise_axes=(0,), tiled_axes={1: 8})
    nf4_qt = _make_qtable(qtype='nf4')
    int32_table = jnp.zeros((8, 4), jnp.int32)

    with self.assertRaisesRegex(NotImplementedError, "'raise'"):
      embedding_table.sparsecore_take(qtable, ids, mode='raise')
    with self.assertRaisesRegex(ValueError, 'Invalid mode'):
      embedding_table.sparsecore_take(qtable, ids, mode='bad')
    with self.assertRaisesRegex(ValueError, 'integer indices'):
      embedding_table.sparsecore_take(
          qtable, jnp.array([0.0], dtype=jnp.float32)
      )
    with self.assertRaisesRegex(ValueError, 'integer indices'):
      embedding_table.sparsecore_take(qtable, jnp.array([True, False]))
    with self.assertRaisesRegex(ValueError, 'scalar fill_value'):
      embedding_table.sparsecore_take(
          qtable,
          ids,
          fill_value=[1.0, 2.0],  # type: ignore[arg-type]
      )
    with self.assertRaisesRegex(ValueError, 'scalar fill_value'):
      embedding_table.sparsecore_take(
          qtable,
          ids,
          mode='clip',
          fill_value=jnp.array([1.0, 2.0]),  # type: ignore[arg-type]
      )
    with self.assertRaisesRegex(ValueError, '2D table'):
      embedding_table.sparsecore_take(qtable_3d, ids)
    with self.assertRaisesRegex(ValueError, 'Asymmetric'):
      embedding_table.sparsecore_take(asym_qt, ids)
    with self.assertRaisesRegex(ValueError, 'float32 scale'):
      embedding_table.sparsecore_take(bf16_qt, ids)
    with self.assertRaisesRegex(ValueError, 'scale shape'):
      embedding_table.sparsecore_take(subch_qt, ids)
    with self.assertRaisesRegex(ValueError, 'Unsupported qtype'):
      embedding_table.sparsecore_take(nf4_qt, ids)
    with self.assertRaisesRegex(ValueError, 'floating-point'):
      embedding_table.sparsecore_take(int32_table, ids)

  @parameterized.product(
      channelwise_axes=((), (0,)),
      mode_and_fill=(
          (None, None),
          (None, 0.0),
          ('fill', -3.5),
          ('clip', 0.0),
          ('wrap', 0.0),
      ),
  )
  def test_sparsecore_take_parity_with_jnp_take(
      self, channelwise_axes, mode_and_fill
  ):
    mode, fill_value = mode_and_fill
    qtable = _make_qtable(channelwise_axes=channelwise_axes)
    # Include the extreme -128 int8 code to check signed conversion on SC.
    qvalue = qtable.qvalue.at[0, 0].set(jnp.int8(-128))
    qtable = qarray.QArray(
        qvalue, qtable.scale, qtable.zero_point, qtable.qtype
    )
    ref_table = qarray.dequantize(qtable)

    ids = jnp.array([[0, 1, 31, -1], [-32, 32, -33, 5]], dtype=jnp.int32)
    take_fn = jax.jit(
        functools.partial(
            embedding_table.sparsecore_take, mode=mode, fill_value=fill_value
        )
    )
    out = take_fn(qtable, ids)
    expected = jnp.take(
        ref_table, ids, axis=0, mode=mode, fill_value=fill_value
    )
    self.assertEqual(out.dtype, expected.dtype)
    np.testing.assert_array_equal(np.asarray(out), np.asarray(expected))

    # Empty integer and untyped/float indices return zeros of expected shape.
    for empty_ids in (jnp.zeros((2, 0), dtype=jnp.int32), jnp.zeros((0,))):
      empty_out = take_fn(qtable, empty_ids)
      empty_expected = jnp.take(
          ref_table, empty_ids, axis=0, mode=mode, fill_value=fill_value
      )
      self.assertEqual(empty_out.shape, empty_expected.shape)
      self.assertEqual(empty_out.dtype, empty_expected.dtype)
      np.testing.assert_array_equal(
          np.asarray(empty_out), np.asarray(empty_expected)
      )

  @parameterized.parameters(jnp.float32, jnp.bfloat16)
  def test_sparsecore_take_unquantized_and_host_qarray(self, dtype):
    rng = np.random.default_rng(2)
    table = jnp.asarray(rng.uniform(-1.0, 1.0, size=(16, 8)), dtype=dtype)
    ids = jnp.array([[0, -1, 15, 16]], dtype=jnp.int32)
    out = jax.jit(
        functools.partial(embedding_table.sparsecore_take, fill_value=0.0)
    )(table, ids)
    expected = jnp.take(table, ids, axis=0, fill_value=0.0)
    self.assertEqual(out.dtype, expected.dtype)
    np.testing.assert_array_equal(np.asarray(out), np.asarray(expected))

    empty_ids = np.zeros((0,))
    empty_out = embedding_table.sparsecore_take(table, empty_ids)
    empty_expected = jnp.take(table, empty_ids, axis=0)
    self.assertEqual(empty_out.shape, empty_expected.shape)
    self.assertEqual(empty_out.dtype, empty_expected.dtype)

    for _, scheme in _SCHEMES:
      host_qt = embedding_table.quantize_table(table, scheme)
      ref_qt = embedding_table.dequantize_table(host_qt)
      qt_out = embedding_table.sparsecore_take(host_qt, ids, fill_value=0.0)
      qt_expected = jnp.take(jnp.asarray(ref_qt), ids, axis=0, fill_value=0.0)
      np.testing.assert_array_equal(np.asarray(qt_out), np.asarray(qt_expected))

  def test_sparsecore_take_ids_dtypes_and_wrapping(self):
    n_small = 16
    qtable_small = _make_qtable(shape=(n_small, 4), seed=3)
    ref_small = np.asarray(qarray.dequantize(qtable_small))

    for alt_qtype in ('int8', np.dtype('int8')):
      alt_qt = dataclasses.replace(qtable_small, qtype=alt_qtype)
      np.testing.assert_array_equal(
          np.asarray(
              embedding_table.sparsecore_take(
                  alt_qt, jnp.array([0, 15], dtype=jnp.int32)
              )
          ),
          ref_small[[0, 15]],
      )

    np.testing.assert_array_equal(
        np.asarray(
            embedding_table.sparsecore_take(qtable_small, jnp.int32(-1))
        ),
        ref_small[15],
    )
    np.testing.assert_array_equal(
        np.asarray(
            embedding_table.sparsecore_take(
                qtable_small, jnp.array([16], dtype=jnp.int32), fill_value=0.0
            )
        ),
        np.zeros((1, 4), dtype=np.float32),
    )
    np.testing.assert_array_equal(
        np.asarray(
            embedding_table.sparsecore_take(
                jnp.asarray(ref_small), jnp.int32(-1)
            )
        ),
        ref_small[15],
    )
    qtable_per_tensor = _make_qtable(
        shape=(n_small, 4), channelwise_axes=(), seed=4
    )
    ref_per_tensor = np.asarray(qarray.dequantize(qtable_per_tensor))
    np.testing.assert_array_equal(
        np.asarray(
            embedding_table.sparsecore_take(qtable_per_tensor, jnp.int32(-1))
        ),
        ref_per_tensor[15],
    )

    for dtype in (jnp.int8, jnp.int16, jnp.int32):
      ids = jnp.array([-17, -16, -1, 0, 15, 16], dtype=dtype)
      out = np.asarray(embedding_table.sparsecore_take(qtable_small, ids))
      self.assertTrue(np.all(np.isnan(out[0])))
      np.testing.assert_array_equal(out[1:5], ref_small[[0, 15, 0, 15]])
      self.assertTrue(np.all(np.isnan(out[5])))
      np.testing.assert_array_equal(
          np.asarray(
              embedding_table.sparsecore_take(qtable_small, ids, mode='wrap')
          ),
          ref_small[[15, 0, 15, 0, 15, 0]],
      )

    for dtype in (jnp.uint8, jnp.uint16, jnp.uint32, jnp.uint64):
      with jax.enable_x64(dtype == jnp.uint64):
        ids = jnp.array([0, 15, 16, jnp.iinfo(dtype).max], dtype=dtype)
        out = np.asarray(embedding_table.sparsecore_take(qtable_small, ids))
        np.testing.assert_array_equal(out[:2], ref_small[[0, 15]])
        self.assertTrue(np.all(np.isnan(out[2:])))
        np.testing.assert_array_equal(
            np.asarray(
                embedding_table.sparsecore_take(qtable_small, ids, mode='clip')
            ),
            ref_small[[0, 15, 15, 15]],
        )
        np.testing.assert_array_equal(
            np.asarray(
                embedding_table.sparsecore_take(qtable_small, ids, mode='wrap')
            ),
            ref_small[[0, 15, 0, 15]],
        )

    with jax.enable_x64():
      ids_i64 = jnp.array([-(2**40), -16, -1, 0, 15, 2**40], dtype=jnp.int64)
      out_i64 = np.asarray(
          embedding_table.sparsecore_take(qtable_small, ids_i64)
      )
      self.assertTrue(np.all(np.isnan(out_i64[0])))
      np.testing.assert_array_equal(out_i64[1:5], ref_small[[0, 15, 0, 15]])
      self.assertTrue(np.all(np.isnan(out_i64[5])))

    # Large table (N > 65_536): narrow signed/unsigned ids must not wrap N.
    # Modulus 123 keeps every tested row index below distinct.
    n_large = 70_000
    qvalue_large = jnp.broadcast_to(
        (jnp.arange(n_large, dtype=jnp.int32) % 123).astype(jnp.int8)[:, None],
        (n_large, 2),
    )
    qtable_large = qarray.QArray(
        qvalue_large, jnp.ones((n_large, 1), dtype=jnp.float32), qtype=jnp.int8
    )
    for dtype, vals, expected_rows in (
        (jnp.uint8, [0, 1, 255], [0, 1, 255]),
        (jnp.uint16, [0, 1000, 65535], [0, 1000, 65535]),
        (jnp.int8, [-128, -1, 0, 127], [n_large - 128, n_large - 1, 0, 127]),
        (jnp.int16, [-1000, -1, 1000], [n_large - 1000, n_large - 1, 1000]),
    ):
      out = np.asarray(
          embedding_table.sparsecore_take(
              qtable_large, jnp.array(vals, dtype=dtype)
          )
      )
      expected = np.asarray(qvalue_large[np.array(expected_rows)], np.float32)
      np.testing.assert_array_equal(out, expected)

    # Non-1024-multiple batch (>1024) on a non-1024-multiple table compiles on
    # SparseCore without hitting 2D [K, 1] layout-copy mismatches.
    large_batch_ids = jnp.arange(1500, dtype=jnp.int32)
    large_batch_out = np.asarray(
        embedding_table.sparsecore_take(qtable_large, large_batch_ids)
    )
    np.testing.assert_array_equal(
        large_batch_out, np.asarray(qvalue_large[:1500], np.float32)
    )

    # D = 1 large unaligned tables (float32, per-row QArray, per-tensor QArray)
    # stage 1D operands into SparseCore without hitting 2D [K, 1] layout issues.
    d1_f32 = jnp.arange(n_large, dtype=jnp.float32)[:, None]
    d1_ids = jnp.array([0, 1, 1000, n_large - 1], dtype=jnp.int32)
    np.testing.assert_array_equal(
        np.asarray(embedding_table.sparsecore_take(d1_f32, d1_ids)),
        np.asarray(d1_f32[d1_ids]),
    )
    d1_qval = qvalue_large[:, :1]
    d1_qt_per_row = qarray.QArray(
        d1_qval, jnp.ones((n_large, 1), dtype=jnp.float32), qtype=jnp.int8
    )
    # Row codes at [0, 1, 1000, 69999] % 123 are [0, 1, 16, 12].
    np.testing.assert_array_equal(
        np.asarray(embedding_table.sparsecore_take(d1_qt_per_row, d1_ids)),
        np.array([[0.0], [1.0], [16.0], [12.0]], dtype=np.float32),
    )
    d1_qt_per_tensor = qarray.QArray(
        d1_qval, jnp.full((1, 1), 2.0, dtype=jnp.float32), qtype=jnp.int8
    )
    np.testing.assert_array_equal(
        np.asarray(embedding_table.sparsecore_take(d1_qt_per_tensor, d1_ids)),
        np.array([[0.0], [2.0], [32.0], [24.0]], dtype=np.float32),
    )

  def _extract_tpu_compute_on(
      self, closed_jaxpr
  ) -> list[tuple[str, list[str]]]:
    """Extracts `(compute_type, op_names)` from TPU `compute_on` branches."""
    matches: list[tuple[str, list[str]]] = []
    for eqn in closed_jaxpr.jaxpr.eqns:
      if eqn.primitive.name != 'cond':
        continue
      branches_platforms = eqn.params.get('branches_platforms')
      if branches_platforms is None:
        continue
      for branch, platforms in zip(
          eqn.params['branches'], branches_platforms, strict=True
      ):
        if platforms is None or 'tpu' not in platforms:
          continue
        for sub_eqn in branch.jaxpr.eqns:
          if sub_eqn.primitive.name == 'compute_on':
            op_names = [
                inner.primitive.name for inner in sub_eqn.params['jaxpr'].eqns
            ]
            matches.append((sub_eqn.params['compute_type'], op_names))
    return matches

  @parameterized.parameters(False, True)
  def test_tpu_export_sparsecore_compute_on(self, per_row_scale):
    qtable_spec = qarray.QArray(
        cast(jax.Array, jax.ShapeDtypeStruct((32, 16), jnp.int8)),
        cast(
            jax.Array,
            jax.ShapeDtypeStruct((32 if per_row_scale else 1, 1), jnp.float32),
        ),
        qtype=jnp.int8,
    )
    ids_spec = jax.ShapeDtypeStruct((4, 3), jnp.int32)
    op_fn = embedding_table.sparsecore_take

    exported = jax.export.export(jax.jit(op_fn), platforms=('tpu',))(
        qtable_spec, ids_spec
    )
    matches = self._extract_tpu_compute_on(
        jax.make_jaxpr(op_fn)(qtable_spec, ids_spec)
    )
    self.assertIn('_xla_compute_type = "sparseoffload"', exported.mlir_module())
    self.assertLen(matches, 1)
    compute_type, ops = matches[0]
    self.assertEqual(compute_type, 'tpu_sparsecore')
    self.assertLessEqual(set(ops), {'broadcast_in_dim', 'gather'})
    self.assertEqual(ops.count('gather'), 2 if per_row_scale else 1)


if __name__ == '__main__':
  absltest.main()
