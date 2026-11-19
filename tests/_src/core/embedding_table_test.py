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

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax import numpy as jnp
import numpy as np
from qwix._src.core import embedding_table
from qwix._src.core import qarray

_F32_UNIT_ROUNDOFF = 2.0**-24
_SCHEMES = (
    ('fixed_range', embedding_table.FixedRangeScheme(-1.0, 1.0)),
    ('row_absmax', embedding_table.RowAbsmaxScheme()),
)
_PLACEMENTS = ('tc', 'sc_gather', 'sc')


class EmbeddingTableTest(parameterized.TestCase):

  def test_validation_errors(self):
    for bad_bounds in (
        (-np.inf, 1.0),
        (-1.0, np.nan),
        (1.0, -1.0),
        (0.0, 0.0),
        (-1.0, 2.0),
        (-2e38, 2e38),
        (-1e-38, 1e-38),
    ):
      with self.subTest(bad_bounds=bad_bounds):
        with self.assertRaises(ValueError):
          embedding_table.FixedRangeScheme(*bad_bounds)

    scheme = embedding_table.RowAbsmaxScheme()
    table = np.ones((4, 8), dtype=np.float32)
    qtable = embedding_table.quantize_table(table, scheme)

    z = jnp.zeros((4, 8), jnp.int8)
    s = jnp.ones((4, 1), jnp.float32)
    for bad_qt in (
        qarray.QArray(z, s, zero_point=jnp.zeros((4, 1), jnp.int8)),
        qarray.QArray(jnp.zeros((4, 8), jnp.uint4), s, qtype='nf4'),
        qarray.QArray(z, s, qtype='uint8_centered'),
        qarray.QArray(z, jnp.ones((4, 2), jnp.float32)),
        qarray.QArray(z, jnp.ones((4, 1), jnp.int32)),
    ):
      with self.assertRaises(ValueError):
        embedding_table.dequantize_table(bad_qt)

    with self.assertRaises(ValueError):
      embedding_table.quantize_table(table, scheme, chunk_bytes=0)
    with self.assertRaises(ValueError):
      embedding_table.quantize_table(np.ones((4,), np.float32), scheme)
    with self.assertRaises(ValueError):
      embedding_table.quantize_table(np.zeros((0, 8), np.float32), scheme)
    with self.assertRaises(ValueError):
      embedding_table.quantize_table(np.ones((4, 8), np.int32), scheme)
    with self.assertRaises(ValueError):
      embedding_table.dequantize_table(qtable, chunk_bytes=0)
    with self.assertRaises(ValueError):
      embedding_table.take(
          qtable, jnp.array([0]), placement='invalid'  # type: ignore[arg-type]
      )
    with self.assertRaises(ValueError):
      embedding_table.take(qtable, jnp.array([0.0], dtype=jnp.float32))
    with self.assertRaises(ValueError):
      embedding_table.take(qtable, jnp.array([True, False]))
    with self.assertRaises(ValueError):
      embedding_table.lookup(
          qtable, jnp.array([0, 1]), jnp.ones((2,), jnp.float32)
      )
    with self.assertRaises(ValueError):
      embedding_table.lookup(
          qtable,
          jnp.array([[0, 1]], jnp.int32),
          jnp.ones((1, 2), jnp.int32),
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

    # bfloat16 scale: dequantize_table multiplies in bfloat16 before upcasting
    # to float32; take/lookup reject non-float32 scales.
    qtable_bf16 = dataclasses.replace(
        full, scale=np.asarray(full.scale, dtype=jnp.bfloat16)
    )
    dq_bf16 = embedding_table.dequantize_table(qtable_bf16)
    self.assertEqual(dq_bf16.dtype, np.float32)
    vals_f32 = (
        full.qvalue.astype(np.float32) - np.float32(127.5)
        if full.qtype == 'uint8_centered'
        else full.qvalue.astype(np.float32)
    )
    expected_bf16 = np.asarray(
        np.asarray(vals_f32, dtype=jnp.bfloat16) * qtable_bf16.scale,
        dtype=np.float32,
    )
    np.testing.assert_array_equal(dq_bf16, expected_bf16)
    with self.assertRaises(ValueError):
      embedding_table.take(qtable_bf16, jnp.array([0], dtype=jnp.int32))
    with self.assertRaises(ValueError):
      embedding_table.lookup(
          qtable_bf16,
          jnp.array([[0]], dtype=jnp.int32),
          jnp.ones((1, 1), dtype=jnp.float32),
          placement='sc',
      )

  def test_quantize_table_exact_grids(self):
    # FixedRangeScheme: exact 256-bucket grid inversion, half-up tie at 0.0,
    # and out-of-range clipping.
    grid = np.linspace(-1.0, 1.0, 256, dtype=np.float32).reshape(2, 128)
    q_fixed = embedding_table.quantize_table(
        grid, embedding_table.FixedRangeScheme(-1.0, 1.0)
    )
    self.assertEqual(q_fixed.qtype, 'uint8_centered')
    self.assertEqual(q_fixed.scale.shape, (1, 1))
    np.testing.assert_array_equal(
        q_fixed.qvalue, np.arange(256, dtype=np.uint8).reshape(2, 128)
    )
    np.testing.assert_array_equal(
        q_fixed.scale, np.full((1, 1), 2.0 / 255.0, dtype=np.float32)
    )
    np.testing.assert_array_equal(
        embedding_table.dequantize_table(q_fixed),
        (np.arange(256, dtype=np.float32).reshape(2, 128) - np.float32(127.5))
        * np.float32(2.0 / 255.0),
    )
    q_edges = embedding_table.quantize_table(
        np.array([[-2.0, 0.0, 2.0]], dtype=np.float32),
        embedding_table.FixedRangeScheme(-1.0, 1.0),
    )
    np.testing.assert_array_equal(
        q_edges.qvalue, np.array([[0, 128, 255]], dtype=np.uint8)
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
    inv_127 = np.float32(1.0) / np.float32(127.0)
    np.testing.assert_array_equal(
        q_abs.scale,
        np.array(
            [[np.float32(127.0) * inv_127], [np.float32(1e-7) * inv_127]],
            dtype=np.float32,
        ),
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
    q_one = embedding_table.quantize_table(
        absmax_rows[:1], embedding_table.RowAbsmaxScheme()
    )
    np.testing.assert_array_equal(
        np.asarray(embedding_table.take(q_one, jnp.array([0]))),
        absmax_rows[:1],
    )

  @parameterized.product(scheme_entry=_SCHEMES, placement=_PLACEMENTS)
  def test_take_and_lookup_placements(self, scheme_entry, placement):
    _, scheme = scheme_entry
    rng = np.random.default_rng(1)
    table = rng.uniform(-1.0, 1.0, size=(32, 16)).astype(np.float32)
    qtable_host = embedding_table.quantize_table(table, scheme)
    ref_table = embedding_table.dequantize_table(qtable_host)
    qtable = jax.tree.map(jnp.asarray, qtable_host)

    ids = jnp.array([[0, 1, 31, 1], [5, -1, -32, 3]], dtype=jnp.int32)
    wrapped_ids = np.where(np.asarray(ids) < 0, np.asarray(ids) + 32, ids)

    take_fn = jax.jit(
        functools.partial(embedding_table.take, placement=placement)
    )
    rows = take_fn(qtable, ids)
    self.assertEqual(rows.dtype, jnp.float32)
    np.testing.assert_array_equal(np.asarray(rows), ref_table[wrapped_ids])

    # Weighted bag lookup (including 0-weight padding and empty K=0 bags).
    weights = jnp.array(
        [[1.0, 0.5, 0.25, 0.0], [0.75, 1.25, 0.5, 0.0]], dtype=jnp.float32
    )
    lookup_fn = jax.jit(
        functools.partial(embedding_table.lookup, placement=placement)
    )
    bags = lookup_fn(qtable, ids, weights)
    self.assertEqual(bags.dtype, jnp.float32)
    ref_terms = (
        ref_table[wrapped_ids].astype(np.float64)
        * np.asarray(weights, dtype=np.float64)[..., None]
    )
    ref_bags = np.sum(ref_terms, axis=1)
    # 2 roundings per term (dequantize, multiply) + (K - 1) additions.
    tol = (
        (ids.shape[1] + 1)
        * _F32_UNIT_ROUNDOFF
        * np.sum(np.abs(ref_terms), axis=1)
    )
    self.assertTrue(np.all(np.abs(np.asarray(bags) - ref_bags) <= tol))

    empty_bags = lookup_fn(
        qtable,
        jnp.zeros((2, 0), dtype=jnp.int32),
        jnp.zeros((2, 0), dtype=jnp.float32),
    )
    np.testing.assert_array_equal(
        np.asarray(empty_bags), np.zeros((2, 16), dtype=np.float32)
    )

    # OOB rows in take and OOB bags in lookup (even with weight 0.0) -> NaN.
    oob_ids = jnp.array([[0, 32], [-33, 1], [2, 3]], dtype=jnp.int32)
    oob_rows = take_fn(qtable, oob_ids)
    self.assertTrue(np.all(np.isnan(np.asarray(oob_rows[0, 1]))))
    self.assertTrue(np.all(np.isnan(np.asarray(oob_rows[1, 0]))))
    np.testing.assert_array_equal(np.asarray(oob_rows[2, 0]), ref_table[2])

    oob_weights = jnp.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], jnp.float32)
    oob_bags = lookup_fn(qtable, oob_ids, oob_weights)
    self.assertTrue(np.all(np.isnan(np.asarray(oob_bags[0]))))
    self.assertTrue(np.all(np.isnan(np.asarray(oob_bags[1]))))
    self.assertTrue(np.all(np.isfinite(np.asarray(oob_bags[2]))))

  def test_ids_dtypes_and_wrapping(self):
    # Small table (N < 256): verify wrap and OOB across narrow & wide dtypes.
    n_small = 16
    table_small = np.arange(n_small * 4, dtype=np.float32).reshape(n_small, 4)
    qtable_small = jax.tree.map(
        jnp.asarray,
        embedding_table.quantize_table(
            table_small, embedding_table.RowAbsmaxScheme()
        ),
    )
    ref_small = embedding_table.dequantize_table(qtable_small)
    np.testing.assert_array_equal(
        np.asarray(embedding_table.take(qtable_small, jnp.int32(-1))),
        ref_small[15],
    )

    for dtype in (jnp.int8, jnp.int16, jnp.int32):
      ids = jnp.array([-17, -16, -1, 0, 15, 16], dtype=dtype)
      out = np.asarray(embedding_table.take(qtable_small, ids))
      self.assertTrue(np.all(np.isnan(out[0])))
      np.testing.assert_array_equal(out[1:5], ref_small[[0, 15, 0, 15]])
      self.assertTrue(np.all(np.isnan(out[5])))

    for dtype in (jnp.uint8, jnp.uint16, jnp.uint32):
      high_oob = 2**31 if dtype == jnp.uint32 else 16
      ids = jnp.array([0, 15, 16, high_oob], dtype=dtype)
      out = np.asarray(embedding_table.take(qtable_small, ids))
      np.testing.assert_array_equal(out[:2], ref_small[[0, 15]])
      self.assertTrue(np.all(np.isnan(out[2:])))

    with jax.enable_x64():
      ids_i64 = jnp.array([-(2**40), -16, -1, 0, 15, 2**40], dtype=jnp.int64)
      out_i64 = np.asarray(embedding_table.take(qtable_small, ids_i64))
      self.assertTrue(np.all(np.isnan(out_i64[0])))
      np.testing.assert_array_equal(out_i64[1:5], ref_small[[0, 15, 0, 15]])
      self.assertTrue(np.all(np.isnan(out_i64[5])))

      ids_u64 = jnp.array([0, 15, 16, 2**63], dtype=jnp.uint64)
      out_u64 = np.asarray(embedding_table.take(qtable_small, ids_u64))
      np.testing.assert_array_equal(out_u64[:2], ref_small[[0, 15]])
      self.assertTrue(np.all(np.isnan(out_u64[2:])))

    # Large table (N > 65_536): narrow signed/unsigned ids must not wrap N.
    n_large = 70_000
    qvalue_large = jnp.broadcast_to(
        (jnp.arange(n_large, dtype=jnp.int32) % 127).astype(jnp.int8)[:, None],
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
          embedding_table.take(qtable_large, jnp.array(vals, dtype=dtype))
      )
      expected = np.asarray(qvalue_large[np.array(expected_rows)], np.float32)
      np.testing.assert_array_equal(out, expected)

  @parameterized.product(scheme_entry=_SCHEMES, placement=_PLACEMENTS)
  def test_tpu_export_compute_type(self, scheme_entry, placement):
    _, scheme = scheme_entry
    is_fixed = isinstance(scheme, embedding_table.FixedRangeScheme)
    qtable_spec = qarray.QArray(
        jax.ShapeDtypeStruct((32, 16), jnp.uint8 if is_fixed else jnp.int8),  # type: ignore[arg-type]
        jax.ShapeDtypeStruct((1 if is_fixed else 32, 1), jnp.float32),  # type: ignore[arg-type]
        qtype='uint8_centered' if is_fixed else jnp.int8,
    )
    ids_spec = jax.ShapeDtypeStruct((4, 3), jnp.int32)
    weights_spec = jax.ShapeDtypeStruct((4, 3), jnp.float32)
    expected_attr = (
        '_xla_compute_type = "dense"'
        if placement == 'tc'
        else '_xla_compute_type = "sparseoffload"'
    )

    exported_take = jax.export.export(
        jax.jit(functools.partial(embedding_table.take, placement=placement)),
        platforms=('tpu',),
    )(qtable_spec, ids_spec)
    self.assertIn(expected_attr, exported_take.mlir_module())

    exported_lookup = jax.export.export(
        jax.jit(functools.partial(embedding_table.lookup, placement=placement)),
        platforms=('tpu',),
    )(qtable_spec, ids_spec, weights_spec)
    self.assertIn(expected_attr, exported_lookup.mlir_module())


if __name__ == '__main__':
  absltest.main()
