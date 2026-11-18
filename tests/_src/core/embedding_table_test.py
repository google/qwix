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

from absl.testing import absltest
from absl.testing import parameterized
from jax import numpy as jnp
import numpy as np
from qwix._src.core import embedding_table
from qwix._src.core import qarray

_SCHEMES = (
    ('fixed_range', embedding_table.FixedRangeScheme(-1.0, 1.0)),
    ('row_absmax', embedding_table.RowAbsmaxScheme()),
)


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
    # to float32.
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
        embedding_table.dequantize_table(q_one), absmax_rows[:1]
    )


if __name__ == '__main__':
  absltest.main()
