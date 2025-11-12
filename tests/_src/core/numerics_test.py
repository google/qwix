# Copyright 2024 Google LLC
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
from absl.testing import absltest
import jax
from jax import numpy as jnp
from qwix._src.core import numerics
from qwix._src.core import stochastic_rounding


class NumericsTest(absltest.TestCase):

  def _assert_equal(self, a, b):
    self.assertTrue(jnp.array_equal(a, b), f"{a=} {b=}")

  def test_invalid_cases(self):
    with self.assertRaisesRegex(
        ValueError, "doesn't support asymmetric quantization"
    ):
      numerics.get_asymmetric_bound("nf4")

  def test_convert_to(self):
    self._assert_equal(
        numerics.convert_to(
            jnp.array([1.2, 3.5, -127.5, 129, -1300]), jnp.int8
        ),
        jnp.array([1, 4, -128, 127, -128], jnp.int8),
    )
    self._assert_equal(
        numerics.convert_to(jnp.array([1.2, 3.5, 8, -1300]), jnp.int4),
        jnp.array([1, 4, 7, -8], jnp.int4),
    )

  def test_inf(self):
    self._assert_equal(
        numerics.convert_to(jnp.array([jnp.inf, -jnp.inf]), jnp.int8),
        jnp.array([127, -128], jnp.int8),
    )

  def test_arbitrary_integer_dtype(self):
    self._assert_equal(numerics.get_symmetric_bound("int6"), 31.5)
    self._assert_equal(
        numerics.convert_to(jnp.array([1.2, 3.5, 129, -1300]), "int6"),
        jnp.array([1, 4, 31, -32], jnp.int8),
    )
    # jnp.int4 and "int4" should be the same.
    self._assert_equal(
        numerics.get_symmetric_bound("int4"),
        numerics.get_symmetric_bound(jnp.int4),
    )
    self._assert_equal(
        numerics.convert_to(jnp.array([1.2, 3.5, 129, -1300]), "int4"),
        numerics.convert_to(jnp.array([1.2, 3.5, 129, -1300]), jnp.int4),
    )
    self._assert_equal(
        numerics.get_symmetric_bound("int8"),
        numerics.get_symmetric_bound(jnp.int8),
    )
    self._assert_equal(
        numerics.convert_to(jnp.array([1.2, 3.5, 129, -1300]), "int8"),
        numerics.convert_to(jnp.array([1.2, 3.5, 129, -1300]), jnp.int8),
    )

  def test_uint(self):
    self._assert_equal(numerics.get_symmetric_bound("uint8"), 255.5)
    self._assert_equal(
        numerics.convert_to(jnp.array([-1.2, 3.5, 129, 1300]), "uint8"),
        jnp.array([0, 4, 129, 255], jnp.uint8),
    )

  def test_stochastic_rounding(self):
    key = jax.random.PRNGKey(0)
    x = jnp.full((10000,), 0.5)
    noise_fn = functools.partial(stochastic_rounding.uniform_noise, key)
    y = numerics.convert_to(x, jnp.int8, noise_fn=noise_fn)
    # Without stochastic rounding, this would be rounded to all zeros based on
    # round-half-to-even.
    self.assertAlmostEqual(jnp.mean(y), 0.5, delta=0.1)

    # Test with negative values.
    x = jnp.full((10000,), -0.5)
    _, subkey = jax.random.split(key)
    noise_fn = functools.partial(stochastic_rounding.uniform_noise, subkey)
    y = numerics.convert_to(x, jnp.int8, noise_fn=noise_fn)
    # Without stochastic rounding, this would be rounded to all zeros based on
    # round-half-to-even.
    self.assertAlmostEqual(jnp.mean(y), -0.5, delta=0.1)

  def test_nf4(self):
    self._assert_equal(
        numerics.convert_to(jnp.array([-1.0, -0.5, 0.0, 0.8, 1.0]), "nf4"),
        jnp.array([0, 2, 7, 14, 15], jnp.uint4),
    )
    self._assert_equal(
        numerics.convert_from(jnp.array([0, 7, 15], jnp.uint4), "nf4"),
        jnp.array([-1.0, 0.0, 1.0]),
    )
    # Shape should be preserved.
    self._assert_equal(
        numerics.convert_to(jnp.zeros((2, 3, 4)), "nf4").shape,
        (2, 3, 4),
    )
    self._assert_equal(
        numerics.convert_from(jnp.zeros((2, 3, 4), jnp.uint4), "nf4").shape,
        (2, 3, 4),
    )

  def test_mxfp(self):
    with self.subTest("mxfp8_bound"):
      bound = numerics.get_symmetric_bound("mxfp8")
      finfo = jnp.finfo(jnp.float8_e4m3fn)
      self.assertEqual(bound, float(finfo.max))

    with self.subTest("mxfp8_conversion"):
      finfo = jnp.finfo(jnp.float8_e4m3fn)
      # Test values: in-range, out-of-range high, out-of-range low
      in_array = jnp.array([10.0, 500.0, -1000.0], dtype=jnp.float32)
      converted = numerics.convert_to(in_array, "mxfp8")

      self.assertEqual(converted.dtype, jnp.float8_e4m3fn)
      expected = jnp.array(
          [10.0, finfo.max, finfo.min], dtype=jnp.float8_e4m3fn
      )
      self._assert_equal(converted, expected)

    with self.subTest("mxfp4_bound"):
      bound = numerics.get_symmetric_bound("mxfp4")
      finfo = jnp.finfo(jnp.float4_e2m1fn)
      self.assertEqual(bound, float(finfo.max))

    with self.subTest("mxfp4_conversion"):
      finfo = jnp.finfo(jnp.float4_e2m1fn)
      # Test values: in-range, out-of-range high, out-of-range low
      in_array = jnp.array([1.5, 10.0, -20.0], dtype=jnp.float32)
      converted = numerics.convert_to(in_array, "mxfp4")

      self.assertEqual(converted.dtype, jnp.float4_e2m1fn)
      expected = jnp.array([1.5, finfo.max, finfo.min], dtype=jnp.float4_e2m1fn)
      self._assert_equal(converted, expected)

    with self.subTest("mxfp_convert_from"):
      # convert_from should be a no-op for these float types
      in_mxfp8 = jnp.array([1.0, 2.0], dtype=jnp.float8_e4m3fn)
      out_mxfp8 = numerics.convert_from(in_mxfp8, "mxfp8")
      self.assertIs(in_mxfp8, out_mxfp8)  # Should return the exact same object

      in_mxfp4 = jnp.array([1.0, 1.5], dtype=jnp.float4_e2m1fn)
      out_mxfp4 = numerics.convert_from(in_mxfp4, "mxfp4")
      self.assertIs(in_mxfp4, out_mxfp4)


if __name__ == "__main__":
  absltest.main()
