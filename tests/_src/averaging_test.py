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

from absl.testing import absltest
from absl.testing import parameterized
from jax import numpy as jnp
from qwix._src import averaging


class SimpleMovingAverageTest(parameterized.TestCase):

  def test_init(self):
    # Check that init initializes the quantization statistics correctly.
    aggregator = averaging.SimpleMovingAverage()
    calibration = {"a": jnp.array(1.0), "b": jnp.array(2.0)}
    quant_stat = aggregator.init(calibration)
    self.assertEqual(quant_stat["count"], 0)
    self.assertEqual(quant_stat["sum_of_a"], 0.0)
    self.assertEqual(quant_stat["sum_of_b"], 0.0)

  def test_update(self):
    # Check that update updates the quantization statistics correctly.
    aggregator = averaging.SimpleMovingAverage()
    calibration = {"a": jnp.array(1.0), "b": jnp.array(2.0)}
    quant_stat = aggregator.init(calibration)
    quant_stat = aggregator.update(quant_stat, calibration)
    self.assertEqual(quant_stat["count"], 1)
    self.assertEqual(quant_stat["sum_of_a"], 1.0)
    self.assertEqual(quant_stat["sum_of_b"], 2.0)

  def test_get_calibration(self):
    # Check that get_calibration returns the average calibration.
    aggregator = averaging.SimpleMovingAverage()
    calibration = {"a": jnp.array(1.0), "b": jnp.array(2.0)}
    quant_stat = aggregator.init(calibration)

    # Check for 1 update
    quant_stat = aggregator.update(quant_stat, calibration)
    self.assertEqual(aggregator.get_calibration(quant_stat), calibration)

    # Check for 2 updates
    calibration2 = {"a": jnp.array(3.0), "b": jnp.array(4.0)}
    quant_stat = aggregator.update(quant_stat, calibration2)
    quant_stat_answer = {"a": jnp.array(2.0), "b": jnp.array(3.0)}
    self.assertEqual(aggregator.get_calibration(quant_stat), quant_stat_answer)

  def test_get_calibration_with_bootstrap_steps(self):
    # Check get_calibration returns the average with bootstrap steps.
    aggregator = averaging.SimpleMovingAverage(bootstrap_steps=1)
    calibration = {"a": jnp.array(1.0), "b": jnp.array(2.0)}

    # Returns default calibration when samples < bootstrap steps.
    quant_stat = aggregator.init(calibration)
    quant_stat = aggregator.update(quant_stat, calibration)
    default_calibration = {"a": jnp.array(-1.0), "b": jnp.array(-2.0)}
    self.assertEqual(
        aggregator.get_calibration(quant_stat, default_calibration),
        default_calibration,
    )

    # Returns the average calibration when samples >= bootstrap steps.
    calibration2 = {"a": jnp.array(3.0), "b": jnp.array(4.0)}
    quant_stat = aggregator.update(quant_stat, calibration2)
    quant_stat_answer = {"a": jnp.array(2.0), "b": jnp.array(3.0)}
    self.assertEqual(
        aggregator.get_calibration(quant_stat, default_calibration),
        quant_stat_answer,
    )

  def test_get_calibration_default_calibration_keys(self):
    # Check get_calibration requires default_calibration to have same keys
    aggregator = averaging.SimpleMovingAverage(bootstrap_steps=2)
    calibration = {"a": jnp.array(1.0), "b": jnp.array(2.0)}

    # Same keys
    quant_stat = aggregator.init(calibration)
    quant_stat = aggregator.update(quant_stat, calibration)
    default_calibration = {"a": jnp.array(-1.0), "b": jnp.array(-2.0)}
    self.assertEqual(
        aggregator.get_calibration(quant_stat, default_calibration),
        default_calibration,
    )

    # Extra keys
    default_calibration = {
        "a": jnp.array(3.0),
        "b": jnp.array(4.0),
        "c": jnp.array(5.0),
    }
    self.assertRaises(
        ValueError,
        aggregator.get_calibration,
        quant_stat,
        default_calibration,
    )

    # Too few keys
    default_calibration = {"c": jnp.array(5.0)}
    self.assertRaises(
        ValueError,
        aggregator.get_calibration,
        quant_stat,
        default_calibration,
    )


if __name__ == "__main__":
  absltest.main()
