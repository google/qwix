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

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
from qwix._src.core import stochastic_rounding


class StochasticRoundingTest(parameterized.TestCase):

  def test_uniform_noise(self):
    key = jax.random.PRNGKey(0)
    shape = (2, 3)
    noise_fn = stochastic_rounding.get_noise_fn(
        "uniform", key=key, channelwise_noise_axes=(0,)
    )
    noise = noise_fn(shape)
    self.assertEqual(noise.shape, (2, 1))
    noise = jnp.broadcast_to(noise, shape)
    # Check that the noise is the same along the shared axis.
    self.assertTrue(jnp.all(noise[0, 0] == noise[0, 1]))
    self.assertTrue(jnp.all(noise[1, 0] == noise[1, 1]))
    # Check that the noise is different along the non-shared axis.
    self.assertFalse(jnp.all(noise[0, 0] == noise[1, 0]))

  def test_low_bit_uniform_noise(self):
    key = jax.random.PRNGKey(0)
    shape = (2, 3)
    noise_fn = stochastic_rounding.get_noise_fn(
        "low_bit_uniform",
        key=key,
        channelwise_noise_axes=(0,),
    )
    noise = noise_fn(shape)
    self.assertEqual(noise.shape, (2, 1))
    noise = jnp.broadcast_to(noise, shape)
    # Check that the noise is the same along the shared axis.
    self.assertTrue(jnp.all(noise[0, 0] == noise[0, 1]))
    self.assertTrue(jnp.all(noise[1, 0] == noise[1, 1]))
    # Check that the noise is different along the non-shared axis.
    self.assertFalse(jnp.all(noise[0, 0] == noise[1, 0]))
    self.assertTrue(jnp.all(noise > -0.5))
    self.assertTrue(jnp.all(noise < 0.5))


if __name__ == "__main__":
  absltest.main()
