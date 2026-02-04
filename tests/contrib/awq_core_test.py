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
"""Tests for AWQ algorithm."""

import functools
import logging

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np
from qwix._src.core import qarray
from qwix.contrib import awq_core


def rel_rmse(x: jax.Array, y: jax.Array) -> jax.Array:
  return jnp.sqrt(jnp.mean((x - y) ** 2)) / jnp.sqrt(jnp.mean(y**2))


class AwqCoreTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name="g128",
          groupsize=128,
      ),
      dict(
          testcase_name="g256",
          groupsize=256,
      ),
  )
  def test_quantize_weight(self, groupsize):
    # Create random weights.
    w = jax.nn.initializers.lecun_normal()(
        jax.random.key(0), (256, 512), jnp.float32
    )

    # AWQ shines when activations have outliers (salient channels).
    # We simulate this by creating activations where a few channels have much
    # larger magnitudes than others.
    # x shape: (n_samples, in_features) = (1024, 512)
    x = jax.random.normal(jax.random.key(1), (1024, 512), jnp.float32)

    # Introduce outliers in specific channels.
    # Channels 0, 10, 100 will have 100x variance.
    scale = jnp.ones((512,))
    scale = scale.at[jnp.array([0, 10, 100])].set(100.0)
    x = x * scale

    how = qarray.HowToQuantize(
        qtype=jnp.int8,
        channelwise_axes=[0],
        tiled_axes={1: groupsize},
    )
    # RTN (Round-To-Nearest) quantization simply rounds weights.
    # It doesn't account for the fact that some input channels (0, 10, 100)
    # are much more important for the output.
    w_rtn = qarray.quantize(w, how)

    # AWQ computes optimal scaling factors based on activation magnitudes.
    # It should identify channels 0, 10, 100 as important and scale them up
    # before quantization to preserve their precision.
    act_scales = awq_core.compute_act_scale(x, axis=0)
    w_awq, optimal_scales = jax.jit(
        functools.partial(awq_core.quantize_weight, how=how)
    )(w, act_scales)

    self.assertEqual(
        jax.tree.map(lambda x: (x.shape, x.dtype), w_awq),
        jax.tree.map(lambda x: (x.shape, x.dtype), w_rtn),
    )
    self.assertEqual(optimal_scales.shape, (1, 512))

    # dequant loss.
    # For AWQ, we need to divide by scales after dequantizing to compare with
    # original w
    w_rtn_deq = qarray.dequantize(w_rtn)
    w_awq_deq = qarray.dequantize(w_awq) / optimal_scales

    mse_rtn = rel_rmse(w_rtn_deq, w)
    mse_awq = rel_rmse(w_awq_deq, w)
    logging.info("dequant loss rtn: %s vs. awq: %s", mse_rtn, mse_awq)

    # matmul loss.
    # w @ x.T if x is (samples, in). w is (out, in).
    # target = w @ x.T
    target = w @ x.T
    y_rtn = w_rtn_deq @ x.T
    y_awq = w_awq_deq @ x.T

    mse_rtn_out = rel_rmse(y_rtn, target)
    mse_awq_out = rel_rmse(y_awq, target)
    self.assertLess(mse_awq_out, mse_rtn_out)

  def test_normalize_weight(self):
    """Tests that normalize_weight correctly reshapes the weight tensor."""
    w = jnp.arange(2 * 3 * 4).reshape(2, 3, 4)
    w2, restore_shape = awq_core.normalize_weight(w, 1)
    # Contraction axis 1 (size 3) moves to last.
    # shape becomes (2*4, 3) = (8, 3)
    self.assertEqual(w2.shape, (8, 3))
    w3 = restore_shape(w2)
    self.assertEqual(w3.shape, (2, 3, 4))
    np.testing.assert_array_equal(w, w3)


if __name__ == "__main__":
  absltest.main()
