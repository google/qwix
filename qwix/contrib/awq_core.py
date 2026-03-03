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
"""AWQ (Activation-aware Weight Quantization) algorithm.

This is a JAX implementation of the AWQ algorithm from:
https://arxiv.org/abs/2306.00978

AWQ identifies salient weights based on activation magnitudes and applies
equivalent transformations to improve quantization accuracy. The key insight
is that not all weights are equally important - weights connected to channels
with high activation magnitudes are more critical to preserve accurately.

AWQ scales are always per-channel (shape: 1, in_features). When combined with
groupwise quantization, the per-channel scales help protect salient channels
within each group, while the quantization uses per-group scales.
"""

import jax
import jax.numpy as jnp
from qwix._src.core import qarray


def compute_act_scale(x: jax.Array, axis: int = 0) -> jax.Array:
  """Computes per-channel activation magnitudes.

  This function calculates the mean absolute value of activations for each
  channel, which is used to identify salient weight channels in AWQ.

  Args:
    x: Input activations with shape (n_samples, in_features) when axis=0, or
      (in_features, n_samples) when axis=1.
    axis: The sample axis to reduce over. Default is 0 (samples in first dim).

  Returns:
    Per-channel mean absolute activation with shape (in_features,).
  """
  return jnp.mean(jnp.abs(x), axis=axis)


def search_optimal_scales(
    w: jax.Array,
    activation_scales: jax.Array,
    how: qarray.HowToQuantize,
    n_grid: int = 20,
    min_scale: float = 1e-4,
) -> jax.Array:
  """Searches for optimal per-channel scaling factors using grid search.

  The AWQ algorithm searches for an optimal exponent 'ratio' such that
  scales = act_scale^ratio provides the best quantization accuracy. Larger
  ratios give more protection to salient channels (those with high activations).

  The key insight of AWQ is to minimize OUTPUT error (activation-weighted),
  not raw weight error. Channels with higher activation magnitudes contribute
  more to the output, so their weight errors should be weighted more heavily.

  AWQ scales are always per-channel, even when using groupwise quantization.
  This allows protecting salient channels within each quantization group.

  Args:
    w: Weight matrix with shape (out_features, in_features), where in_features
      is the contraction dimension.
    activation_scales: Per-channel activation scale with shape (in_features,).
    how: How to quantize the weights.
    n_grid: Number of grid points to search. Default is 20.
    min_scale: Minimum scale value to prevent division by zero.

  Returns:
    The optimal per-channel scaling factors with shape (1, in_features).
  """
  ratios = jnp.linspace(0.0, 1.0, n_grid)

  # Normalize act_scale to prevent numerical issues.
  activation_scales_normalized = activation_scales / (
      activation_scales.max() + 1e-8
  )
  activation_scales_normalized = jnp.clip(
      activation_scales_normalized, min=min_scale
  )

  def compute_loss_and_scales(ratio: jax.Array) -> tuple[jax.Array, jax.Array]:
    # Compute per-channel scales from activation magnitudes.
    scales = jnp.power(activation_scales_normalized, ratio)
    scales = jnp.clip(scales, min=min_scale)
    # Normalize scales to prevent extreme values.
    scales = scales / jnp.sqrt(jnp.maximum(scales.max() * scales.min(), 1e-8))
    scales = scales.reshape(1, -1)

    # Apply per-channel scaling and quantize (possibly with groupwise quant).
    w_scaled = w * scales
    w_q = qarray.quantize(w_scaled, how)
    w_dq = qarray.dequantize(w_q)
    # Restore original scale with per-channel division.
    w_restored = w_dq / scales

    # Compute activation-weighted error.
    # This approximates output error: ||W @ X - W_q @ X||^2
    weight_error = w - w_restored
    weighted_error = weight_error * activation_scales.reshape(1, -1)
    loss = jnp.mean(weighted_error**2)
    return loss, scales

  # Vectorize over all ratios for efficient computation.
  losses, scales = jax.vmap(compute_loss_and_scales)(ratios)

  # Find the ratio with minimum loss.
  best_idx = jnp.argmin(losses)
  best_scales = scales[best_idx]

  return best_scales


def quantize_weight(
    w: jax.Array,
    activation_scales: jax.Array,
    how: qarray.HowToQuantize,
    n_grid: int = 20,
) -> tuple[qarray.QArray, jax.Array]:
  """Quantizes a weight matrix using AWQ.

  This function finds optimal per-channel scaling factors based on activation
  magnitudes and applies them before quantization to preserve salient channels.

  The returned scales are per-channel (1, in_features). When using groupwise
  quantization, the per-channel scales help protect important channels within
  each group while the quantization itself uses per-group scales.

  Args:
    w: Weight matrix with shape (out_features, in_features), where in_features
      is the contraction dimension.
    activation_scales: Per-channel activation scale from compute_act_scale with
      shape (in_features,).
    how: How to quantize the weights.
    n_grid: Number of grid points for scale search.

  Returns:
    A tuple of (W_q, scales):
      - W_q: The quantized weight matrix as a QArray. The weights have been
        scaled by per-channel factors before quantization.
      - scales: The per-channel scaling factors that were applied, shape
        (1, in_features). Dequantizing and dividing by scales gives the best
        approximation of the original weights.
  """
  optimal_scales = search_optimal_scales(w, activation_scales, how, n_grid)

  # Scale up salient channels before quantization.
  w_scaled = w * optimal_scales

  # Quantize the scaled weights (may use groupwise quantization).
  w_q = qarray.quantize(w_scaled, how)

  return w_q, optimal_scales
