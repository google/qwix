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
"""QEP (Quantization Error Propagation) core algorithms.

This contains the pure-function QEP primitives for computing QEP statistics
and applying weight correction. QEP extends GPTQ by accounting for
quantization noise in input activations from previous layers.

Reference: https://arxiv.org/abs/2504.09629
"""

# We try to use the same naming as in the PyTorch implementation, thus
# pylint: disable=invalid-name

import jax
import jax.numpy as jnp
from qwix.contrib import gptq_core


def compute_qep_stats(
    X_q: jax.Array, X_float: jax.Array
) -> dict[str, jax.Array]:
  """Computes QEP (Quantization Error Propagation) statistics.

  QEP extends GPTQ by accounting for quantization noise in input activations.
  Instead of minimizing ||W @ X - W_q @ X||^2 (standard GPTQ), QEP minimizes
  ||W @ X - W_q @ X_q||^2 where X_q are quantized inputs from previous layers.

  This requires two statistics:
    - hessian: X_q @ X_q^T (Hessian from quantized inputs)
    - hessian_delta: (X_float - X_q) @ X_q^T (cross-correlation of input error)

  Args:
    X_q: Quantized input activations, shape (in_features, n_samples).
    X_float: Float input activations, shape (in_features, n_samples).

  Returns:
    A dict with 'hessian' and 'hessian_delta', both (in_features, in_features).
  """
  delta = X_float - X_q
  hessian = X_q @ X_q.T
  hessian_delta = delta @ X_q.T
  return {'hessian': hessian, 'hessian_delta': hessian_delta}


def weight_correct(
    W: jax.Array,
    H: jax.Array,
    H_delta: jax.Array,
    *,
    correction_factor: float = 0.5,
    damping_factor: float = 0.01,
) -> jax.Array:
  """Applies QEP weight correction to compensate for input quantization noise.

  This adjusts W so that W_corrected @ X_q better approximates W @ X_float,
  partially canceling the effect of quantized inputs.

  The correction formula is:
    W_corrected = W + correction_factor * (W @ H_delta @ H_inv)

  Args:
    W: Weight matrix, shape (rows, columns) where columns = in_features.
    H: Hessian from quantized inputs, shape (columns, columns).
    H_delta: Cross-correlation matrix from compute_qep_stats, shape (columns,
      columns).
    correction_factor: Weight correction factor. 0.0 = no correction, 1.0 = full
      correction. Default 0.5 per QEP paper recommendations.
    damping_factor: Dampening factor for Hessian inversion as a fraction of the
      average diagonal. Default 0.01.

  Returns:
    The corrected weight matrix, same shape as W.
  """
  columns = H.shape[0]
  assert H.shape == (columns, columns)
  assert H_delta.shape == (columns, columns)
  assert W.shape[1] == columns

  # Handle dead columns (zero diagonal in H).
  H_diag = jnp.diag(H)
  dead = H_diag == 0
  H = jnp.where(dead & jnp.eye(columns, dtype=bool), 1.0, H)
  W = jnp.where(dead, 0.0, W)

  # Dampen the Hessian (using higher dampening than GPTQ).
  damp = damping_factor * jnp.mean(H_diag)
  diag = jnp.arange(columns)
  H = H.at[diag, diag].add(damp)

  # Compute H_inv via Cholesky factorization.
  # Unlike quantize_weight, we do NOT re-Cholesky to upper triangular --
  # weight_correct needs Hinv directly.
  H = jnp.linalg.cholesky(H)
  Hinv = gptq_core.cholesky_inverse(H)

  # Apply weight correction.
  W = W + correction_factor * (W @ H_delta @ Hinv)
  return W
