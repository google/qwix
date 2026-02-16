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
"""QEP (Quantization Error Propagation) core algorithm.

This module contains the pure-function QEP primitives for computing QEP
statistics and applying weight correction. QEP extends existing PTQ methods
like GPTQ by accounting for quantization noise in input activations from
previous layers.

Paper Reference: https://arxiv.org/abs/2504.09629
Implementation Reference: https://github.com/FujitsuResearch/qep
"""

import jax
import jax.numpy as jnp
from qwix.contrib import gptq_core


def compute_qep_stats(
    x_quantized: jax.Array, x_float: jax.Array
) -> dict[str, jax.Array]:
  """Computes QEP (Quantization Error Propagation) statistics.

  QEP extends existing PTQ methods like GPTQ by accounting for quantization
  noise in input activations. Instead of minimizing ||W @ X - W_q @ X||^2
  (standard GPTQ), QEP minimizes ||W @ X - W_q @ X_q||^2 where X_q are
  quantized inputs from previous layers.

  This requires two statistics:
    - hessian: X_q @ X_q^T (Hessian of the QEP objective with respect to the
      weights, capturing the covariance of the quantized input activations)
    - hessian_delta: (X_float - X_q) @ X_q^T (cross-correlation of input error)

  Args:
    x_quantized: Quantized input activations, shape (in_features, n_samples).
    x_float: Float input activations, shape (in_features, n_samples).

  Returns:
    A dict with 'hessian' and 'hessian_delta', both (in_features, in_features).
  """
  delta = x_float - x_quantized

  # The Hessian is a matrix of second derivatives.
  # For a single linear layer, the QEP objective function E is the squared error
  # between the original model and the quantized model:
  # E(W_q) = || W @ X - W_q @ X_q ||_2^2
  #
  # The first derivative of the objective function E with respect to the
  # quantized weights W_q is:
  # dE/dW_q = -2(W @ X - W_q @ X_q) @ X_q^T
  #
  # The second derivative of the objective function E with respect to the
  # quantized weights W_q is:
  # H = 2 * X_q @ X_q^T
  # (Note: We can drop the constant 2 since it doesn't affect the optimization)
  hessian = x_quantized @ x_quantized.T

  # Cross-correlation of input error
  hessian_delta = delta @ x_quantized.T
  return {'hessian': hessian, 'hessian_delta': hessian_delta}


def weight_correct(
    w: jax.Array,
    h: jax.Array,
    h_delta: jax.Array,
    *,
    correction_factor: float = 0.5,
    dampening_factor: float = 0.01,
) -> jax.Array:
  """Applies QEP weight correction to compensate for input quantization noise.

  This adjusts W so that W_corrected @ X_q better approximates W @ X_float,
  partially canceling the effect of quantized inputs.

  The correction formula is:
    W_corrected = W + correction_factor * (W @ H_delta @ H_inv)

  Reference: qep/src/gptq.py Helper.run_weight_correct

  Args:
    w: Weight matrix, shape (rows, columns) where columns = in_features.
    h: Hessian from quantized inputs, shape (columns, columns).
    h_delta: Cross-correlation matrix from compute_qep_stats, shape (columns,
      columns).
    correction_factor: Weight correction factor. 0.0 = no correction, 1.0 = full
      correction. Default 0.5 per QEP paper recommendations.
    dampening_factor: Dampening factor for Hessian inversion as a fraction of
      the average diagonal. Defaults to 0.01.

  Returns:
    The corrected weight matrix, same shape as W.
  """
  columns = h.shape[0]
  # Ensure the Hessian matrix from quantized inputs is square.
  assert h.shape == (columns, columns)
  # Ensure the cross-correlation matrix matches the Hessian shape.
  assert h_delta.shape == (columns, columns)
  # Ensure the weight matrix's input dimension matches the Hessian size.
  assert w.shape[1] == columns

  # Handle dead columns (zero diagonal in hessian) by adding 1 to the diagonal,
  # which basically makes the corresponding row and column an identity matrix.
  h_diag = jnp.diag(h)
  dead = h_diag == 0
  h = jnp.where(dead & jnp.eye(columns, dtype=bool), 1.0, h)
  # Zero out the corresponding weights to prevent the dummy 1.0 from the Hessian
  # diagonal from propagating pseudo-gradients to other active weights.
  w = jnp.where(dead, 0.0, w)

  # Dampen the Hessian (using higher dampening than GPTQ).
  damp = dampening_factor * jnp.mean(h_diag)
  diagonal = jnp.arange(columns)
  h = h.at[diagonal, diagonal].add(damp)

  # Compute H_inv via Cholesky factorization.
  h = jnp.linalg.cholesky(h)
  h_inverse = gptq_core.cholesky_inverse(h)

  # Apply weight correction.
  w = w + correction_factor * (w @ h_delta @ h_inverse)
  return w
