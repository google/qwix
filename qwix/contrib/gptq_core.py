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
"""GPTQ algorithm.

This is a JAX implementation of the GPTQ algorithm from:
https://arxiv.org/pdf/2210.17323

The original implementation is in PyTorch and can be found here:
https://github.com/IST-DASLab/gptq/blob/main/gptq.py
"""

# We try to use the same naming as in the PyTorch implementation, thus
# pylint: disable=invalid-name

from collections.abc import Callable

import jax
import jax.numpy as jnp
from qwix._src.core import qarray


def cholesky_inverse(L: jax.Array) -> jax.Array:
  return jax.scipy.linalg.cho_solve((L, True), jnp.eye(L.shape[0]))


def find_params(
    w: jax.Array, how: qarray.HowToQuantize
) -> tuple[jax.Array, jax.Array | None]:
  """Finds the optimal quantization parameters for a given weight tensor.

  Args:
    w: weight matrix of shape (rows, groupsize).
    how: how to quantize this group of weights. tiled_axes should either be
      empty or {1: groupsize}.

  Returns:
    A tuple of scale and optional zero_point, which should have shape (1, 1) or
    (rows, 1) when channelwise quantization is enabled.
  """
  calibration = qarray.calibrate(w, how)
  return qarray.compute_scale_zero_point(calibration, how.qtype)


def quantize(
    w: jax.Array,
    qtype: jax.typing.DTypeLike,
    scale: jax.Array,
    zero_point: jax.Array | None,
) -> tuple[jax.Array, jax.Array]:
  """Quantize w and return the raw quantized and dequantized w.

  Args:
    w: weight matrix of shape (rows, 1).
    qtype: The target quantized data type (e.g., jnp.int8).
    scale: The quantization scale returned by find_params.
    zero_point: The quantization zero point (optional) returned by find_params.

  Returns:
    A tuple of:
      - The raw quantized integer values.
      - The dequantized floating-point values (approximation of original w).
  """
  qw = qarray.quantize_with_scale_zero_point(w, qtype, scale, zero_point)
  return qw.qvalue, qarray.dequantize(qw)


def quantize_weight(
    W: jax.Array,
    H: jax.Array,
    how: qarray.HowToQuantize,
    blocksize: int = 128,
    percdamp: float = 0.01,
) -> tuple[qarray.QArray, jax.Array]:
  """Quantize a weight matrix using GPTQ.

  This function minimizes Loss = ||W @ X - W_q @ X||^2 using the Hessian as
  H = X @ X.T.

  Args:
    W: weight matrix in W @ X with shape (rows, columns), where columns is the
      input/contraction dimension and rows is the output dimension.
    H: Hessian of with shape (columns, columns), usually computed as X @ X.T.
    how: how to quantize W.
    blocksize: the GPTQ algorithm blocksize (should usually not be changed)
    percdamp: the percentage of the average diagonal used for dampening H.

  Returns:
    a tuple of (W_q, Losses), where W_q is the quantized weight matrix as a
    QArray and Losses is the overall quantization losses.
  """
  rows, columns = W.shape
  assert H.shape == (columns, columns)

  groupsize = how.tiled_axes.get(1, rows)

  H_diag = jnp.diag(H)
  dead = H_diag == 0
  H = jnp.where(dead & jnp.eye(columns, dtype=bool), 1.0, H)
  W = jnp.where(dead, 0.0, W)

  # Dampen the Hessian.
  damp = percdamp * jnp.mean(H_diag)
  diag = jnp.arange(columns)
  H = H.at[diag, diag].add(damp)

  # Cholesky Inverse for Hessian.
  H = jnp.linalg.cholesky(H)
  H = cholesky_inverse(H)
  H = jnp.linalg.cholesky(H, upper=True)
  Hinv = H

  # Q: the final low-precision integer weights. Will be populated step by step.
  # TODO(dangyi): support synthetic qtype.
  Q = jax.new_ref(jnp.zeros_like(W, dtype=how.qtype))
  # Losses: the quantization error for each element in W. Will be populated step
  # by step.
  Losses = jax.new_ref(jnp.zeros_like(W))

  # scales, zero_points: quantization parameters for each group.
  scales, zero_points = [], []

  # The GPTQ algorithm processes the weight matrix `W` in blocks of columns.
  # In each block, weights are quantized column by column, and an error
  # compensation is applied to the remaining unquantized columns in the block.
  for i1 in range(0, columns, blocksize):
    i2 = min(i1 + blocksize, columns)
    count = i2 - i1

    # Prepare a block of columns from the entire weight matrix and hessian:
    # W1 and Hinv1.
    W1 = jax.new_ref(W[:, i1:i2])
    Err1 = jax.new_ref(jnp.zeros_like(W1))
    Losses1 = jax.new_ref(jnp.zeros_like(W1))
    Hinv1 = Hinv[i1:i2, i1:i2]

    # Process each column within the current block.
    for i in range(count):
      w = W1[:, i]
      d = Hinv1[i, i]

      # Find the quantization parameters for the current group of weights.
      # This is done once every `groupsize` columns.
      if (i1 + i) % groupsize == 0:
        scale, zero_point = find_params(
            W[:, (i1 + i) : (i1 + i + groupsize)], how
        )
        scales.append(scale.squeeze(1))
        zero_points.append(zero_point and zero_point.squeeze(1))

      # Quantize the current column `w` and get the dequantized value `dq`.
      q, dq = quantize(w, how.qtype, scales[-1], zero_points[-1])
      Q[:, i1 + i] = q
      # Calculate the quantization loss for this column.
      Losses1[:, i] = (w - dq) ** 2 / d**2

      # Update all remaining unquantized columns in the current block to
      # compensate for the quantization error introduced in the current column.
      err1 = (w - dq) / d
      W1[:, i:] -= jnp.outer(err1, Hinv1[i, i:])
      Err1[:, i] = err1

    # Accumulate the losses from the current block to the total Losses.
    Losses[:, i1:i2] = Losses1[...] / 2

    # Update the full weight matrix `W`. The columns in the current block are
    # updated with the error-compensated `W1`. The columns in subsequent blocks
    # are also adjusted based on the errors accumulated in `Err1`.
    W = W.at[:, i1:i2].set(W1[...])
    W = W.at[:, i2:].subtract(Err1[...] @ Hinv[i1:i2, i2:])

  # Stack the scales and zero points along the column axis.
  scale = jnp.stack(scales, axis=1)
  zero_point = None
  if None not in zero_points:
    zero_point = jnp.stack(zero_points, axis=1)
  return qarray.QArray(Q[...], scale, zero_point), Losses[...]


def compute_hessian(X: jax.Array) -> jax.Array:
  """Computes the Hessian of the GPTQ objective function.

  The GPTQ algorithm minimizes the squared error of the layer's output, rather
  than just the weights. This specifically relies on the input data X to
  determine which weights are most critical to maintain accurately.

  Derivation:
    1. Objective: Minimize difference between original output (Y = W @ X)
       and quantized output (Y_q = W_q @ X).
          L(W_q) = || W @ X - W_q @ X ||^2

    2. Define error term ∆W = W - W_q:
          L(W_q) = || ∆W @ X ||^2

    3. Express squared Frobenius norm as a Trace (||A||^2 = Trace(A @ A.T)):
          L(W_q) = Trace((∆W @ X) @ (∆W @ X).T)
               = Trace(∆W @ X @ X.T @ ∆W.T)

    4. Identify the Hessian:
      The loss is a quadratic form with respect to ∆W. The second derivative
      (Hessian) is exactly twice the constant matrix in the middle term.
          Hessian = 2 * (X @ X.T)
      By convention, the constant factor of 2 is ignored during implementation.

  Args:
    X: Input data matrix of shape (in_features, n_samples).

  Returns:
    The Hessian matrix of shape (in_features, in_features).
  """
  return X @ X.T


def normalize_weight(
    x: jax.Array, contraction_axis: int
) -> tuple[jax.Array, Callable[..., jax.Array]]:
  """Normalizes the weight into (ra, ca) format for GPTQ."""
  # Move the contraction axis to the last dimension.
  x = jnp.moveaxis(x, contraction_axis, -1)
  before_shape = x.shape
  # Reshape the weight to (ra, ca).
  x = x.reshape(-1, x.shape[-1])

  def restore_shape(x: qarray.MaybeQArray) -> qarray.MaybeQArray:
    x = x.reshape(before_shape)
    return jax.tree.map(lambda x: jnp.moveaxis(x, -1, contraction_axis), x)

  return x, restore_shape
