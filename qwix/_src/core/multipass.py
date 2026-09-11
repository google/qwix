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
"""Multi-pass emulation for microscaled floating-point and integer formats.

This module provides multi-pass residual quantization emulation:
- 2-pass residual decomposition:
    Pass 0: X_0 = quantize(X),   R_1 = X - dequantize(X_0)
    Pass 1: X_1 = quantize(R_1)
- Multi-pass matrix multiplication modes:
    1. 'triangular' (3 passes): A_0 B_0 + A_0 B_1 + A_1 B_0
       Drops second-order cross-residual A_1 B_1 (scaled by ~2^-14).
    2. 'full_cross' (4 passes): A_0 B_0 + A_0 B_1 + A_1 B_0 + A_1 B_1
       Evaluates all 4 cross-products.
    3. 'lhs_high_precision' (2 passes): A_0 B_0 + A_1 B_0
       LHS uses 2 residual passes, RHS uses 1 pass.
    4. 'rhs_high_precision' (2 passes): A_0 B_0 + A_0 B_1
       LHS uses 1 pass, RHS uses 2 residual passes.
- Asymmetric MX-INT dot general (mxint4 x mxint8) with block-scale accumulation.
"""

from typing import Literal, TypeAlias
import jax
from jax import lax
import jax.numpy as jnp
from qwix._src.core import dot_general
from qwix._src.core import qarray

MultiPassMode: TypeAlias = Literal[
    'triangular',
    'full_cross',
    'lhs_high_precision',
    'rhs_high_precision',
]


def residual_decompose(
    x: jax.Array,
    how: qarray.HowToQuantize,
    n_passes: int = 2,
) -> tuple[qarray.QArray, ...]:
  """Decomposes tensor x into residual quantized passes.

  For each pass p:
    q_p = quantize(r_p, how)
    x_p = dequantize(q_p)
    r_{p+1} = r_p - x_p

  Args:
    x: Input array.
    how: HowToQuantize configuration specifying format, tile size, scaling, etc.
    n_passes: Number of residual passes (default 2).

  Returns:
    Tuple of quantized QArray objects (q_0, q_1, ...).
  """
  passes = []
  current = x
  for _ in range(n_passes):
    q = qarray.quantize(current, how)
    passes.append(q)
    deq = qarray.dequantize(q)
    current = current - deq
  return tuple(passes)


def multipass_dot(
    lhs: jax.Array,
    rhs: jax.Array,
    dimension_numbers: jax.lax.DotDimensionNumbers = (((1,), (0,)), ((), ())),
    mode: MultiPassMode = 'triangular',
    *,
    lhs_how: qarray.HowToQuantize | None = None,
    rhs_how: qarray.HowToQuantize | None = None,
    lhs_qtype: str = 'mxfp8_16',
    rhs_qtype: str = 'mxfp8_16',
    tile_size: int = 16,
    lhs_scale_method: str = 'default',
    rhs_scale_method: str = 'default',
    preferred_element_type: jax.typing.DTypeLike | None = None,
) -> jax.Array:
  """Executes multi-pass emulated matrix multiplication.

  Args:
    lhs: Left-hand side tensor.
    rhs: Right-hand side tensor.
    dimension_numbers: Standard JAX dot_general dimension specification.
    mode: Multi-pass mode: - 'triangular': Computes 3 GEMMs: A_0 B_0 + A_0 B_1 +
      A_1 B_0. - 'full_cross': Computes 4 GEMMs: A_0 B_0 + A_0 B_1 + A_1 B_0 +
      A_1 B_1. - 'lhs_high_precision': Computes 2 GEMMs: A_0 B_0 + A_1 B_0 (2
      passes on LHS, 1 on RHS). - 'rhs_high_precision': Computes 2 GEMMs: A_0
      B_0 + A_0 B_1 (1 pass on LHS, 2 on RHS).
    lhs_how: Explicit HowToQuantize for LHS (overrides lhs_qtype, tile_size,
      etc.).
    rhs_how: Explicit HowToQuantize for RHS (overrides rhs_qtype, tile_size,
      etc.).
    lhs_qtype: QType for LHS if lhs_how is None (default 'mxfp8_16').
    rhs_qtype: QType for RHS if rhs_how is None (default 'mxfp8_16').
    tile_size: Microscaling tile size (default 16 for mxfp8_16).
    lhs_scale_method: Scale method for LHS ('default', 'oas', 'ceil').
    rhs_scale_method: Scale method for RHS ('default', 'oas', 'ceil').
    preferred_element_type: Output/accumulator dtype.

  Returns:
    The resulting matrix product array.
  """
  if lhs_how is None:
    lhs_how = dot_general.get_how_to_quantize(
        dimension_numbers=dimension_numbers,
        ndims=(lhs.ndim, rhs.ndim),
        for_lhs=True,
        qtype=lhs_qtype,
        tile_size=tile_size,
        scale_method=lhs_scale_method,
    )
  if rhs_how is None:
    rhs_how = dot_general.get_how_to_quantize(
        dimension_numbers=dimension_numbers,
        ndims=(lhs.ndim, rhs.ndim),
        for_lhs=False,
        qtype=rhs_qtype,
        tile_size=tile_size,
        scale_method=rhs_scale_method,
    )

  if mode == 'lhs_high_precision':
    a_passes = residual_decompose(lhs, lhs_how, n_passes=2)
    a0, a1 = a_passes[0], a_passes[1]
    b0 = qarray.quantize(rhs, rhs_how)
    c0 = dot_general.dot_general(
        a0, b0, dimension_numbers, preferred_element_type=preferred_element_type
    )
    c1 = dot_general.dot_general(
        a1, b0, dimension_numbers, preferred_element_type=preferred_element_type
    )
    return c0 + c1

  elif mode == 'rhs_high_precision':
    a0 = qarray.quantize(lhs, lhs_how)
    b_passes = residual_decompose(rhs, rhs_how, n_passes=2)
    b0, b1 = b_passes[0], b_passes[1]
    c0 = dot_general.dot_general(
        a0, b0, dimension_numbers, preferred_element_type=preferred_element_type
    )
    c1 = dot_general.dot_general(
        a0, b1, dimension_numbers, preferred_element_type=preferred_element_type
    )
    return c0 + c1

  elif mode in ('triangular', 'full_cross'):
    a_passes = residual_decompose(lhs, lhs_how, n_passes=2)
    a0, a1 = a_passes[0], a_passes[1]
    b_passes = residual_decompose(rhs, rhs_how, n_passes=2)
    b0, b1 = b_passes[0], b_passes[1]
    c00 = dot_general.dot_general(
        a0, b0, dimension_numbers, preferred_element_type=preferred_element_type
    )
    c01 = dot_general.dot_general(
        a0, b1, dimension_numbers, preferred_element_type=preferred_element_type
    )
    c10 = dot_general.dot_general(
        a1, b0, dimension_numbers, preferred_element_type=preferred_element_type
    )
    res = c00 + c01 + c10
    if mode == 'full_cross':
      c11 = dot_general.dot_general(
          a1,
          b1,
          dimension_numbers,
          preferred_element_type=preferred_element_type,
      )
      res = res + c11
    return res

  else:
    raise ValueError(
        f"Unknown multipass mode: {mode!r}. Expected one of: 'triangular',"
        " 'full_cross', 'lhs_high_precision', 'rhs_high_precision'."
    )


multipass_mxfp8_dot = multipass_dot


def asymmetric_mxint_dot(
    lhs: jax.Array | qarray.QArray,
    rhs: jax.Array | qarray.QArray,
    dimension_numbers: jax.lax.DotDimensionNumbers = (((1,), (0,)), ((), ())),
    *,
    lhs_qtype: str = 'mxint4',
    rhs_qtype: str = 'mxint8',
    tile_size: int = 32,
    lhs_scale_method: str = 'default',
    rhs_scale_method: str = 'default',
    preferred_element_type: jax.typing.DTypeLike | None = None,
    accumulate_blocks: bool = False,
) -> jax.Array:
  """Asymmetric microscaled integer dot (e.g. mxint4 x mxint8).

  Computes matrix multiplication between microscaled integer operands with block
  size 32 and power-of-2 scaling (supporting OAS for mxint4 and ceil for
  mxint8).

  Args:
    lhs: Left-hand side operand (jax.Array or QArray).
    rhs: Right-hand side operand (jax.Array or QArray).
    dimension_numbers: Standard JAX dot_general dimension specification.
    lhs_qtype: QType for LHS if not already QArray (default 'mxint4').
    rhs_qtype: QType for RHS if not already QArray (default 'mxint8').
    tile_size: Block size along contracting axis (default 32).
    lhs_scale_method: Scaling method for LHS ('default', 'oas', 'ceil').
    rhs_scale_method: Scaling method for RHS ('default', 'oas', 'ceil').
    preferred_element_type: Output precision (e.g. jnp.float32, jnp.bfloat16).
    accumulate_blocks: If True, computes integer dot products per block in int32
      scaled by block scales and accumulates across blocks. If False,
      dequantizes operands and executes full tensor GEMM.

  Returns:
    The resulting matrix multiplication array.
  """
  if not isinstance(lhs, qarray.QArray):
    lhs_how = dot_general.get_how_to_quantize(
        dimension_numbers=dimension_numbers,
        ndims=(lhs.ndim, rhs.ndim),
        for_lhs=True,
        qtype=lhs_qtype,
        tile_size=tile_size,
        scale_method=lhs_scale_method,
    )
    q_lhs = qarray.quantize(lhs, lhs_how)
  else:
    q_lhs = lhs

  if not isinstance(rhs, qarray.QArray):
    rhs_how = dot_general.get_how_to_quantize(
        dimension_numbers=dimension_numbers,
        ndims=(lhs.ndim, rhs.ndim),
        for_lhs=False,
        qtype=rhs_qtype,
        tile_size=tile_size,
        scale_method=rhs_scale_method,
    )
    q_rhs = qarray.quantize(rhs, rhs_how)
  else:
    q_rhs = rhs

  if accumulate_blocks:
    # Upcast int4 and int8 values to int32 for exact integer dot_general per
    # tile.
    q_lhs_int32 = qarray.QArray(
        qvalue=q_lhs.qvalue.astype(jnp.int32),
        scale=q_lhs.scale,
        zero_point=q_lhs.zero_point,
        qtype=jnp.int32,
    )
    q_rhs_int32 = qarray.QArray(
        qvalue=q_rhs.qvalue.astype(jnp.int32),
        scale=q_rhs.scale,
        zero_point=q_rhs.zero_point,
        qtype=jnp.int32,
    )
    # Call _fast_dot_general on int32 QArrays to compute block-level integer
    # GEMM.
    return dot_general._fast_dot_general(  # pylint: disable=protected-access
        q_lhs_int32,
        q_rhs_int32,
        dimension_numbers=dimension_numbers,
        preferred_element_type=preferred_element_type,
    )

  # Dequantized fallback (matches block-wise integer dot scaled by scales)
  lhs_deq = qarray.dequantize(q_lhs)
  rhs_deq = qarray.dequantize(q_rhs)
  return lax.dot_general(
      lhs_deq,
      rhs_deq,
      dimension_numbers=dimension_numbers,
      preferred_element_type=preferred_element_type,
  )


def compute_snr_db(true_val: jax.Array, approx_val: jax.Array) -> float:
  """Computes Signal-to-Noise Ratio (SNR) in decibels (dB)."""
  true_f = true_val.astype(jnp.float32)
  approx_f = approx_val.astype(jnp.float32)
  noise = true_f - approx_f
  signal_power = jnp.mean(jnp.square(true_f))
  noise_power = jnp.mean(jnp.square(noise))
  snr = 10.0 * jnp.log10(
      jnp.maximum(signal_power / jnp.maximum(noise_power, 1e-12), 1e-12)
  )
  return float(snr)


def compute_relative_error(true_val: jax.Array, approx_val: jax.Array) -> float:
  """Computes relative L2 error ||true - approx|| / ||true||."""
  true_f = true_val.astype(jnp.float32)
  approx_f = approx_val.astype(jnp.float32)
  norm_diff = jnp.linalg.norm(true_f - approx_f)
  norm_true = jnp.linalg.norm(true_f)
  return float(norm_diff / jnp.maximum(norm_true, 1e-12))
