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

Scaling factor design choice for residual passes:
While an alternative formulation could independently compute fresh scaling
factors for each residual pass (via block-level max-reduction searches over
R_1),

we instead derive residual scales by shifting the initial pass's scale factor by
a fixed power-of-2 exponent offset (e.g. 2^-4 for FP8 E4M3, 2^-2 for FP4 E2M1).
This design choice is made because:

1. Hardware efficiency: It eliminates the expensive second block-level reduction
   tree across elements, replacing dynamic exponent search with a single ALU
   integer shift (E_1 = E_0 - shift_bits).
2. Memory bandwidth and storage: The residual scale is implicit rather than a
   separate scale tensor that must be stored and loaded from memory, cutting
   scale metadata traffic by 50%.
3. Fixed accumulator alignment: Constant power-of-2 scale offsets allow matrix
   units to align cross-term products with simple arithmetic bit-shifts prior
   to register accumulation, rather than performing variable per-block floating-
   point scale multiplications.
4. Numerical closeness: Because the maximum rounding residual of round-to-
   nearest is bounded by the top bin step size, shifting by these exact bit
   yields virtually identical SQNR to independent scale choices.
"""

from typing import Literal, TypeAlias
import jax
import jax.numpy as jnp
from qwix._src.core import dot_general
from qwix._src.core import qarray

MultiPassMode: TypeAlias = Literal[
    'triangular',
    'full_cross',
    'lhs_high_precision',
    'rhs_high_precision',
]


def get_residual_scale_shift_bits(qtype: jax.typing.DTypeLike) -> int | None:
  """Returns the power-of-2 scale shift bits for residual quantization.

  For microscaled floating-point formats, the residual quantization scale
  can be derived directly from the preceding pass's scale by shifting its
  exponent, eliminating redundant block-level reduction passes:
  - mxfp8 / mxfp8_16 (E4M3): 16 / 448 = 1/28 -> shift by 4 bits (2^-4 = 1/16).
  - mxfp4 / nvfp4 (E2M1): 1 / 6 = 1/6 -> shift by 2 bits (2^-2 = 1/4).
  - float8_e5m2 (E5M2): 8192 / 57344 = 1/7 -> shift by 2 bits (2^-2 = 1/4).

  For integer formats (e.g. int4, int8, mxint8), returns None to indicate
  exact algebraic decomposition or independent calibration.

  Args:
    qtype: Quantization format or dtype string.

  Returns:
    The integer exponent shift in bits, or None.
  """
  match qtype:
    case 'mxfp8' | 'mxfp8_16':
      return 4
    case 'mxfp4' | 'nvfp4':
      return 2
    case _:
      if qtype == jnp.float8_e4m3fn or qtype == 'float8_e4m3fn':
        return 4
      elif qtype == jnp.float4_e2m1fn or qtype == 'float4_e2m1fn':
        return 2
      elif qtype == jnp.float8_e5m2 or qtype == 'float8_e5m2':
        return 2
      return None


def residual_decompose(
    x: jax.Array,
    how: qarray.HowToQuantize,
    n_passes: int = 2,
) -> tuple[qarray.QArray, ...]:
  """Decomposes tensor x into residual quantized passes.

  For floating-point formats (e.g., mxfp8, mxfp4), residual passes reuse the
  initial pass's scale factor shifted by the format's bit-precision (e.g. 2^-4
  for FP8, 2^-2 for FP4), eliminating redundant block-level reduction passes.
  For integer formats, passes are calibrated independently.

  Args:
    x: Input array.
    how: HowToQuantize configuration specifying format, tile size, scaling, etc.
    n_passes: Number of residual passes (default 2).

  Returns:
    Tuple of quantized QArray objects (q_0, q_1, ...).
  """
  passes = []
  current = x
  shift_bits = get_residual_scale_shift_bits(how.qtype)

  # Pass 0
  q0 = qarray.quantize(current, how)
  passes.append(q0)
  deq0 = qarray.dequantize(q0)
  current = current - deq0

  base_scale = q0.scale
  for p in range(1, n_passes):
    if shift_bits is not None:
      scale_p = (base_scale * (2.0 ** (-shift_bits * p))).astype(
          base_scale.dtype
      )
      q = qarray.quantize_with_scale_zero_point(
          current,
          how.qtype,
          scale=scale_p,
          zero_point=q0.zero_point,
          noise_fn=how.noise_fn,
      )
    else:
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
    )
  if rhs_how is None:
    rhs_how = dot_general.get_how_to_quantize(
        dimension_numbers=dimension_numbers,
        ndims=(lhs.ndim, rhs.ndim),
        for_lhs=False,
        qtype=rhs_qtype,
        tile_size=tile_size,
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
