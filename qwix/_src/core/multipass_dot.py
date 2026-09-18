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
"""Multi-pass for fp8 formats.

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

All decomposed passes are evaluated on quantized operands directly through the
hardware FP8 path (via `_fast_dot_general` with hardware FP8 accumulation) or
via `jax.nn.scaled_matmul` on supported GPUs.

Scaling factor design choice for residual passes:
While an alternative formulation could independently compute fresh scaling
factors for each residual pass (via block-level max-reduction searches over
R_1), we instead derive residual scales by shifting the initial pass's scale
factor by a fixed power-of-2 exponent offset (e.g. 2^-4 for FP8 E4M3).
This design choice is made because:

1. Hardware efficiency: It eliminates the expensive second reduction
   tree across elements, replacing dynamic exponent search with a single ALU
   integer shift (E_1 = E_0 - shift_bits).
2. Memory bandwidth and storage: The residual scale is implicit rather than a
   separate scale tensor that must be stored and loaded from memory, cutting
   scale metadata traffic.
3. Fixed accumulator alignment: Constant power-of-2 scale offsets allow matrix
   units to align cross-term products with simple arithmetic bit-shifts prior
   to register accumulation, rather than performing variable floating-
   point scale multiplications.
4. Numerical closeness: Because the maximum rounding residual of round-to-
   nearest is bounded by the top bin step size, shifting by these exact bits
   yields similar SQNR to independent scale choices.
"""

from collections.abc import Mapping
import functools
from typing import Any, Literal, TypeAlias
import jax
import jax.numpy as jnp
from qwix._src.core import dot_general as dg
from qwix._src.core import qarray

MultiPassMode: TypeAlias = Literal[
    'triangular',
    'full_cross',
    'lhs_high_precision',
    'rhs_high_precision',
]


def get_how_to_quantize(
    *,
    dimension_numbers: jax.lax.DotDimensionNumbers,
    ndims: tuple[int, int],
    for_lhs: bool,
    tile_size: Mapping[int, int | float] | int | float | None,
    **kwargs: Any,
) -> qarray.HowToQuantize:
  """Get how to quantize from dimension_numbers and remaining_dims."""
  if for_lhs:
    ndim = ndims[0]
    contracting_axes = dimension_numbers[0][0]
  else:
    ndim = ndims[1]
    contracting_axes = dimension_numbers[0][1]

  if isinstance(tile_size, Mapping):
    tiled_axes = tile_size
  else:
    tiled_axes = {}
    if tile_size:
      tiled_axes = {contracting_axes[-1]: tile_size}

  channelwise_axes = sorted(
      set(range(ndim)) - set(contracting_axes) - set(tiled_axes.keys())
  )

  return qarray.HowToQuantize(
      channelwise_axes=channelwise_axes,
      tiled_axes=tiled_axes,
      **kwargs,
  )


def get_residual_scale_shift_bits(qtype: jax.typing.DTypeLike) -> int | None:
  """Returns the power-of-2 scale shift bits for residual quantization.

  For floating-point and microscaled formats, the residual quantization scale
  can be derived directly from the preceding pass's scale by shifting its
  exponent, eliminating redundant block-level reduction passes:
  - fp8 / float8_e4m3fn / mxfp8 / mxfp8_16 (E4M3): shift by 4 bits (2^-4 =
  1/16).
  - mxfp4 / nvfp4 / float4_e2m1fn (E2M1): shift by 2 bits (2^-2 = 1/4).
  - float8_e5m2 (E5M2): shift by 2 bits (2^-2 = 1/4).

  For integer formats (e.g. int4, int8, mxint8), returns None to indicate
  exact algebraic decomposition or independent calibration.

  Args:
    qtype: Quantization format or dtype string.

  Returns:
    The integer exponent shift in bits, or None.
  """
  # Normalize synthetic string aliases to standard JAX dtypes.
  match qtype:
    case 'mxfp8' | 'mxfp8_16' | 'float8_e4m3' | 'fp8':
      qtype = jnp.float8_e4m3fn
    case 'mxfp4' | 'nvfp4':
      qtype = jnp.float4_e2m1fn

  try:
    dt = jnp.dtype(qtype)
  except (TypeError, ValueError):
    return None

  if dt == jnp.float8_e4m3fn:
    return 4
  if dt in (jnp.float4_e2m1fn, jnp.float8_e5m2):
    return 2
  return None


def residual_decompose(
    x: qarray.MaybeQArray,
    how: qarray.HowToQuantize,
    n_passes: int = 2,
) -> tuple[qarray.QArray, ...]:
  """Decomposes tensor x into residual quantized passes.

  For floating-point formats (e.g., mxfp8, mxfp4, float8_e4m3fn), residual
  passes reuse the initial pass's scale factor shifted by the format's
  bit-precision (e.g. 2^-4 for FP8, 2^-2 for FP4), eliminating redundant
  block-level reduction passes. For integer formats, passes are calibrated
  independently.

  If x is already a QArray, it is used as the first pass, and subsequent passes
  are zero.

  Args:
    x: Input array or QArray.
    how: HowToQuantize configuration specifying format, tile size, scaling, etc.
    n_passes: Number of residual passes (default 2).

  Returns:
    Tuple of quantized QArray objects (q_0, q_1, ...).
  """
  if isinstance(x, qarray.QArray):
    raise ValueError(
        'Input to residual_decompose must strictly be unquantized jax.Array,'
        f' but got {type(x)}.'
    )

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


def multipass_dot_general(
    lhs: jax.Array,
    rhs: jax.Array,
    dimension_numbers: jax.lax.DotDimensionNumbers = (((1,), (0,)), ((), ())),
    precision: jax.lax.PrecisionLike = None,
    preferred_element_type: jax.typing.DTypeLike | None = None,
    **kwargs: Any,
) -> jax.Array:
  """Executes multi-pass emulated matrix multiplication.

  Inputs must strictly be unquantized jax.Array to allow residual calculation.

  Args:
    lhs: Left-hand side unquantized array.
    rhs: Right-hand side unquantized array.
    dimension_numbers: Standard JAX dot_general dimension specification.
    precision: The precision for dot_general.
    preferred_element_type: Output/accumulator dtype.
    **kwargs: Additional arguments forwarded to dot_general, including optional
      'multipass_mode', 'lhs_how', 'rhs_how', 'lhs_qtype', 'rhs_qtype',
      'tile_size'.

  Returns:
    The resulting matrix product array.
  """
  if isinstance(lhs, qarray.QArray) or isinstance(rhs, qarray.QArray):
    raise ValueError(
        'Inputs to multipass_dot_general must strictly be unquantized'
        f' jax.Array, but got lhs={type(lhs)}, rhs={type(rhs)}.'
    )

  kwargs = dict(kwargs)
  mode: MultiPassMode = (
      kwargs.pop('mode', None)
      or kwargs.pop('multipass_mode', None)
      or kwargs.pop('multipass_mode', None)
      or 'triangular'
  )
  kwargs.pop('multipass_mode', None)
  kwargs.pop('multipass_mode', None)
  lhs_how = kwargs.pop('lhs_how', None)
  rhs_how = kwargs.pop('rhs_how', None)
  lhs_qtype = kwargs.pop('lhs_qtype', jnp.float8_e4m3fn)
  rhs_qtype = kwargs.pop('rhs_qtype', jnp.float8_e4m3fn)
  tile_size = kwargs.pop('tile_size', None)
  kwargs.pop('dot_general_fn', None)

  if lhs_how is None:
    lhs_how = get_how_to_quantize(
        dimension_numbers=dimension_numbers,
        ndims=(lhs.ndim, rhs.ndim),
        for_lhs=True,
        qtype=lhs_qtype,
        tile_size=tile_size,
    )
  if rhs_how is None:
    rhs_how = get_how_to_quantize(
        dimension_numbers=dimension_numbers,
        ndims=(lhs.ndim, rhs.ndim),
        for_lhs=False,
        qtype=rhs_qtype,
        tile_size=tile_size,
    )

  def _dot(a: qarray.QArray, b: qarray.QArray) -> jax.Array:
    return dg.dot_general(
        a,
        b,
        dimension_numbers=dimension_numbers,
        precision=precision,
        preferred_element_type=preferred_element_type,
        **kwargs,
    )

  if mode == 'lhs_high_precision':
    a_passes = residual_decompose(lhs, lhs_how, n_passes=2)
    a0, a1 = a_passes[0], a_passes[1]
    b0 = qarray.quantize(rhs, rhs_how)
    c0 = _dot(a0, b0)
    c1 = _dot(a1, b0)
    return c0 + c1

  elif mode == 'rhs_high_precision':
    a0 = qarray.quantize(lhs, lhs_how)
    b_passes = residual_decompose(rhs, rhs_how, n_passes=2)
    b0, b1 = b_passes[0], b_passes[1]
    c0 = _dot(a0, b0)
    c1 = _dot(a0, b1)
    return c0 + c1

  elif mode in ('triangular', 'full_cross'):
    a_passes = residual_decompose(lhs, lhs_how, n_passes=2)
    a0, a1 = a_passes[0], a_passes[1]
    b_passes = residual_decompose(rhs, rhs_how, n_passes=2)
    b0, b1 = b_passes[0], b_passes[1]
    c00 = _dot(a0, b0)
    c01 = _dot(a0, b1)
    c10 = _dot(a1, b0)
    res = c00 + c01 + c10
    if mode == 'full_cross':
      c11 = _dot(a1, b1)
      res = res + c11
    return res

  elif mode == 'int8_triangular':
    compute_dtype = kwargs.pop('compute_dtype', jnp.float8_e4m3fn)
    tile_size = kwargs.pop('tile_size', 256)
    return int8_multipass_dot_general(
        lhs,
        rhs,
        dimension_numbers=dimension_numbers,
        compute_dtype=compute_dtype,
        method='triangular',
        tile_size=tile_size,
        preferred_element_type=preferred_element_type,
        **kwargs,
    )
  elif mode in ('int8_int4', 'int4_int8'):
    int4_op = 'rhs' if mode == 'int8_int4' else 'lhs'
    compute_dtype = kwargs.pop('compute_dtype', jnp.float8_e4m3fn)
    tile_size = kwargs.pop('tile_size', 256)
    return asymmetric_int_dot_general(
        lhs,
        rhs,
        dimension_numbers=dimension_numbers,
        compute_dtype=compute_dtype,
        int4_operand=int4_op,
        tile_size=tile_size,
        preferred_element_type=preferred_element_type,
        **kwargs,
    )
  else:
    raise ValueError(
        f"Unknown multipass mode: {mode!r}. Expected one of: 'triangular',"
        " 'full_cross', 'lhs_high_precision', 'rhs_high_precision',"
        " 'int8_karatsuba', 'int8_naive', 'int8_triangular', 'int8_int4',"
        " 'int4_int8'."
    )


def karatsuba_signed_int8_dot_general(
    a: jax.Array,
    b: jax.Array,
    dimension_numbers: jax.lax.DotDimensionNumbers = (((1,), (0,)), ((), ())),
    *,
    compute_dtype: jax.typing.DTypeLike,
    preferred_element_type: jax.typing.DTypeLike | None = None,
) -> jax.Array:
  """Algebraic Karatsuba signed int8 GEMM using strictly 3 native int4 passes."""
  a_int = a.astype(jnp.int32)
  b_int = b.astype(jnp.int32)
  (lhs_ca, rhs_ca), (lhs_ba, rhs_ba) = dimension_numbers

  a_h = a_int >> 4
  a_l = ((a_int.astype(jnp.uint8) & 0x0F).astype(jnp.int32)) - 8

  b_h = b_int >> 4
  b_l = ((b_int.astype(jnp.uint8) & 0x0F).astype(jnp.int32)) - 8

  s_a = a_h + a_l
  s_b = b_h + b_l

  a_half = s_a >> 1
  r_a = s_a & 1

  b_half = s_b >> 1
  r_b = s_b & 1

  def _dot(x: jax.Array, y: jax.Array) -> jax.Array:
    return jax.lax.dot_general(
        x.astype(compute_dtype),
        y.astype(compute_dtype),
        dimension_numbers=dimension_numbers,
        preferred_element_type=jnp.float32,
    )

  z2 = _dot(a_h, b_h)
  z0 = _dot(a_l, b_l)
  z1_core = _dot(a_half, b_half)

  z1 = (
      (z1_core.astype(jnp.int32) << 2)
      + (_dot(a_half, r_b).astype(jnp.int32) << 1)
      + (_dot(r_a, b_half).astype(jnp.int32) << 1)
      + _dot(r_a, r_b).astype(jnp.int32)
  )

  cross_term = z1 - z2.astype(jnp.int32) - z0.astype(jnp.int32)
  xy_product = (
      (z2.astype(jnp.int32) << 8) + (cross_term << 4) + z0.astype(jnp.int32)
  )

  k_size = 1
  for d in lhs_ca:
    k_size *= a.shape[d]

  sum_a = jnp.sum(a_int, axis=lhs_ca)
  sum_b = jnp.sum(b_int, axis=rhs_ca)

  lhs_rem_ndims = a.ndim - len(lhs_ca) - len(lhs_ba)
  rhs_rem_ndims = b.ndim - len(rhs_ca) - len(rhs_ba)

  for _ in range(rhs_rem_ndims):
    sum_a = jnp.expand_dims(sum_a, axis=-1)
  insert_pos = len(lhs_ba)
  for _ in range(lhs_rem_ndims):
    sum_b = jnp.expand_dims(sum_b, axis=insert_pos)

  res = xy_product + (sum_a * 8) + (sum_b * 8) - (64 * k_size)
  if preferred_element_type is not None:
    return res.astype(preferred_element_type)
  return res


def naive_signed_int8_dot_general(
    a: jax.Array,
    b: jax.Array,
    dimension_numbers: jax.lax.DotDimensionNumbers = (((1,), (0,)), ((), ())),
    *,
    compute_dtype: jax.typing.DTypeLike,
    preferred_element_type: jax.typing.DTypeLike | None = None,
) -> jax.Array:
  """Naive signed int8 GEMM using 4 uncoupled native signed int4 passes."""
  a_int = a.astype(jnp.int32)
  b_int = b.astype(jnp.int32)
  (lhs_ca, rhs_ca), (lhs_ba, rhs_ba) = dimension_numbers

  a_h = a_int >> 4
  a_l = ((a_int.astype(jnp.uint8) & 0x0F).astype(jnp.int32)) - 8

  b_h = b_int >> 4
  b_l = ((b_int.astype(jnp.uint8) & 0x0F).astype(jnp.int32)) - 8

  def _dot(x: jax.Array, y: jax.Array) -> jax.Array:
    return jax.lax.dot_general(
        x.astype(compute_dtype),
        y.astype(compute_dtype),
        dimension_numbers=dimension_numbers,
        preferred_element_type=jnp.float32,
    )

  p11 = _dot(a_h, b_h)
  p10 = _dot(a_h, b_l)
  p01 = _dot(a_l, b_h)
  p00 = _dot(a_l, b_l)

  xy_product = (
      (p11.astype(jnp.int32) << 8)
      + ((p10.astype(jnp.int32) + p01.astype(jnp.int32)) << 4)
      + p00.astype(jnp.int32)
  )

  k_size = 1
  for d in lhs_ca:
    k_size *= a.shape[d]

  sum_a = jnp.sum(a_int, axis=lhs_ca)
  sum_b = jnp.sum(b_int, axis=rhs_ca)

  lhs_rem_ndims = a.ndim - len(lhs_ca) - len(lhs_ba)
  rhs_rem_ndims = b.ndim - len(rhs_ca) - len(rhs_ba)

  for _ in range(rhs_rem_ndims):
    sum_a = jnp.expand_dims(sum_a, axis=-1)
  insert_pos = len(lhs_ba)
  for _ in range(lhs_rem_ndims):
    sum_b = jnp.expand_dims(sum_b, axis=insert_pos)

  res = xy_product + (sum_a * 8) + (sum_b * 8) - (64 * k_size)
  if preferred_element_type is not None:
    return res.astype(preferred_element_type)
  return res


def int8_multipass_dot_general(
    lhs: jax.Array | qarray.QArray,
    rhs: jax.Array | qarray.QArray,
    dimension_numbers: jax.lax.DotDimensionNumbers = (((1,), (0,)), ((), ())),
    *,
    compute_dtype: jax.typing.DTypeLike = jnp.float8_e4m3fn,
    method: Literal[
        'karatsuba', 'naive', 'triangular', 'asymmetric'
    ] = 'karatsuba',
    int4_operand: Literal['lhs', 'rhs'] = 'rhs',
    tile_size: int | None = 256,
    preferred_element_type: jax.typing.DTypeLike | None = None,
) -> jax.Array:
  """Executes scaled INT8 (channelwise, subchannel, or mxint8) GEMM via INT4 passes."""
  if not isinstance(lhs, qarray.QArray):
    lhs_qtype = 'mxint8' if tile_size == 32 else jnp.int8
    lhs_how = get_how_to_quantize(
        dimension_numbers=dimension_numbers,
        ndims=(lhs.ndim, rhs.ndim),
        for_lhs=True,
        tile_size=tile_size,
        qtype=lhs_qtype,
    )
    q_lhs = qarray.quantize(lhs, lhs_how)
  else:
    q_lhs = lhs

  if not isinstance(rhs, qarray.QArray):
    rhs_qtype = 'mxint8' if tile_size == 32 else jnp.int8
    rhs_how = get_how_to_quantize(
        dimension_numbers=dimension_numbers,
        ndims=(lhs.ndim, rhs.ndim),
        for_lhs=False,
        tile_size=tile_size,
        qtype=rhs_qtype,
    )
    q_rhs = qarray.quantize(rhs, rhs_how)
  else:
    q_rhs = rhs

  if method == 'karatsuba':
    dot_fn = karatsuba_signed_int8_dot_general
  elif method == 'naive':
    dot_fn = naive_signed_int8_dot_general
  elif method == 'triangular':
    dot_fn = triangular_signed_int8_dot_general
  elif method == 'asymmetric':
    dot_fn = functools.partial(
        asymmetric_signed_int4_int8_dot_general, int4_operand=int4_operand
    )
  else:
    raise ValueError(f'Unknown method: {method}')

  (lhs_ca, rhs_ca), (lhs_ba, rhs_ba) = dimension_numbers
  if (
      len(lhs_ca) == 1
      and len(rhs_ca) == 1
      and len(lhs_ba) == 0
      and len(rhs_ba) == 0
      and lhs.ndim == 2
      and rhs.ndim == 2
  ):
    m, k = q_lhs.qvalue.shape
    k2, n = q_rhs.qvalue.shape
    eff_tile = k if tile_size is None else tile_size
    if k % eff_tile == 0 and k == k2:
      n_blocks = k // eff_tile
      a_blocks = q_lhs.qvalue.reshape(m, n_blocks, eff_tile)
      b_blocks = q_rhs.qvalue.reshape(n_blocks, eff_tile, n)
      sa_blocks = q_lhs.scale.reshape(m, n_blocks).T[:, :, None]
      sb_blocks = q_rhs.scale.reshape(n_blocks, n)[:, None, :]

      def block_gemm(a_b, b_b):
        return dot_fn(
            a_b,
            b_b,
            (((1,), (0,)), ((), ())),
            compute_dtype=compute_dtype,
        )

      block_prods = jax.vmap(block_gemm, in_axes=(1, 0), out_axes=0)(
          a_blocks, b_blocks
      )
      scale_prods = sa_blocks * sb_blocks
      accum = jnp.sum(
          block_prods.astype(jnp.float32) * scale_prods.astype(jnp.float32),
          axis=0,
      )
      if preferred_element_type is not None:
        return accum.astype(preferred_element_type)
      return accum

  lhs_deq = qarray.dequantize(q_lhs)
  rhs_deq = qarray.dequantize(q_rhs)
  return jax.lax.dot_general(
      lhs_deq,
      rhs_deq,
      dimension_numbers=dimension_numbers,
      preferred_element_type=preferred_element_type,
  )


def triangular_signed_int8_dot_general(
    a: jax.Array,
    b: jax.Array,
    dimension_numbers: jax.lax.DotDimensionNumbers = (((1,), (0,)), ((), ())),
    *,
    compute_dtype: jax.typing.DTypeLike = jnp.float8_e4m3fn,
    preferred_element_type: jax.typing.DTypeLike | None = None,
) -> jax.Array:
  """Triangular signed int8 GEMM using 3 uncoupled native signed int4 passes."""
  a_int = a.astype(jnp.int32)
  b_int = b.astype(jnp.int32)
  (lhs_ca, rhs_ca), (lhs_ba, rhs_ba) = dimension_numbers

  a_h = a_int >> 4
  a_l = ((a_int.astype(jnp.uint8) & 0x0F).astype(jnp.int32)) - 8

  b_h = b_int >> 4
  b_l = ((b_int.astype(jnp.uint8) & 0x0F).astype(jnp.int32)) - 8

  def _dot(x: jax.Array, y: jax.Array) -> jax.Array:
    return jax.lax.dot_general(
        x.astype(compute_dtype),
        y.astype(compute_dtype),
        dimension_numbers=dimension_numbers,
        preferred_element_type=jnp.float32,
    )

  p11 = _dot(a_h, b_h)
  p10 = _dot(a_h, b_l)
  p01 = _dot(a_l, b_h)

  xy_product = (p11.astype(jnp.int32) << 8) + (
      (p10.astype(jnp.int32) + p01.astype(jnp.int32)) << 4
  )

  k_size = 1
  for d in lhs_ca:
    k_size *= a.shape[d]

  sum_a = jnp.sum(a_int, axis=lhs_ca)
  sum_b = jnp.sum(b_int, axis=rhs_ca)

  lhs_rem_ndims = a.ndim - len(lhs_ca) - len(lhs_ba)
  rhs_rem_ndims = b.ndim - len(rhs_ca) - len(rhs_ba)

  for _ in range(rhs_rem_ndims):
    sum_a = jnp.expand_dims(sum_a, axis=-1)
  insert_pos = len(lhs_ba)
  for _ in range(lhs_rem_ndims):
    sum_b = jnp.expand_dims(sum_b, axis=insert_pos)

  res = xy_product + (sum_a * 8) + (sum_b * 8) - (64 * k_size)
  if preferred_element_type is not None:
    return res.astype(preferred_element_type)
  return res


def asymmetric_signed_int4_int8_dot_general(
    a: jax.Array,
    b: jax.Array,
    dimension_numbers: jax.lax.DotDimensionNumbers = (((1,), (0,)), ((), ())),
    *,
    compute_dtype: jax.typing.DTypeLike = jnp.float8_e4m3fn,
    int4_operand: Literal['lhs', 'rhs'] = 'lhs',
    preferred_element_type: jax.typing.DTypeLike | None = None,
) -> jax.Array:
  """Asymmetric signed INT4 x INT8 matrix multiplication via 2 INT4 passes."""
  a_int = a.astype(jnp.int32)
  b_int = b.astype(jnp.int32)
  (lhs_ca, rhs_ca), (lhs_ba, rhs_ba) = dimension_numbers

  def _dot(x: jax.Array, y: jax.Array) -> jax.Array:
    return jax.lax.dot_general(
        x.astype(compute_dtype),
        y.astype(compute_dtype),
        dimension_numbers=dimension_numbers,
        preferred_element_type=jnp.float32,
    )

  if int4_operand == 'lhs':
    b_h = b_int >> 4
    b_l = ((b_int.astype(jnp.uint8) & 0x0F).astype(jnp.int32)) - 8

    p1 = _dot(a_int, b_h)
    p0 = _dot(a_int, b_l)

    sum_a = jnp.sum(a_int, axis=lhs_ca)
    rhs_rem_ndims = b.ndim - len(rhs_ca) - len(rhs_ba)
    for _ in range(rhs_rem_ndims):
      sum_a = jnp.expand_dims(sum_a, axis=-1)

    res = (p1.astype(jnp.int32) << 4) + p0.astype(jnp.int32) + (sum_a * 8)
  else:
    a_h = a_int >> 4
    a_l = ((a_int.astype(jnp.uint8) & 0x0F).astype(jnp.int32)) - 8

    p1 = _dot(a_h, b_int)
    p0 = _dot(a_l, b_int)

    sum_b = jnp.sum(b_int, axis=rhs_ca)
    lhs_rem_ndims = a.ndim - len(lhs_ca) - len(lhs_ba)
    insert_pos = len(lhs_ba)
    for _ in range(lhs_rem_ndims):
      sum_b = jnp.expand_dims(sum_b, axis=insert_pos)

    res = (p1.astype(jnp.int32) << 4) + p0.astype(jnp.int32) + (sum_b * 8)

  if preferred_element_type is not None:
    return res.astype(preferred_element_type)
  return res


def asymmetric_int_dot_general(
    lhs: jax.Array | qarray.QArray,
    rhs: jax.Array | qarray.QArray,
    dimension_numbers: jax.lax.DotDimensionNumbers = (((1,), (0,)), ((), ())),
    *,
    compute_dtype: jax.typing.DTypeLike = jnp.float8_e4m3fn,
    int4_operand: Literal['lhs', 'rhs'] = 'rhs',
    tile_size: int | None = 256,
    preferred_element_type: jax.typing.DTypeLike | None = None,
    lhs_qtype: Any = None,
    rhs_qtype: Any = None,
    **kwargs: Any,
) -> jax.Array:
  """Evaluates asymmetric integer matrix multiplication."""
  del kwargs, rhs_qtype
  if lhs_qtype is not None:
    int4_operand = 'lhs' if lhs_qtype in ('mxint4', 'int4', jnp.int4) else 'rhs'

  if int4_operand == 'lhs':
    lhs_qtype_actual = 'mxint4' if tile_size == 32 else jnp.int4
    rhs_qtype_actual = 'mxint8' if tile_size == 32 else jnp.int8
  else:
    lhs_qtype_actual = 'mxint8' if tile_size == 32 else jnp.int8
    rhs_qtype_actual = 'mxint4' if tile_size == 32 else jnp.int4

  if not isinstance(lhs, qarray.QArray):
    lhs_how = get_how_to_quantize(
        dimension_numbers=dimension_numbers,
        ndims=(lhs.ndim, rhs.ndim),
        for_lhs=True,
        tile_size=tile_size,
        qtype=lhs_qtype_actual,
    )
    q_lhs = qarray.quantize(lhs, lhs_how)
  else:
    q_lhs = lhs

  if not isinstance(rhs, qarray.QArray):
    rhs_how = get_how_to_quantize(
        dimension_numbers=dimension_numbers,
        ndims=(lhs.ndim, rhs.ndim),
        for_lhs=False,
        tile_size=tile_size,
        qtype=rhs_qtype_actual,
    )
    q_rhs = qarray.quantize(rhs, rhs_how)
  else:
    q_rhs = rhs

  (lhs_ca, rhs_ca), (lhs_ba, rhs_ba) = dimension_numbers
  if (
      len(lhs_ca) == 1
      and len(rhs_ca) == 1
      and len(lhs_ba) == 0
      and len(rhs_ba) == 0
      and lhs.ndim == 2
      and rhs.ndim == 2
  ):
    m, k = q_lhs.qvalue.shape
    k2, n = q_rhs.qvalue.shape
    eff_tile = k if tile_size is None else tile_size
    if k % eff_tile == 0 and k == k2:
      n_blocks = k // eff_tile
      a_blocks = q_lhs.qvalue.reshape(m, n_blocks, eff_tile)
      b_blocks = q_rhs.qvalue.reshape(n_blocks, eff_tile, n)
      sa_blocks = q_lhs.scale.reshape(m, n_blocks).T[:, :, None]
      sb_blocks = q_rhs.scale.reshape(n_blocks, n)[:, None, :]

      def block_gemm(a_b, b_b):
        return asymmetric_signed_int4_int8_dot_general(
            a_b,
            b_b,
            (((1,), (0,)), ((), ())),
            compute_dtype=compute_dtype,
            int4_operand=int4_operand,
        )

      block_prods = jax.vmap(block_gemm, in_axes=(1, 0), out_axes=0)(
          a_blocks, b_blocks
      )
      scale_prods = sa_blocks * sb_blocks
      accum = jnp.sum(
          block_prods.astype(jnp.float32) * scale_prods.astype(jnp.float32),
          axis=0,
      )
      if preferred_element_type is not None:
        return accum.astype(preferred_element_type)
      return accum

  lhs_deq = qarray.dequantize(q_lhs)
  rhs_deq = qarray.dequantize(q_rhs)
  return jax.lax.dot_general(
      lhs_deq,
      rhs_deq,
      dimension_numbers=dimension_numbers,
      preferred_element_type=preferred_element_type,
  )
