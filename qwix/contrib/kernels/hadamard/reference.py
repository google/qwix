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

"""Hadamard constructions, numerics, and plain-XLA reference.

Implements the Randomized Hadamard Transform (RHT) and row-wise symmetric
quantization:

* Forward RHT (`inverse=False`): `y = ((x * signs) @ H) / sqrt(n)` along `axis`,
  where `H` is an unnormalized `+-1` Hadamard matrix of order `n` (`H @ H.T =
  n * I`) and `signs` is an optional `+-1` vector along `axis`.
* Inverse RHT (`inverse=True`): `z = ((y @ H.T) / sqrt(n)) * signs`, the exact
  orthogonal inverse (`z == x` in exact arithmetic).
* Block-Hadamard (`block_size = b`): applies `I_{k/b} (x) (H_b / sqrt(b))`
  block-diagonally along `axis`, while `hadamard_rotate_quantize` still
  calibrates one symmetric absmax scale across the full rotated axis `k`.

Hadamard orders are powers of two (Sylvester) or `2^k * (q + 1)` for a prime
`q == 3 (mod 4)` (Paley-I), e.g. `768 = 12 * 64`.

Quantization divides in `float32` and rounds once to the input dtype, which for
`bfloat16` is correctly rounded. `qarray.quantize_api` can differ by one step on
TPU, where XLA rounds its `bfloat16` divisions less exactly.
"""

import functools
import math

import jax
import jax.numpy as jnp
import numpy as np
from qwix._src.core import numerics
from qwix._src.core import qarray

QArray = qarray.QArray

SUPPORTED_QTYPES: frozenset[jnp.dtype] = frozenset((
    jnp.dtype(jnp.int8),
    jnp.dtype(jnp.float8_e4m3fn),
    jnp.dtype(jnp.float8_e5m2),
))


def is_supported_qtype(qtype: jax.typing.DTypeLike | None) -> bool:
  """Returns whether `qtype` is supported by `hadamard_rotate_quantize`."""
  if qtype is None:
    return False
  try:
    qdtype = jnp.dtype(qtype)
  except TypeError:
    return False
  return qdtype in SUPPORTED_QTYPES


def normalize_qtype(qtype: jax.typing.DTypeLike) -> jnp.dtype:
  """Normalizes `qtype` to a canonical `jnp.dtype` and validates support."""
  if not is_supported_qtype(qtype):
    raise ValueError(
        f"Unsupported qtype {qtype!r}; expected one of int8, float8_e4m3fn,"
        " float8_e5m2."
    )
  return jnp.dtype(qtype)


def _is_prime(n: int) -> bool:
  """Returns True if `n` is prime."""
  if n < 2:
    return False
  if n % 2 == 0:
    return n == 2
  i = 3
  while i * i <= n:
    if n % i == 0:
      return False
    i += 2
  return True


def _sylvester_hadamard(power: int) -> np.ndarray:
  """Returns the Sylvester Hadamard matrix of order `2**power` (`int8`, +-1).

  Built with NumPy because callers run while tracing, where the `jax.Array`
  constructors in `hadamard_rot` would add device work and host transfers.

  Args:
    power: The base-2 logarithm of the order.
  """
  h = np.ones((1, 1), dtype=np.int8)
  for _ in range(power):
    h = np.block([[h, h], [h, -h]])
  return h


def paley_hadamard(q: int) -> np.ndarray:
  """Returns the Paley-I Hadamard matrix of order `q + 1` (`int8`, +-1).

  Constructs `H = I + S` with `S = [[0, 1^T], [-1, Q]]`, where `Q[i, j] =
  chi(j - i)` is the Jacobsthal matrix of Legendre symbols modulo a prime
  `q == 3 (mod 4)`. Because `-1` is a quadratic non-residue mod `q`, `Q` is
  skew-symmetric (`Q.T = -Q`), giving `H @ H.T = (q + 1) * I`.

  Args:
    q: A prime satisfying `q % 4 == 3`.
  """
  if not (_is_prime(q) and q % 4 == 3):
    raise ValueError(f"Paley-I requires a prime q == 3 (mod 4), got q={q}.")
  chi = -np.ones(q, dtype=np.int8)
  chi[0] = 0
  for a in range(1, q):
    chi[(a * a) % q] = 1
  idx = (np.arange(q)[None, :] - np.arange(q)[:, None]) % q
  h = np.eye(q + 1, dtype=np.int8)
  h[0, 1:] = 1
  h[1:, 0] = -1
  h[1:, 1:] += chi[idx]
  return h


def hadamard_factors(n: int) -> tuple[np.ndarray, np.ndarray]:
  """Returns `(f1, f2)` (`int8`, +-1) whose Kronecker product is Hadamard(`n`).

  For powers of two `n = 2**k`, splits into balanced Sylvester factors
  `(2**(k - k//2), 2**(k//2))`. Otherwise searches `n = (q + 1) * 2**kk` from
  largest `kk` downward for a prime `q == 3 (mod 4)` and returns
  `(paley_hadamard(q), H_{2**kk})`.

  Args:
    n: Positive Hadamard order.

  Raises:
    ValueError: If `n <= 0` or no Sylvester / Paley-I factorization exists.
  """
  if n <= 0:
    raise ValueError(f"n must be positive, got {n}.")
  k = (n & -n).bit_length() - 1
  if n == 1 << k:
    a = k // 2
    return _sylvester_hadamard(k - a), _sylvester_hadamard(a)
  for kk in range(k, -1, -1):
    m = n >> kk
    if _is_prime(m - 1) and (m - 1) % 4 == 3:
      return paley_hadamard(m - 1), _sylvester_hadamard(kk)
  raise ValueError(
      f"No Sylvester / Paley-I Hadamard factorization for n={n}; pad the"
      " contraction dimension or pass a supported block_size."
  )


@functools.lru_cache(maxsize=32)
def has_hadamard_factors(n: int) -> bool:
  """Returns whether `hadamard_factors(n)` is defined."""
  if n <= 0:
    return False
  try:
    hadamard_factors(n)
  except ValueError:
    return False
  return True


@functools.lru_cache(maxsize=32)
def hadamard_matrix(n: int) -> np.ndarray:
  """Returns a read-only `int8` `(n, n)` unnormalized Hadamard matrix."""
  f1, f2 = hadamard_factors(n)
  h = np.kron(f1, f2).astype(np.int8, copy=False)
  h.setflags(write=False)
  return h


def random_signs(
    key: jax.Array,
    size: int,
    *,
    dtype: jax.typing.DTypeLike = jnp.bfloat16,
) -> jax.Array:
  """Samples a 1-D Rademacher (`+1` / `-1`) sign vector of length `size`."""
  if size <= 0:
    raise ValueError(f"size must be positive, got {size}.")
  bits = jax.random.bernoulli(key, 0.5, shape=(size,))
  return jnp.where(bits, 1, -1).astype(dtype)


def scale_from_absmax_f32(
    absmax_f32: jax.Array,
    *,
    orig_dtype: jax.typing.DTypeLike,
    qdtype: jnp.dtype,
) -> jax.Array:
  """Computes symmetric per-row scale rounded to `orig_dtype` in `float32`.

  See the module docstring for how this compares with
  `qarray.compute_scale_zero_point`. Only 32-bit values are broadcast, so Mosaic
  TPU kernels avoid 16-bit sublane+lane scalar broadcasts.

  Args:
    absmax_f32: Per-row maximum absolute values in `float32`, shape `(m, 1)`.
    orig_dtype: Unquantized activation dtype (`bfloat16` or `float32`).
    qdtype: Normalized target quantized dtype.

  Returns:
    Scale tensor of shape `(m, 1)` in `float32` whose values are representable
    in `orig_dtype`.
  """
  orig_dt = jnp.dtype(orig_dtype)
  qmax = float(np.asarray(numerics.get_symmetric_bound(qdtype), dtype=orig_dt))
  tiny_sqrt = float(np.sqrt(np.finfo(orig_dt).tiny))
  raw_scale_f32 = (absmax_f32 / qmax).astype(orig_dt).astype(jnp.float32)
  return jnp.where(raw_scale_f32 < tiny_sqrt, 1.0, raw_scale_f32)


def quantize_with_scale_2d(
    x: jax.Array,
    scale_f32: jax.Array,
    qdtype: jnp.dtype,
) -> jax.Array:
  """Quantizes 2-D `x` given precomputed `scale_f32` of shape `(m, 1)`.

  See the module docstring for how this compares with
  `qarray.quantize_with_scale_zero_point`.

  Args:
    x: 2-D activation tile of shape `(m, n)` in `bfloat16` or `float32`.
    scale_f32: Per-row scale in `float32` (rounded to `x.dtype`), shape `(m,
      1)`.
    qdtype: Normalized target quantized dtype.

  Returns:
    Quantized array of shape `(m, n)` with dtype `qdtype`.
  """
  scaled_f32 = (
      (x.astype(jnp.float32) / scale_f32).astype(x.dtype).astype(jnp.float32)
  )
  if jnp.issubdtype(qdtype, jnp.integer):
    iinfo = np.iinfo(qdtype)
    scaled_f32 = jnp.clip(scaled_f32, float(iinfo.min), float(iinfo.max))
  return numerics.convert_to(scaled_f32, qdtype)


def quantize_rows_2d(
    x: jax.Array,
    qtype: jax.typing.DTypeLike,
) -> tuple[jax.Array, jax.Array]:
  """Quantizes 2-D `x` symmetrically per row, returning `(xq, scale_f32)`."""
  qdtype = normalize_qtype(qtype)
  absmax_f32 = jnp.max(jnp.abs(x.astype(jnp.float32)), axis=-1, keepdims=True)
  scale_f32 = scale_from_absmax_f32(
      absmax_f32, orig_dtype=x.dtype, qdtype=qdtype
  )
  return quantize_with_scale_2d(x, scale_f32, qdtype), scale_f32


def validate_hadamard_args(
    x: jax.Array,
    *,
    signs: jax.Array | None = None,
    block_size: int | None = None,
    axis: int = -1,
) -> int:
  """Validates the RHT arguments and returns `axis` as a non-negative index."""
  if x.ndim < 1:
    raise ValueError(f"x must have rank >= 1, got shape {x.shape}.")
  if x.dtype not in (jnp.bfloat16, jnp.float32):
    raise ValueError(f"x must be bfloat16 or float32, got {x.dtype}.")
  norm_axis = axis if axis >= 0 else x.ndim + axis
  if norm_axis < 0 or norm_axis >= x.ndim:
    raise ValueError(
        f"axis {axis} is out of bounds for array of shape {x.shape}."
    )
  k = x.shape[norm_axis]
  if signs is not None:
    if signs.ndim != 1 or signs.shape[0] != k:
      raise ValueError(
          f"signs must have shape ({k},) to match x.shape[{norm_axis}], got"
          f" {signs.shape}."
      )
  if block_size is not None:
    if block_size <= 0 or k % block_size != 0:
      raise ValueError(
          f"x.shape[{norm_axis}]={k} must be divisible by positive block_size,"
          f" got block_size={block_size}."
      )
    if not has_hadamard_factors(block_size):
      raise ValueError(
          f"block_size={block_size} has no Sylvester / Paley-I Hadamard"
          " factorization."
      )
  elif not has_hadamard_factors(k):
    raise ValueError(
        f"x.shape[{norm_axis}]={k} has no Sylvester / Paley-I Hadamard"
        " factorization."
    )
  return norm_axis


def hadamard_rotate_reference(
    x: jax.Array,
    *,
    signs: jax.Array | None = None,
    block_size: int | None = None,
    axis: int = -1,
    inverse: bool = False,
) -> jax.Array:
  """Plain-XLA reference for the (inverse) Randomized Hadamard Transform."""
  norm_axis = validate_hadamard_args(
      x, signs=signs, block_size=block_size, axis=axis
  )
  k = x.shape[norm_axis]
  rot_dim = k if block_size is None else block_size
  f1, f2 = hadamard_factors(rot_dim)
  if inverse:
    f1, f2 = f1.T, f2.T
  m1, m2 = f1.shape[0], f2.shape[0]

  y = jnp.moveaxis(x, norm_axis, -1).astype(jnp.float32)
  if signs is not None and not inverse:
    y = y * signs.astype(jnp.float32)

  lead_shape = y.shape[:-1]
  num_blocks = k // rot_dim
  y = y.reshape(lead_shape + (num_blocks, m1, m2))
  y = jnp.einsum(
      "...gab,aA->...gAb",
      y,
      jnp.asarray(f1, dtype=jnp.float32),
      precision=jax.lax.Precision.HIGHEST,
  )
  y = jnp.einsum(
      "...gAb,bB->...gAB",
      y,
      jnp.asarray(f2, dtype=jnp.float32),
      precision=jax.lax.Precision.HIGHEST,
  )
  y = (y * np.float32(1.0 / math.sqrt(rot_dim))).reshape(lead_shape + (k,))
  if signs is not None and inverse:
    y = y * signs.astype(jnp.float32)
  return jnp.moveaxis(y.astype(x.dtype), -1, norm_axis)


def hadamard_rotate_quantize_reference(
    x: jax.Array,
    *,
    qtype: jax.typing.DTypeLike = jnp.int8,
    signs: jax.Array | None = None,
    block_size: int | None = None,
    axis: int = -1,
) -> QArray:
  """Plain-XLA reference for `hadamard_rotate_quantize` returning `QArray`."""
  qdtype = normalize_qtype(qtype)
  norm_axis = validate_hadamard_args(
      x, signs=signs, block_size=block_size, axis=axis
  )
  y = hadamard_rotate_reference(
      x, signs=signs, block_size=block_size, axis=norm_axis
  )
  channelwise_axes = tuple(i for i in range(x.ndim) if i != norm_axis)
  return qarray.quantize_api(y, qdtype, channelwise_axes=channelwise_axes)
