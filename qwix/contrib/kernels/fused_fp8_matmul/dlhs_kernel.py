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

"""Pallas/Mosaic TPU kernel for the fused FP8 activation gradient (dLHS).

Computes `dlhs = dout @ rhs^T` for the fused FP8 matmul, where `rhs` is the
`(k, n)` FP8 weight with one float32 scale per column, i.e. the residuals the
forward pass saved. The contraction runs over `n`, so the column scale cannot
be factored out of it: the kernel folds it into the cotangent, quantizes the
scaled cotangent to FP8 with one scale per row, runs the contraction on the FP8
MXU and applies the row scale to the result. The quantized cotangent never
leaves VMEM.

The numerics are those of `reference.fused_fp8_dlhs_reference`.
"""

import functools
import itertools

import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
from qwix.contrib.kernels.fused_fp8_matmul import forward_kernel
from qwix.contrib.kernels.fused_fp8_matmul import reference

_CORE_AXIS_NAME = "core"

# Budget used when choosing dLHS tiles. Higher than the forward kernel's
# `VMEM_BUDGET_BYTES` because `dlhs_vmem_bytes` is a tighter model: every
# buffer it counts is one the pipeline actually places. For a bfloat16
# cotangent the scale-folded tile is left out because Mosaic rematerializes it
# rather than holding a full `(bm, bn)` copy; for float32 it is counted, see
# `dlhs_vmem_bytes`. Calibrated against compile outcomes on TPU7x:
# `dlhs_kernel_test` compiles the heuristic tiling for every test shape, up to
# a modeled 60.4 MiB on the single-block path ((2048, 7168, 7168) in bfloat16)
# and 58.0 MiB on the cotangent-cache path ((2048, 4096, 16384) in bfloat16);
# see `test_heuristic_tiling_at_the_vmem_budget`.
VMEM_BUDGET_BYTES = 62 * 1024 * 1024


def dlhs_vmem_bytes(
    bm: int, bk: int, bn: int, k: int, n: int, *, cotangent_itemsize: int
) -> int:
  """Estimates the dLHS kernel's scoped VMEM footprint in bytes.

  Mirrors `forward_kernel.forward_vmem_bytes`: the weight block is single
  buffered when `bk == k` and `bn == n` make its index map constant, and the
  accumulator and the cotangent cache are counted only on the paths that read
  them.

  Args:
    bm: Block size along `m`.
    bk: Block size along `k`.
    bn: Block size along `n`, the contracting dimension.
    k: The output column dimension of `dlhs`.
    n: The contracting dimension.
    cotangent_itemsize: Bytes per element of the cotangent, which is also the
      output dtype: 2 for `bfloat16`, 4 for `float32`.

  Returns:
    The estimated footprint in bytes.
  """
  grid_k = -(-k // bk)
  grid_n = -(-n // bn)
  w_buffers = 1 if (grid_k == 1 and grid_n == 1) else 2
  g_block = bm * bn * cotangent_itemsize * 2  # cotangent in, double buffered
  o_block = bm * bk * cotangent_itemsize * 2  # gradient out, double buffered
  gq = bm * bn * 1  # fp8 quantized cotangent tile
  dot_out = bm * bk * 4  # float32 dot_general result
  w_block = bk * bn * 1 * w_buffers  # fp8 weights
  accum = bm * bk * 4 if grid_n > 1 else 0
  gq_cache = grid_n * bm * bn * 1 if grid_k > 1 else 0
  # The scale-folded cotangent tile. For bfloat16 Mosaic rematerializes it and
  # the production tiling compiles with no room for a copy. For float32 it does
  # not: at (512, 1024, 8192) on (4096, 8192, 8192) the tile showed up as
  # 16 MiB of spill slots and the kernel ran out of VMEM.
  g_folded = bm * bn * cotangent_itemsize if cotangent_itemsize > 2 else 0
  return (
      g_block + o_block + gq + dot_out + w_block + accum + gq_cache + g_folded
  )


def _tile_candidates(dim: int, candidates: tuple[int, ...]) -> tuple[int, ...]:
  """Returns the candidates that divide `dim`, largest-preferred order kept.

  Falls back to `forward_kernel.pick_tile`'s power-of-two divisor when none of
  them divides `dim`, so odd shapes still get a valid tile.

  Args:
    dim: The array dimension to tile.
    candidates: Candidate tile sizes, listed largest-preferred.

  Returns:
    A non-empty tuple of tile sizes, each dividing `dim`.
  """
  dividing = tuple(c for c in candidates if c <= dim and dim % c == 0)
  return dividing or (forward_kernel.pick_tile(dim, candidates),)


def fit_dlhs_tiles(
    m: int, k: int, n: int, *, cotangent_itemsize: int
) -> tuple[int, int, int]:
  """Picks `(bm, bk, bn)` for the dLHS kernel.

  `bn == n` matters most: it removes the reduction loop and quantizes each
  cotangent tile exactly once. Next comes `bk == k`, so the cotangent is read
  once rather than `k / bk` times, and with `bn == n` as well the weight block
  index is constant across the whole grid, so the weights load once too.
  Measured on (32768, 3072, 4096) at `bm = 512`: 571.4 us with `bk == k`
  against 651.2 us at `bk = 1536` and 666.3 us at `bk = 1024`, i.e. 1.14x over
  the best smaller k block. `bm` carries no traffic penalty, but a larger block
  means fewer grid steps, worth a few percent where it fits: so take the
  largest `bm` the VMEM model allows.

  The search follows that order of preference: for each `bn` (largest first),
  for each `bk` (largest first), the largest `bm` that fits the VMEM budget
  wins. So `bk` shrinks before `bn` does, and `bn` only shrinks once no
  `(bk, bm)` fits at the larger `bn`, which happens when `k * n` alone
  outgrows VMEM.

  Args:
    m: The row dimension.
    k: The output column dimension of `dlhs`.
    n: The contracting dimension.
    cotangent_itemsize: Bytes per element of the cotangent, see
      `dlhs_vmem_bytes`.

  Returns:
    `(bm, bk, bn)`, each dividing its dimension.
  """
  bns = _tile_candidates(n, (n, 2048, 1024, 512))
  bks = _tile_candidates(k, (k, 1536, 1024, 512, 256))
  bms = _tile_candidates(m, (1024, 512, 256, 128))
  for bn, bk, bm in itertools.product(bns, bks, bms):
    vmem = dlhs_vmem_bytes(
        bm, bk, bn, k, n, cotangent_itemsize=cotangent_itemsize
    )
    if vmem <= VMEM_BUDGET_BYTES:
      return bm, bk, bn
  # Nothing fits the model's budget; the smallest tiles are the best bet.
  return bms[-1], bks[-1], bns[-1]


def get_dlhs_heuristics_config(
    m: int, k: int, n: int, *, lhs_dtype: jax.typing.DTypeLike = jnp.bfloat16
) -> forward_kernel.Config:
  """Returns the tiling the measured heuristics pick for the dLHS kernel.

  With `block_n == n` each cotangent row is quantized against a single scale
  and no accumulator is needed; with `block_k < k` the kernel caches the
  quantized cotangent across k blocks. Measured on TPU7x.

  Args:
    m: The row dimension of the cotangent and of `dlhs`.
    k: The column dimension of `dlhs`.
    n: The contracting dimension (columns of the cotangent).
    lhs_dtype: The activation dtype, which is also the cotangent's. See
      `forward_kernel.get_heuristics_config` for why it matters.
  """
  bm, bk, bn = fit_dlhs_tiles(
      m, k, n, cotangent_itemsize=jnp.dtype(lhs_dtype).itemsize
  )
  return forward_kernel.Config(block_m=bm, block_k=bk, block_n=bn)


def _quantize_g(g_vmem: jax.Ref, sw_vmem: jax.Ref):
  """Folds the column scales into the cotangent tile and quantizes it.

  The multiply stays in the cotangent's own dtype. Promoting the `(bm, bn)`
  tile to float32 here, as an earlier version did, made a 16 MB temporary at
  `bm = 1024, bn = 4096` and most large tilings failed to compile.

  Args:
    g_vmem: The `(bm, bn)` cotangent block.
    sw_vmem: The `(bn,)` column scales.

  Returns:
    `(gq, sg)`: the FP8 tile and its `(bm, 1)` float32 row scales.
  """
  g = g_vmem[...]
  return reference.quantize_rows_fp8(g * sw_vmem[...].astype(g.dtype))


def _dlhs_kernel(
    g_hbm: jax.Ref,
    w_hbm: jax.Ref,
    sw_hbm: jax.Ref,
    o_hbm: jax.Ref,
    accum_vmem: jax.Ref,
    gq_cache: jax.Ref,
    sg_cache: jax.Ref,
    *,
    bm: int,
    bk: int,
    bn: int,
):
  """Pipelined dLHS kernel body. See `fused_fp8_dlhs_pallas`.

  Args:
    g_hbm: `(m, n)` cotangent.
    w_hbm: `(k, n)` FP8 weights.
    sw_hbm: `(n,)` float32 weight scales.
    o_hbm: `(m, k)` activation gradient.
    accum_vmem: `(bm, bk)` float32 accumulator, read only when `bn < n`.
    gq_cache: `(n / bn, bm, bn)` FP8 cache of the quantized cotangent, read only
      when `bk < k`.
    sg_cache: `(n / bn, bm, 1)` float32 cache of the cotangent row scales, read
      only when `bk < k`.
    bm: Block size along `m`.
    bk: Block size along `k`.
    bn: Block size along `n`.
  """
  m, n = g_hbm.shape
  k, _ = w_hbm.shape
  grid = (pl.cdiv(m, bm), pl.cdiv(k, bk), pl.cdiv(n, bn))
  single_k = grid[1] == 1
  single_n = grid[2] == 1

  g_spec = pl.BlockSpec((bm, bn), lambda mi, ki, ni: (mi, ni))
  w_spec = pl.BlockSpec((bk, bn), lambda mi, ki, ni: (ki, ni))
  # A 1-D VMEM block for the column scales; see the forward kernel's
  # `SX_LANES` for why a `(1, bn)` block would be a bad idea.
  sw_spec = pl.BlockSpec((bn,), lambda mi, ki, ni: (ni,))
  o_spec = pl.BlockSpec((bm, bk), lambda mi, ki, ni: (mi, ki))

  def _load_gq(g_vmem: jax.Ref, sw_vmem: jax.Ref):
    """Quantizes each `(bm, bn)` cotangent tile once and reuses it across k.

    `g_spec` does not depend on the k-block index `ki`, so a naive body would
    re-quantize the identical tile `cdiv(k, bk)` times; that VPU work, not the
    FP8 dot, dominated the first version of this kernel. The cache is keyed by
    the n-block index `ni`: with `ni` innermost, the `ki == 0` sweep fills every
    entry before any `ki > 0` step reads it. With a single k block there is
    nothing to reuse and the cache is bypassed.

    Args:
      g_vmem: The `(bm, bn)` cotangent block.
      sw_vmem: The `(bn,)` column scales.

    Returns:
      `(gq, sg)`: the FP8 tile and its `(bm, 1)` float32 row scales.
    """
    if single_k:
      return _quantize_g(g_vmem, sw_vmem)

    kind = pl.program_id(1)
    nind = pl.program_id(2)

    @pl.when(kind == 0)
    def _fill():
      gq_cache[nind], sg_cache[nind] = _quantize_g(g_vmem, sw_vmem)

    return gq_cache[nind], sg_cache[nind]

  def _partial(g_vmem: jax.Ref, w_vmem: jax.Ref, sw_vmem: jax.Ref):
    gq, sg = _load_gq(g_vmem, sw_vmem)
    gw = jax.lax.dot_general(
        gq,
        w_vmem[...],
        dimension_numbers=(((1,), (1,)), ((), ())),
        preferred_element_type=jnp.float32,
    )
    return gw * sg  # `(bm, 1)` broadcasts along lanes for free.

  def body_single_n(g_vmem, w_vmem, sw_vmem, o_vmem):
    """Direct pass when `bn == n`: no accumulator needed."""
    o_vmem[...] = _partial(g_vmem, w_vmem, sw_vmem).astype(o_vmem.dtype)

  def body_multi_n(g_vmem, w_vmem, sw_vmem, o_vmem):
    """Pipelined reduction over n blocks when `bn < n`."""
    nind = pl.program_id(2)

    @pl.when(nind == 0)
    def _init():
      accum_vmem[...] = jnp.zeros_like(accum_vmem)

    # Each n block is quantized against its own row absmax, so the partial
    # product is already rescaled by `_partial` before it is accumulated.
    accum_vmem[...] += _partial(g_vmem, w_vmem, sw_vmem)

    @pl.when(nind == pl.num_programs(2) - 1)
    def _write():
      o_vmem[...] = accum_vmem[...].astype(o_vmem.dtype)

  pltpu.emit_pipeline(
      body_single_n if single_n else body_multi_n,
      grid=grid,
      in_specs=[g_spec, w_spec, sw_spec],
      out_specs=o_spec,
      core_axis_name=_CORE_AXIS_NAME,
      # The cotangent cache is filled on the `ki == 0` sweep and read on later
      # k steps of the same m block, so the k axis must not be split across
      # cores when there is more than one k block; nor the n reduction when it
      # is accumulated. Only the m axis is always safe to partition.
      dimension_semantics=(
          pltpu.PARALLEL,
          pltpu.PARALLEL if single_k else pltpu.ARBITRARY,
          pltpu.PARALLEL if single_n else pltpu.ARBITRARY,
      ),
  )(g_hbm, w_hbm, sw_hbm, o_hbm)


def fused_fp8_dlhs_pallas(
    dout: jax.Array,
    rhs_q: jax.Array,
    rhs_scale: jax.Array,
    *,
    bm: int,
    bk: int,
    bn: int,
    out_dtype: jnp.dtype | None = None,
) -> jax.Array:
  """Fused FP8 activation gradient: `quantize_rows(dout * rhs_scale) @ rhs_q^T`.

  Args:
    dout: Cotangent of shape `(m, n)`, `bfloat16` or `float32`.
    rhs_q: Weights of shape `(k, n)` already quantized to `float8_e4m3fn`.
    rhs_scale: Per-column weight scales of shape `(n,)`, `float32`.
    bm: Block size along `m`. Must divide `m`.
    bk: Block size along `k`. Must divide `k`.
    bn: Block size along `n`, the contracting dimension. Must divide `n`. With
      `bn == n` each cotangent row is quantized against a single scale and no
      accumulator is needed; with `bn < n` each n block gets its own scale.
    out_dtype: Output dtype. Defaults to `dout.dtype`.

  Returns:
    `dlhs` of shape `(m, k)`.
  """
  m, n = dout.shape
  k, n_w = rhs_q.shape
  if n_w != n:
    raise ValueError(f"Contracting dims differ: dout has n={n}, rhs has {n_w}.")
  if rhs_scale.shape != (n,):
    raise ValueError(
        f"rhs_scale must have shape ({n},), got {rhs_scale.shape}."
    )
  if rhs_q.dtype != reference.FP8_DTYPE:
    raise ValueError(f"rhs_q must be {reference.FP8_DTYPE}, got {rhs_q.dtype}.")
  if m % bm:
    raise ValueError(f"m ({m}) must be divisible by bm ({bm}).")
  if k % bk:
    raise ValueError(f"k ({k}) must be divisible by bk ({bk}).")
  if n % bn:
    raise ValueError(f"n ({n}) must be divisible by bn ({bn}).")

  grid_k = pl.cdiv(k, bk)
  grid_n = pl.cdiv(n, bn)

  # Scratch order matches the kernel signature: accum, gq_cache, sg_cache.
  # Buffers the chosen tiling leaves dead are allocated at a minimal shape
  # rather than dropped, so the positional layout stays fixed. At the tiling
  # the heuristics prefer (`bk == k`, `bn == n`) both caches are dead, and
  # holding them at full size (8.4 MB at `bm = 512` on (32768, 3072, 4096))
  # is what used to keep `bm = 1024` from compiling.
  dead = (8, 128)  # One (sublane, lane) tile, the smallest useful allocation.
  scratch_types = [
      pltpu.VMEM((bm, bk) if grid_n > 1 else dead, jnp.float32),
      pltpu.VMEM(
          (grid_n, bm, bn) if grid_k > 1 else (1,) + dead, reference.FP8_DTYPE
      ),
      pltpu.VMEM((grid_n, bm, 1) if grid_k > 1 else (1,) + dead, jnp.float32),
  ]

  kernel = functools.partial(_dlhs_kernel, bm=bm, bk=bk, bn=bn)
  return pl.kernel(
      kernel,
      out_type=jax.ShapeDtypeStruct((m, k), out_dtype or dout.dtype),
      mesh=pltpu.create_tensorcore_mesh(axis_name=_CORE_AXIS_NAME),
      scratch_types=scratch_types,
      compiler_params=pltpu.CompilerParams(
          vmem_limit_bytes=forward_kernel.VMEM_LIMIT_BYTES
      ),
  )(dout, rhs_q, rhs_scale)
