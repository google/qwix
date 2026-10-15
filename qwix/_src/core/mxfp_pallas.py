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
"""Pallas TPU kernel for MXFP4 weights x FP8 activations.

The kernel folds the power-of-two MXFP4 block scales into the e5m2 exponent of
the weight inside VMEM, so the weight stays fp4 in HBM and the matmul runs at
fp8 speed on the MXU. Like all Mosaic kernels it is not partitioned by XLA:
under tensor parallelism it must be called from a `shard_map`.
"""

import functools
import math
import jax
from jax import numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

BLOCK_SIZE = 32
# 6 * 2**13 fits e5m2; blocks 2**26 below the tile max flush to zero.
_HEADROOM = 2.0**13


def _kernel(
    x_ref, x_scale_ref, w_ref, w_scale_ref, o_ref, acc_ref, *, num_steps: int
):
  """Computes one output tile over the contraction steps."""
  step = pl.program_id(w_ref.ndim)
  tk, tn = w_ref.shape[0], math.prod(w_ref.shape[1:])

  @pl.when(step == 0)
  def _():
    acc_ref[...] = jnp.zeros_like(acc_ref)

  # Shift each column's max scale to _HEADROOM so w * ratio fits e5m2.
  w = w_ref[...].astype(jnp.bfloat16).reshape(tk // BLOCK_SIZE, BLOCK_SIZE, tn)
  s = w_scale_ref[...].reshape(tk // BLOCK_SIZE, tn)
  s_max = jnp.max(s, axis=0, keepdims=True)
  s_max = jnp.where(s_max == 0, 1.0, s_max)
  ratio = s * (_HEADROOM / s_max)
  w = (w * ratio[:, None, :]).reshape(tk, tn).astype(jnp.float8_e5m2)
  dot = jnp.dot(x_ref[...], w, preferred_element_type=jnp.float32)
  acc_ref[...] += dot * (s_max.astype(jnp.float32) / _HEADROOM)

  @pl.when(step == num_steps - 1)
  def _():
    out = acc_ref[...] * x_scale_ref[...]
    o_ref[...] = out.astype(o_ref.dtype).reshape(o_ref.shape)


def mxfp4_matmul(
    x: jax.Array,
    x_scale: jax.Array,
    w_q: jax.Array,
    w_scale: jax.Array,
    *,
    out_dtype: jax.typing.DTypeLike,
    interpret: bool = False,
) -> jax.Array:
  """Computes (x * x_scale) @ (w_q * w_scale) without dequantizing w_q in HBM.

  Args:
    x: [M, K] fp8 activations.
    x_scale: [M, 1] activation scales.
    w_q: [K, ...N] fp4 weights.
    w_scale: [K // 32, ...N] power-of-two block scales.
    out_dtype: The dtype of the result.
    interpret: Run the kernel in interpret mode, e.g. for tests on CPU.

  Returns:
    The [M, ...N] result.
  """
  m, k = x.shape
  out_free_shape = w_q.shape[1:]
  n1, n2 = math.prod(out_free_shape[:-1]), out_free_shape[-1]
  # Keep 3D [K, N1, N2] when (N1, N2) is (8, 128)-aligned (e.g. multi-head
  # q/k/v weights) so XLA does not relayout between 3D and 2D in HBM.
  if len(out_free_shape) >= 2 and n1 % 8 == 0 and n2 % 128 == 0:
    n_shape, tn_shape = (n1, n2), (8, 128)
  else:
    n = n1 * n2
    n_shape, tn_shape = (n,), (1024 if n % 1024 == 0 else n,)
  w_q = w_q.reshape(k, *n_shape)
  w_scale = w_scale.reshape(w_scale.shape[0], *n_shape).astype(jnp.bfloat16)
  x_scale = x_scale.astype(jnp.float32)

  # Multiples of 256 keep the (tk // 32, tn) scale blocks 8-aligned.
  tk = next((t for t in (2048, 1024, 512, 256) if k >= 4096 and k % t == 0), k)
  m_padded = m if m <= 1024 else pl.cdiv(m, 256) * 256
  tm = next((t for t in (1024, 512, 256) if m > 1024 and m_padded % t == 0), m)
  if m_padded != m:
    x = jnp.pad(x, ((0, m_padded - m), (0, 0)))
    x_scale = jnp.pad(x_scale, ((0, m_padded - m), (0, 0)))

  num_steps = k // tk
  grid = (
      m_padded // tm,
      *(s // ts for s, ts in zip(n_shape, tn_shape)),
      num_steps,
  )
  # Pass num_steps via partial so Pallas's Jaxpr cache (keyed by block shapes,
  # not grid) does not reuse a trace compiled for a different K // tk.
  out = pl.pallas_call(
      functools.partial(_kernel, num_steps=num_steps),
      grid=grid,
      in_specs=[
          pl.BlockSpec((tm, tk), lambda i, *js: (i, js[-1])),
          pl.BlockSpec((tm, 1), lambda i, *js: (i, 0)),
          pl.BlockSpec((tk, *tn_shape), lambda i, *js: (js[-1], *js[:-1])),
          pl.BlockSpec(
              (tk // BLOCK_SIZE, *tn_shape), lambda i, *js: (js[-1], *js[:-1])
          ),
      ],
      out_specs=pl.BlockSpec((tm, *tn_shape), lambda i, *js: (i, *js[:-1])),
      out_shape=jax.ShapeDtypeStruct((m_padded, *n_shape), out_dtype),
      scratch_shapes=[pltpu.VMEM((tm, math.prod(tn_shape)), jnp.float32)],
      compiler_params=pltpu.CompilerParams(
          dimension_semantics=('parallel',) * len(n_shape)
          + ('parallel', 'arbitrary'),
      ),
      interpret=interpret,
  )(x, x_scale, w_q, w_scale)
  return out[:m].reshape(m, *out_free_shape)
