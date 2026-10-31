"""ADMM-Q algorithm on Qwix's quantization grid.

This is the ADMM-Q weight quantizer from

  ADMM-Q: An Improved Hessian-based Weight Quantizer for Post-Training
  Quantization of Large Language Models (COLM 2026)

written against Qwix's quantization primitives, so that its result is a regular
``qarray.QArray`` that Qwix's PtqProvider can serve. It follows the same
contract as ``qwix.contrib.gptq_core.quantize_weight``: it minimizes
||W @ X - W_q @ X||^2 given the Hessian H = X @ X.T.

The algorithm is the one in ``admm-q-main/solvers_jax.py`` (itself a port of
``solvers.py``):

* Scales come from a clip-ratio search over Qwix's own calibration (absmax,
  minmax, ...), shrinking the calibrated range and keeping the ratio with the
  smallest |error|^2.4 per scale tile.
* ADMM splits the weights into a free copy B and an on-grid copy D. Each
  iteration solves for B exactly via (H + rho I)^-1, built from one float64
  eigendecomposition, projects onto the grid for D, and updates the dual V.
  rho grows every ``update_iter`` iterations.
* If the error stalls, the scales are re-fitted once to the current iterate and
  kept only if they lower the true Hessian-weighted error.
* Ill-conditioned layers get a smaller rho, slower growth, and a final greedy
  PairSwap pass.

Grid conventions are Qwix's, not SpinQuant's: for int4, codes lie in [-8, 7];
symmetric scales are absmax / 7.5 (``numerics.get_symmetric_bound``) and
asymmetric ones use an integer zero point in the same signed range.

Only integer qtypes are supported (int2-int8, including the synthetic 'int3'
etc.). ``is_supported`` tells callers when to fall back to plain PTQ.
"""

# We try to use the same naming as solvers.py / gptq_core.py, thus
# pylint: disable=invalid-name

import functools
from typing import Any

from absl import logging
import jax
from jax import lax
import jax.numpy as jnp
import numpy as np
from qwix._src.core import qarray

F32 = jnp.float32
F64 = jnp.float64


# ---------------------------------------------------------------------------
# Grid helpers
# ---------------------------------------------------------------------------


def _int_bounds(qtype: Any) -> tuple[float, float]:
  """Returns the (qmin, qmax) integer codes for qtype, or raises ValueError."""
  if isinstance(qtype, str):
    if qtype in ('int2', 'int3', 'int5', 'int6', 'int7'):
      bits = int(qtype[3:])
      return -(2.0 ** (bits - 1)), 2.0 ** (bits - 1) - 1
    raise ValueError(f'ADMM-Q does not support qtype {qtype!r}.')
  dtype = jnp.dtype(qtype)
  if not jnp.issubdtype(dtype, jnp.integer) or dtype.itemsize > 1:
    raise ValueError(f'ADMM-Q does not support qtype {qtype!r}.')
  info = jnp.iinfo(dtype)
  return float(info.min), float(info.max)


def is_supported(how: qarray.HowToQuantize) -> bool:
  """Whether ADMM-Q can quantize with this HowToQuantize."""
  try:
    _int_bounds(how.qtype)
  except (ValueError, TypeError):
    return False
  return how.noise_fn is None


def _how_key(how: qarray.HowToQuantize) -> tuple[Any, ...]:
  """A hashable stand-in for HowToQuantize (its tiled_axes is a dict)."""
  return (
      how.qtype,
      tuple(how.channelwise_axes),
      tuple(sorted(how.tiled_axes.items())),
      how.calibration_method,
  )


def _how_from_key(key: tuple[Any, ...]) -> qarray.HowToQuantize:
  qtype, channelwise_axes, tiled_axes, calibration_method = key
  return qarray.HowToQuantize(
      qtype=qtype,
      channelwise_axes=channelwise_axes,
      tiled_axes=dict(tiled_axes),
      calibration_method=calibration_method,
  )


def _expand(x: jax.Array, shape: tuple[int, int]) -> jax.Array:
  """Broadcasts a (sR, sC) scale/zero-point grid to the full (R, C) shape."""
  (R, C), (sR, sC) = shape, x.shape
  return jnp.repeat(jnp.repeat(x, R // sR, axis=0), C // sC, axis=1)


def _reduce(e: jax.Array, scale_shape: tuple[int, ...]) -> jax.Array:
  """Sums a full (R, C) array over each scale tile, giving (sR, sC)."""
  (R, C), (sR, sC) = e.shape, scale_shape
  return e.reshape(sR, R // sR, sC, C // sC).sum(axis=(1, 3))


def _zero_like_scale(zp: jax.Array | None, scale: jax.Array) -> jax.Array:
  return jnp.zeros_like(scale) if zp is None else zp.astype(scale.dtype)


def fake_quant(x, scale, zero, qmin, qmax):
  """Rounds x onto the grid and maps it back. scale/zero are full size."""
  q = jnp.clip(jnp.round(x / scale + zero), qmin, qmax)
  return (q - zero) * scale


def find_scale_zero(
    w: jax.Array,
    target: jax.Array,
    how: qarray.HowToQuantize,
    *,
    mse: bool = True,
    norm: float = 2.4,
    grid: int = 100,
    maxshrink: float = 0.8,
) -> tuple[jax.Array, jax.Array | None]:
  """Qwix calibration plus ADMM-Q's clip-ratio search.

  The calibrated range of ``w`` (absmax, or min/max) is shrunk by ratios
  1, 0.99, ..., 1 - maxshrink, and each scale tile keeps the ratio whose
  quantization of ``w`` is closest to ``target`` in the ``norm``-norm.

  Args:
    w: weights that define the grid, shape (rows, columns).
    target: weights the error is measured against, same shape.
    how: how to quantize.
    mse: if False, return Qwix's plain calibration (no search).
    norm: error exponent.
    grid: number of ratio steps per unit.
    maxshrink: the smallest ratio is 1 - maxshrink.

  Returns:
    (scale, zero_point) in Qwix's scale shape; zero_point is None for
    symmetric quantization.
  """
  calib = qarray.calibrate(w, how)
  scale, zp = qarray.compute_scale_zero_point(calib, how.qtype)
  if not mse:
    return scale, zp
  qmin, qmax = _int_bounds(how.qtype)
  shrink = jnp.asarray(
      1.0 - np.arange(int(maxshrink * grid)) / grid, dtype=w.dtype
  )
  zp0 = _zero_like_scale(zp, scale)

  def step(carry, p):
    best, s_best, z_best = carry
    s, z = qarray.compute_scale_zero_point(
        {k: v * p for k, v in calib.items()}, how.qtype
    )
    z = _zero_like_scale(z, s)
    dq = fake_quant(w, _expand(s, w.shape), _expand(z, w.shape), qmin, qmax)
    err = _reduce(jnp.abs(dq - target) ** norm, s.shape)
    better = err < best
    return (
        jnp.where(better, err, best),
        jnp.where(better, s, s_best),
        jnp.where(better, z, z_best),
    ), None

  best = jnp.full(scale.shape, jnp.inf, dtype=w.dtype)
  (_, scale, zp_f), _ = lax.scan(step, (best, scale, zp0), shrink)
  return scale, (None if zp is None else zp_f.astype(zp.dtype))


def _recon(D, W, H):
  """Hessian-weighted error  sum((D - W) H (D - W)^T)."""
  diff = D - W
  return jnp.sum((diff @ H) * diff)


# ---------------------------------------------------------------------------
# ADMM
# ---------------------------------------------------------------------------


@jax.jit
def _prepare(W, H, percdamp):
  """Damps and normalizes H, and eigendecomposes it once in float64."""
  diag_idx = jnp.arange(H.shape[0])
  damp = (percdamp * jnp.mean(jnp.diag(H))).astype(F32)
  XtX = H.at[diag_idx, diag_idx].add(damp)
  X_norm = jnp.maximum(jnp.sqrt(jnp.diag(XtX)), 1e-8)
  XtX = XtX / X_norm
  XtX = (XtX.T / X_norm).T
  YtX = (W * X_norm) @ XtX
  L, Q = jnp.linalg.eigh(XtX.astype(F64))
  return XtX, X_norm, YtX, L, Q


@functools.partial(
    jax.jit,
    static_argnames=(
        'how_key', 'asym', 'max_iter', 'update_iter', 'switch_iter',
        'stagnation_window', 'refresh_on_stagnation',
    ),
)
def _admm_loop(
    W, H, XtX, X_norm, YtX, L, Q, scale, zp,
    rho, rho_update, stagnation_tol, refresh_accept_rel, *,
    how_key, asym, max_iter, update_iter, switch_iter,
    stagnation_window, refresh_on_stagnation,
):
  """All ADMM iterations and the final projection, as one compiled loop.

  ``zp`` is always an array here (zeros for symmetric quantization, where
  ``asym`` is False).
  """
  how = _how_from_key(how_key)
  qmin, qmax = _int_bounds(how.qtype)
  n_checks = max(max_iter // update_iter, 1)
  xn = X_norm[:, None]
  YtX_T = YtX.T
  B_orig = (W * X_norm).T
  Res0 = jnp.sum(B_orig * YtX_T)

  def full(s, z):  # Qwix-shaped grid -> full size, transposed to B's layout
    return _expand(s, W.shape).T, _expand(z, W.shape).T

  def quant_B(B, S_B, Z_B):
    return fake_quant(B / xn, S_B, Z_B, qmin, qmax) * xn

  def inverse(rho):
    # (XtX + rho I)^{-1} = Q diag(1 / (L + rho)) Q^T
    return ((Q * (1.0 / (L + rho))) @ Q.T).astype(F32)

  S_B, Z_B = full(scale, zp)
  D = quant_B(B_orig, S_B, Z_B)
  D_base = jnp.maximum(jnp.sum(D**2), 1e-12)

  state = dict(
      i=jnp.int32(0), done=jnp.bool_(False),
      B=B_orig, V=jnp.zeros_like(B_orig), D=D, D_prev=D,
      rho=rho, inv=inverse(rho), scale=scale, zp=zp,
      k=jnp.int32(0), err=jnp.zeros(n_checks, F64),
      did_refresh=jnp.bool_(False), refresh_accepted=jnp.bool_(False),
  )
  can_refresh = refresh_on_stagnation and 1 <= stagnation_window <= n_checks

  def refresh(B, scale, zp):
    S_B, Z_B = full(scale, zp)
    old = _recon(quant_B(B, S_B, Z_B).T / X_norm, W, H).astype(F64)
    new_scale, new_zp = find_scale_zero((B / xn).T, W, how, norm=2.0)
    new_zp = _zero_like_scale(new_zp if asym else None, new_scale)
    S_B, Z_B = full(new_scale, new_zp)
    new = _recon(quant_B(B, S_B, Z_B).T / X_norm, W, H).astype(F64)
    accepted = (old - new) / jnp.maximum(jnp.abs(old), 1e-12) > refresh_accept_rel
    return (
        jnp.where(accepted, new_scale, scale),
        jnp.where(accepted, new_zp, zp),
        accepted,
    )

  def check(s):
    k, B, D = s['k'], s['B'], s['D']
    d_rel = jnp.sum((D - s['D_prev']) ** 2) / D_base
    S_B, Z_B = full(s['scale'], s['zp'])
    Btest = quant_B(B, S_B, Z_B)
    Resc = jnp.sum((Btest - B_orig) * (XtX @ Btest - YtX_T))
    err = s['err'].at[k].set((Resc / Res0).astype(F64))
    s = dict(s, k=k + 1, D_prev=D, err=err)

    if can_refresh:
      n = k + 1
      recent = lax.dynamic_slice(
          err, (jnp.maximum(n - stagnation_window, 0),), (stagnation_window,)
      )
      recent_rel_improve = (recent[0] - recent.min()) / jnp.maximum(
          jnp.abs(recent[0]), 1e-12
      )
      should = (
          ~s['did_refresh']
          & (n >= max(2, stagnation_window))
          & (recent_rel_improve < stagnation_tol)
      )
      scale, zp, accepted = lax.cond(
          should,
          lambda: refresh(B, s['scale'], s['zp']),
          lambda: (s['scale'], s['zp'], jnp.bool_(False)),
      )
      s = dict(
          s, scale=scale, zp=zp,
          did_refresh=s['did_refresh'] | should,
          refresh_accepted=s['refresh_accepted'] | accepted,
      )

    rho_new = jnp.minimum(s['rho'] * rho_update, 1e6)
    converged = (s['i'] >= switch_iter) & (d_rel < 1e-6)
    return dict(s, rho=rho_new, inv=inverse(rho_new), done=converged)

  def admm_step(s):
    rho32 = s['rho'].astype(F32)
    V, B = s['V'], s['B']
    S_B, Z_B = full(s['scale'], s['zp'])
    D = quant_B((V + rho32 * B) / rho32, S_B, Z_B)
    B = s['inv'] @ (YtX_T - V + rho32 * D)
    V = V + rho32 * (B - D)
    s = dict(s, B=B, V=V, D=D)
    s = lax.cond((s['i'] + 1) % update_iter == 0, check, lambda s: s, s)
    return dict(s, i=s['i'] + 1)

  s = lax.while_loop(
      lambda s: (s['i'] < max_iter) & ~s['done'], admm_step, state
  )

  S_B, Z_B = full(s['scale'], s['zp'])
  W_q = quant_B(s['B'], S_B, Z_B).T / X_norm
  W_q = fake_quant(W_q, S_B.T, Z_B.T, qmin, qmax)
  info = dict(
      iterations=s['i'], converged=s['done'],
      refreshed=s['did_refresh'], refresh_accepted=s['refresh_accepted'],
  )
  return W_q, s['scale'], s['zp'], info


# ---------------------------------------------------------------------------
# PairSwap
# ---------------------------------------------------------------------------


@functools.partial(
    jax.jit, static_argnames=('qmin', 'qmax', 'n_rounds', 'n_batches', 'batch')
)
def _pairswap(W, H, S, Z, D, key, *, qmin, qmax, n_rounds, n_batches, batch):
  """Greedy pair-swap local search; S and Z are full size (R, C).

  Each batch samples ``batch`` disjoint column pairs, scores all four +/-1
  grid-step moves per pair and output channel, and applies the single best
  pair. Stops early after a round with no flips.
  """
  _, D_in = W.shape
  diag_idx = jnp.arange(D_in)
  damp = (1e-4 * jnp.mean(jnp.diag(H))).astype(F32)
  Hd = H.at[diag_idx, diag_idx].add(damp)
  H_diag = jnp.diag(Hd)
  sign1 = jnp.array([+1.0, +1.0, -1.0, -1.0], F32)
  sign2 = jnp.array([+1.0, -1.0, +1.0, -1.0], F32)

  def one_batch(b, carry):
    D, G, n_flips, round_key = carry
    perm = jax.random.permutation(jax.random.fold_in(round_key, b), D_in)
    j1, j2 = perm[:batch], perm[batch : 2 * batch]
    g1, g2 = G[:, j1], G[:, j2]
    s1, s2 = S[:, j1], S[:, j2]
    h11, h22, h12 = H_diag[j1], H_diag[j2], Hd[j1, j2]

    # Which +/-1 steps stay on the grid
    q1 = jnp.round(D[:, j1] / s1) + Z[:, j1]
    q2 = jnp.round(D[:, j2] / s2) + Z[:, j2]
    up1, dn1 = q1 < qmax, q1 > qmin
    up2, dn2 = q2 < qmax, q2 > qmin
    valid = jnp.stack([up1 & up2, up1 & dn2, dn1 & up2, dn1 & dn2])

    # Change in loss for all 4 directions
    d1v = sign1[:, None, None] * s1[None]
    d2v = sign2[:, None, None] * s2[None]
    dL = (
        -2.0 * d1v * g1 + d1v * d1v * h11
        - 2.0 * d2v * g2 + d2v * d2v * h22
        + 2.0 * d1v * d2v * h12
    )
    dL = jnp.where(valid, dL, 0.0)

    best_dL = dL.min(axis=0)
    best_id = dL.argmin(axis=0)
    accept = best_dL < 0
    improve = jnp.where(accept, -best_dL, 0.0).sum(axis=0)

    # Apply the single best pair (a no-op when nothing improves)
    bp = improve.argmax()
    acc = accept[:, bp] & (improve[bp] > 0)
    bid = best_id[:, bp]
    c1, c2 = j1[bp], j2[bp]
    d1 = jnp.where(acc, S[:, c1] * sign1[bid], 0.0)
    d2 = jnp.where(acc, S[:, c2] * sign2[bid], 0.0)
    D = D.at[:, c1].add(d1).at[:, c2].add(d2)
    G = G - jnp.outer(d1, Hd[c1]) - jnp.outer(d2, Hd[c2])
    return D, G, n_flips + acc.sum(dtype=jnp.int32), round_key

  def one_round(st):
    rd, D, _ = st
    G = (W - D) @ Hd
    D, _, n_flips, _ = lax.fori_loop(
        0, n_batches, one_batch,
        (D, G, jnp.int32(0), jax.random.fold_in(key, rd)),
    )
    return rd + 1, D, n_flips == 0

  _, D, _ = lax.while_loop(
      lambda st: (st[0] < n_rounds) & ~st[2],
      one_round,
      (jnp.int32(0), D, jnp.bool_(False)),
  )
  return fake_quant(D, S, Z, qmin, qmax)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def quantize_weight(
    W: jax.Array,
    H: jax.Array,
    how: qarray.HowToQuantize,
    *,
    rho: float = 0.1,
    rho_update: float = 1.1,
    max_iter: int = 300,
    update_iter: int = 3,
    switch_iter: int = 30,
    percdamp: float = 0.01,
    mse: bool = True,
    stagnation_window: int = 4,
    stagnation_tol: float = 1e-3,
    refresh_on_stagnation: bool = True,
    refresh_accept_rel: float = 1e-3,
    pairswap: bool = True,
    seed: int = 0,
) -> tuple[qarray.QArray, jax.Array]:
  """Quantizes a weight matrix with ADMM-Q.

  Minimizes ||W @ X - W_q @ X||^2 given H = X @ X.T, like
  ``gptq_core.quantize_weight``.

  Args:
    W: weight matrix in W @ X with shape (rows, columns), where columns is the
      input/contraction dimension and rows is the output dimension.
    H: Hessian with shape (columns, columns), usually X @ X.T.
    how: how to quantize W. Must satisfy ``is_supported``.
    rho: initial ADMM penalty.
    rho_update: factor rho grows by every ``update_iter`` iterations.
    max_iter: maximum ADMM iterations.
    update_iter: iterations between error checks / rho updates.
    switch_iter: earliest iteration at which early stopping may trigger.
    percdamp: fraction of the mean diagonal of H added for damping.
    mse: run the clip-ratio search for the initial scales.
    stagnation_window: number of checks used to detect stalled progress.
    stagnation_tol: relative improvement below which progress has stalled.
    refresh_on_stagnation: re-fit the scales once when progress stalls.
    refresh_accept_rel: relative error drop needed to keep re-fitted scales.
    pairswap: run PairSwap on ill-conditioned layers (condition number of the
      normalized, damped H above 5e4).
    seed: random seed for PairSwap.

  Returns:
    A tuple of (W_q, loss): W_q is the quantized weight matrix as a QArray,
    and loss is the Hessian-weighted error sum((W_q - W) H (W_q - W)^T).
  """
  rows, columns = W.shape
  assert H.shape == (columns, columns)
  if not is_supported(how):
    raise ValueError(f'ADMM-Q does not support {how}.')
  qmin, qmax = _int_bounds(how.qtype)
  out_dtype = W.dtype
  W = W.astype(F32)
  H = H.astype(F32)

  # float64 is needed only for the eigendecomposition and (H + rho I)^-1, and
  # float32 matmuls must run at full precision (on NVIDIA GPUs JAX otherwise
  # allows TF32, which costs ADMM-Q accuracy). Scope both here rather than
  # changing them globally.
  with jax.enable_x64(True), jax.default_matmul_precision('highest'):
    XtX, X_norm, YtX, L, Q = _prepare(W, H, jnp.asarray(percdamp, F32))
    cond_num = float(L[-1]) / max(float(L[0]), 1e-12)
    ill_conditioned = cond_num > 5e4
    if ill_conditioned:
      rho = 0.01 if cond_num > 3e5 else 0.05
      rho_update = 1.05

    scale, zp = find_scale_zero(W, W, how, mse=mse)
    asym = zp is not None
    W_q, scale, zp_f, info = _admm_loop(
        W, H, XtX, X_norm, YtX, L, Q,
        scale, _zero_like_scale(zp, scale),
        jnp.asarray(rho, F64), jnp.asarray(rho_update, F64),
        jnp.asarray(stagnation_tol, F64), jnp.asarray(refresh_accept_rel, F64),
        how_key=_how_key(how), asym=asym, max_iter=max_iter,
        update_iter=update_iter, switch_iter=switch_iter, stagnation_window=stagnation_window,
        refresh_on_stagnation=refresh_on_stagnation,
    )

    if ill_conditioned and pairswap:
      batch = min(64, columns // 2)
      if batch > 0:
        W_q = _pairswap(
            W, H, _expand(scale, W.shape), _expand(zp_f, W.shape), W_q,
            jax.random.key(seed),
            qmin=qmin, qmax=qmax, n_rounds=5,
            n_batches=max(1, columns // batch), batch=batch,
        )

    loss = _recon(W_q, W, H)
    zp = zp_f.astype(zp.dtype) if asym else None

  # W_q is on the grid, so re-quantizing it recovers exactly its codes.
  qw = qarray.quantize_with_scale_zero_point(W_q, how.qtype, scale, zp)
  qw = qw.astype(out_dtype)
  logging.info(
      'ADMM-Q %dx%d: cond=%.2e, iterations=%d, converged=%s, refresh=%s/%s,'
      ' pairswap=%s, loss=%.4e',
      rows, columns, cond_num, int(info['iterations']),
      bool(info['converged']), bool(info['refreshed']),
      bool(info['refresh_accepted']), ill_conditioned and pairswap,
      float(loss),
  )
  return qw, loss


def compute_hessian(X: jax.Array) -> jax.Array:
  """The Hessian of ||W @ X - W_q @ X||^2 up to a factor 2, i.e. X @ X.T.

  This is the same statistic GPTQ collects (see gptq_core.compute_hessian).

  Args:
    X: Input data matrix of shape (in_features, n_samples).

  Returns:
    The Hessian matrix of shape (in_features, in_features).
  """
  return X @ X.T
