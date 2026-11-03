"""Integration of ADMM-Q into Qwix.

ADMM-Q plugs into Qwix exactly like GPTQ (qwix.contrib.gptq):

1. Calibration: ``AdmmqCalibrationProvider`` intercepts the model's matmuls
   for weights matched by an ``AdmmqRule`` and accumulates the Hessian
   X @ X.T into the ``quant_stats`` collection. No quantization happens here.
2. Weight transformation: ``quantize_params`` runs ADMM-Q on every weight that
   has stats, and falls back to Qwix's default PTQ for the rest (and for
   qtypes ADMM-Q does not support, e.g. fp8 or nf4).
3. Inference: the result has the same structure as PTQ's quantized params,
   so the model is served with Qwix's ``PtqProvider``.

Example (Linen; NNX works the same way as in Qwix's GPTQ docs):

  rules = [admmq.AdmmqRule(module_path='.*', weight_qtype=jnp.int4)]
  cal_model = qwix.quantize_model(model, admmq.AdmmqCalibrationProvider(rules))
  for batch in calibration_dataset:
    _, new_vars = cal_model.apply(variables, batch, mutable='quant_stats')
    variables.update(new_vars)

  ptq_model = qwix.quantize_model(model, qwix.PtqProvider(rules))
  abs_vars = jax.eval_shape(ptq_model.init, jax.random.key(0), model_input)
  params = admmq.quantize_params(
      variables['params'], abs_vars['params'], variables['quant_stats'])
  y = ptq_model.apply({'params': params}, model_input)
"""

import dataclasses
from typing import Any

from absl import logging
import jax
from qwix._src import qconfig
from qwix._src.core import qarray
from qwix.contrib import calibration

from . import admmq_core

_STATS_SUFFIX = '_admmq'


@dataclasses.dataclass(frozen=True, kw_only=True)
class AdmmqRule(qconfig.QuantizationRule):
  """Use this rule to enable ADMM-Q."""


class AdmmqCalibrationProvider(calibration.SinglePassCalibrationProvider):
  """Calibration provider for ADMM-Q.

  Collects the Hessian X @ X.T for every weight matched by an ``AdmmqRule``,
  using ``SinglePassCalibrationProvider`` to intercept compatible operations.
  The statistic is the same one GPTQ collects. This provider does not perform
  quantization or use quantized operations.
  """

  def get_rule_type(self) -> type[qconfig.QuantizationRule]:
    return AdmmqRule

  def compute_stats(self, lhs: jax.Array) -> dict[str, Any]:
    return {'hessian': admmq_core.compute_hessian(lhs)}

  def get_stats_suffix(self) -> str:
    return _STATS_SUFFIX


def quantize_params(
    params: Any,
    abstract_quantized_params: Any,
    admmq_quant_stats: Any,
    *,
    allow_extra_params: bool = False,
    admmq_damping_factor: float = 0.01,
    admmq_rho: float = 0.1,
    admmq_max_iter: int = 300,
    admmq_mse_clip: bool = True,
    admmq_pairswap: bool = True,
    admmq_seed: int = 0,
) -> Any:
  """Quantizes the params with ADMM-Q.

  Args:
    params: See ptq.quantize_params.
    abstract_quantized_params: See ptq.quantize_params.
    admmq_quant_stats: The quant_stats dict from AdmmqCalibrationProvider.
      Params with no stats are quantized with the default PTQ algorithm.
    allow_extra_params: See ptq.quantize_params.
    admmq_damping_factor: Fraction of the mean Hessian diagonal added as
      damping.
    admmq_rho: Initial ADMM penalty (lowered automatically for ill-conditioned
      layers).
    admmq_max_iter: Maximum ADMM iterations per weight.
    admmq_mse_clip: Search clipping ratios for the initial scales.
    admmq_pairswap: Run the PairSwap pass on ill-conditioned layers.
    admmq_seed: Random seed for PairSwap.

  Returns:
    The quantized params consumable by PtqProvider.
  """

  def _quantize(ctx: calibration.CalibratedQuantContext) -> Any:
    hessian = ctx.calibration_stats['hessian']
    assert hessian.shape == (ctx.weight.shape[1], ctx.weight.shape[1])
    if admmq_core.is_supported(ctx.how):
      w = admmq_core.quantize_weight(
          ctx.weight,
          hessian,
          ctx.how,
          percdamp=admmq_damping_factor,
          rho=admmq_rho,
          max_iter=admmq_max_iter,
          mse=admmq_mse_clip,
          pairswap=admmq_pairswap,
          seed=admmq_seed,
      )[0]
    else:
      logging.warning(
          'ADMM-Q does not support qtype %s for %s; using PTQ.',
          ctx.how.qtype, '/'.join(ctx.path),
      )
      w = qarray.quantize(ctx.weight, ctx.how)
    w = ctx.restore_shape(w)
    return ctx.abs_w.replace(array=w)  # pyrefly: ignore[missing-attribute]

  return calibration.quantize_params_with_calibration(
      params,
      abstract_quantized_params,
      admmq_quant_stats,
      _STATS_SUFFIX,
      _quantize,
      allow_extra_params=allow_extra_params,
  )
