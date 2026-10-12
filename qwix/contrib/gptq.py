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

"""Integration of GPTQ into Qwix.

During inference, GPTQ uses the same PtqProvider as PTQ. The only difference is
that GPTQ requires an extra calibration step to produce gptq_quant_stats, which
will then be consumed by the GPTQ's quantize_params function. After that, the
quantized params tree will look exactly the same as PTQ's.

Please check the test for an example usage.
"""

import dataclasses
from typing import Any

import jax
from qwix._src import qconfig
from qwix.contrib import calibration
from qwix.contrib import gptq_core

_STATS_SUFFIX = '_gptq'


@dataclasses.dataclass(frozen=True, kw_only=True)
class GptqRule(qconfig.QuantizationRule):
  """Use this rule to enable GPTQ."""


class GptqCalibrationProvider(calibration.SinglePassCalibrationProvider):
  """Calibration provider for GPTQ.

  This provider collects Hessian `quant_stats` information by using
  `SinglePassCalibrationProvider` to intercept compatible operations. These
  statistics are used by `quantize_params` to compute GPTQ updates. This
  provider does not perform actual quantization or use quantized operations.
  """

  def get_rule_type(self) -> type[qconfig.QuantizationRule]:
    return GptqRule

  def compute_stats(self, lhs: jax.Array) -> dict[str, Any]:
    hessian = gptq_core.compute_hessian(lhs)
    return {'hessian': hessian}

  def get_stats_suffix(self) -> str:
    return _STATS_SUFFIX


def quantize_params(
    params: Any,
    abstract_quantized_params: Any,
    gptq_quant_stats: Any,
    *,
    allow_extra_params: bool = False,
    gptq_block_size: int = 128,
    gptq_damping_factor: float = 0.01,
) -> Any:
  """Quantizes the params with GPTQ.

  Args:
    params: See ptq.quantize_params.
    abstract_quantized_params: See ptq.quantize_params.
    gptq_quant_stats: The quant_stats dict from GptqCalibrationProvider. SRQ is
      not supported yet. For params with no gptq_quant_stats, they will be
      quantized with the default PTQ algorithm.
    allow_extra_params: See ptq.quantize_params.
    gptq_block_size: The block size of GPTQ.
    gptq_damping_factor: The damping factor of GPTQ.

  Returns:
    The quantized params consumable by PtqProvider.
  """
  def _quantize(ctx: calibration.CalibratedQuantContext) -> Any:
    hessian = ctx.calibration_stats['hessian']
    assert (
        hessian.shape[0] == ctx.weight.shape[1]
        and hessian.shape[1] == ctx.weight.shape[1]
    )
    w = gptq_core.quantize_weight(
        ctx.weight,
        hessian,
        ctx.how,
        blocksize=gptq_block_size,
        percdamp=gptq_damping_factor,
    )[0]
    w = ctx.restore_shape(w)
    return ctx.abs_w.replace(array=w)

  return calibration.quantize_params_with_calibration(
      params,
      abstract_quantized_params,
      gptq_quant_stats,
      _STATS_SUFFIX,
      _quantize,
      allow_extra_params=allow_extra_params,
  )
