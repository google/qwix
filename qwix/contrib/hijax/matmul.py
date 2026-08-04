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
"""Matmul operations."""

# pyrefly: ignore-errors

import dataclasses
import enum
import typing

import jax
import jax.experimental.hijax as hjx
import jax.numpy as jnp
from qwix.contrib.hijax import convert
from qwix.contrib.hijax import hiqarray as hq

LojaxMatmulFnTy = typing.Callable[[jax.Array, jax.Array], jax.Array]
LojaxQuantizedMatmulFnTy = typing.Callable[
    [
        jax.Array,
        jax.Array,
        jax.Array | None,
        jax.Array,
        jax.Array,
        jax.Array | None,
    ],
    jax.Array,
]


@dataclasses.dataclass(frozen=True)
class OpConfig:
  """Op configuration containing the operation and its kwargs."""

  op: LojaxMatmulFnTy | LojaxQuantizedMatmulFnTy
  kwargs: dict[str, typing.Any] = dataclasses.field(default_factory=dict)

  @staticmethod
  def null_op():
    def error_fn(*args, **kwargs):
      raise ValueError(f"Null op called with args: {args} and kwargs: {kwargs}")

    return OpConfig(
        op=error_fn,
        kwargs=dict(),
    )


# Backward pass configs
@dataclasses.dataclass(frozen=True)
class BwdNullMatmulConfig:
  """Matmul backward pass null mode for default initialization."""

  ...


@dataclasses.dataclass(frozen=True)
class BwdDequantizeMatmulConfig:
  """Matmul backward pass dequantization mode."""

  dequantize_fn: typing.Callable[..., jax.Array]
  dequantize_kwargs: dict[str, typing.Any]


@dataclasses.dataclass(frozen=True)
class BwdStaticQuantizeMatmulConfig:
  """Matmul backward pass static quantization mode."""

  quantize_fn: typing.Callable[..., jax.Array]
  quantize_kwargs: dict[str, typing.Any]
  grad_scale: jax.Array
  grad_zero_point: jax.Array | None


@dataclasses.dataclass(frozen=True)
class BwdDynamicQuantizeMatmulConfig:
  """Matmul backward pass dynamic quantization mode."""

  quantize_fn: typing.Callable[..., jax.Array]
  quantize_kwargs: dict[str, typing.Any]
  scale_zp_fn: typing.Callable[..., tuple[jax.Array, jax.Array | None]]


# Union type for the backward pass modes
BackwardPassConfig = typing.Union[
    BwdNullMatmulConfig,
    BwdDequantizeMatmulConfig,
    BwdStaticQuantizeMatmulConfig,
    BwdDynamicQuantizeMatmulConfig,
]


@dataclasses.dataclass(frozen=True)
class MatmulFwdBwdConfig:
  """Matmul forward and backward pass config."""

  # matmul parameters
  fwd_config: OpConfig

  # dlhs parameters
  dlhs_mode: BackwardPassConfig = BwdNullMatmulConfig()
  dlhs_config: OpConfig = OpConfig.null_op()

  # drhs parameters
  drhs_mode: BackwardPassConfig = BwdNullMatmulConfig()
  drhs_config: OpConfig = OpConfig.null_op()


class _WhichOp(enum.Enum):
  """Enum to decide which op to use in the matmul primitive."""

  FWD = "fwd"
  DLHS = "dlhs"
  DRHS = "drhs"


class Matmul(hjx.VJPHiPrimitive):
  """Hijax matmul primitive for HiQArray.

  This class implements the hijax primitive for a matmul operation. It supports
  multiple types of backward passes, including dequantization, static
  quantization, and dynamic quantization. The user can also specify the
  forward pass config, including the operation and kwargs.

  This class wraps the user-provided functions and configurations to perform the
  forward and backward pass for a matmul operation. The forward pass is
  performed using the user-provided forward pass config. The backward pass
  is performed using the user-provided backward pass config, which can be
  different from the forward pass config. Note that the user is responsible for
  defining the forward and backward pass configs appropriately for their use
  case.


  Attributes:
    in_avals: The input avals of the HiQArrays.
    out_aval: The output aval of the HiQArray.
    params: The parameters of the matmul operation.
    config: The MatmulFwdBwdConfig to use for the matmul operation.
    _which_op: The current op mode of the matmul operation.
  """

  def __init__(
      self,
      in_avals: tuple[hq.HiQArrayTy, ...],
      *,
      config: MatmulFwdBwdConfig,
  ):
    self.in_avals = in_avals

    # compute out_aval
    out_sds = jax.eval_shape(
        jnp.matmul,
        in_avals[0].array_ty,
        in_avals[1].array_ty,
        **config.fwd_config.kwargs,
    )
    self.out_aval = jax._src.core._sds_aval_mapping(out_sds)
    self.params = dict(config=config)
    # For pytype warnings
    self.config = config

    # Stateful parameter to determine which op to use.
    self._which_op = _WhichOp.FWD
    super().__init__()

  def expand(self, lhs: hq.HiQArray, rhs: hq.HiQArray):
    # Select the config based on the current op path
    if self._which_op == _WhichOp.FWD:
      config = self.config.fwd_config
    elif self._which_op == _WhichOp.DLHS:
      config = self.config.dlhs_config
    elif self._which_op == _WhichOp.DRHS:
      config = self.config.drhs_config
    else:
      raise ValueError(f"Unsupported op mode: {self._which_op}")

    if config is None:
      raise ValueError(
          f"Expected config for op mode {self._which_op}, but got None."
      )

    op = config.op
    kwargs = config.kwargs

    return op(
        lhs.qvalue,
        lhs.scale,
        lhs.zero_point,
        rhs.qvalue,
        rhs.scale,
        rhs.zero_point,
        **kwargs,
    )

  # Reverse mode ad
  def vjp_fwd(self, nzs_in, lhs: hq.HiQArray, rhs: hq.HiQArray):
    return self(lhs, rhs), (lhs, rhs)

  def vjp_bwd_retval(self, res, g, /):
    lhs, rhs = res

    # Dlhs
    self._which_op = _WhichOp.DLHS
    dlhs_mode = self.config.dlhs_mode
    rhs_t = convert.transpose(rhs)
    if isinstance(dlhs_mode, BwdNullMatmulConfig):
      raise ValueError(
          f"Dlhs mode {dlhs_mode} not supported. Please specify a valid"
          " backward pass config."
      )
    if isinstance(dlhs_mode, BwdDequantizeMatmulConfig):
      # Dequantize rhs
      y = convert.from_hiqarray(
          rhs_t,
          **dataclasses.asdict(dlhs_mode),
      )
      # Compute dlhs
      dlhs = self.config.dlhs_config.op(g, y, **self.config.dlhs_config.kwargs)
    else:
      # Compute or fetch scale and zero point
      if isinstance(dlhs_mode, BwdStaticQuantizeMatmulConfig):
        scale = dlhs_mode.grad_scale
        zp = dlhs_mode.grad_zero_point
      elif isinstance(dlhs_mode, BwdDynamicQuantizeMatmulConfig):
        scale, zp = dlhs_mode.scale_zp_fn(g)
      else:
        raise ValueError(
            f"Unsupported dlhs mode for backward pass: {dlhs_mode}"
        )
      # Quantize g
      gq = convert.to_hiqarray(
          g,
          scale=scale,
          zero_point=zp,
          quantize_fn=dlhs_mode.quantize_fn,
          **dlhs_mode.quantize_kwargs,
      )
      # Compute dlhs
      dlhs = self(gq, rhs_t)

    # Drhs
    self._which_op = _WhichOp.DRHS
    drhs_mode = self.config.drhs_mode
    lhs_t = convert.transpose(lhs)
    if isinstance(drhs_mode, BwdNullMatmulConfig):
      raise ValueError(
          f"Drhs mode {drhs_mode} not supported. Please specify a valid"
          " backward pass config."
      )
    if isinstance(drhs_mode, BwdDequantizeMatmulConfig):
      # Dequantize lhs
      lhs_dq = convert.from_hiqarray(
          lhs_t,
          **dataclasses.asdict(drhs_mode),
      )
      # Compute drhs
      drhs = self.config.drhs_config.op(
          lhs_dq, g, **self.config.drhs_config.kwargs
      )
    else:
      # Compute or fetch scale and zero point
      if isinstance(drhs_mode, BwdStaticQuantizeMatmulConfig):
        scale = drhs_mode.grad_scale
        zp = drhs_mode.grad_zero_point
      elif isinstance(drhs_mode, BwdDynamicQuantizeMatmulConfig):
        scale, zp = drhs_mode.scale_zp_fn(g)
      else:
        raise ValueError(
            f"Unsupported drhs mode for backward pass: {drhs_mode}"
        )
      # Quantize g
      gq = convert.to_hiqarray(
          g,
          scale=scale,
          zero_point=zp,
          quantize_fn=drhs_mode.quantize_fn,
          **drhs_mode.quantize_kwargs,
      )
      # Compute drhs
      drhs = self(lhs_t, gq)

    # Reset back to forward mode
    self._which_op = _WhichOp.FWD
    return (dlhs, drhs)


def matmul(
    a: hq.HiQArray,
    b: hq.HiQArray,
    *,
    config: MatmulFwdBwdConfig,
) -> jax.Array:
  """Matmul operation for HiQArray.

  Args:
    a: The left-hand side HiQArray.
    b: The right-hand side HiQArray.
    config: The MatmulFwdBwdConfig to use for the matmul operation.

  Returns:
    The result of the matmul operation.
  """
  in_avals = (jax.typeof(a), jax.typeof(b))
  matmul_instance = Matmul(in_avals, config=config)
  return matmul_instance(a, b)
