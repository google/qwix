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
"""Utilities for handling checkpoints and prequantized parameters."""

from typing import Any, Mapping, cast

from absl import logging
import flax
from flax import nnx
import jax
from jax import numpy as jnp
from qwix._src import qconfig
from qwix._src.core import qarray
from qwix._src.providers import qt
from qwix._src.utils import flax_util

_PREQUANTIZED_ARRAY_LEAF_NAMES = frozenset(('qvalue', 'scale', 'zero_point'))


def _is_leaf(path, x):
  """Checks if x is a pre-quantized leaf.

  Stop at dicts that contain the 'qvalue' key. This identifies them
  as pre-quantized weight payloads.

  Args:
    path: The parameter path. Unused.
    x: The value to check.

  Returns:
    True if x is a pre-quantized leaf, False otherwise.
  """
  del path
  return isinstance(x, dict) and 'qvalue' in x


def _get_template_field(obj: Any, field_name: str) -> Any:
  """Gets a field from the template parameter."""
  if isinstance(obj, dict):
    return obj.get(field_name)
  return getattr(obj, field_name, None)


def _validate_prequantized_dict(
    checkpoint_param: Any,
    path: tuple[str, ...],
) -> None:
  """Validates the flat quantized parameter dictionary format.

  Checks that all the keys in the input dictionary are in
  _PREQUANTIZED_ARRAY_LEAF_NAMES.

  Args:
    checkpoint_param: A dict containing quantized leaves (`qvalue`, `scale`, and
      optional `zero_point`).
    path: The parameter path for error messages.
  """
  if not isinstance(checkpoint_param, dict):
    raise ValueError(
        f'{path} is quantized in the template_params. Expected a'
        ' dict containing qvalue/scale leaves.'
    )

  unsupported_keys = set(checkpoint_param) - _PREQUANTIZED_ARRAY_LEAF_NAMES
  if unsupported_keys:
    raise ValueError(
        f'{path} has unsupported quantized leaves'
        f' {sorted(unsupported_keys)}. Expected only'
        f' {sorted(_PREQUANTIZED_ARRAY_LEAF_NAMES)}.'
    )

  if 'qvalue' not in checkpoint_param or 'scale' not in checkpoint_param:
    param_summary = {
        k: (type(v), getattr(v, 'shape', None))
        for k, v in checkpoint_param.items()
    }
    raise ValueError(
        f'{path} is missing required "qvalue" or "scale" in pre-quantized'
        f' dictionary. Found keys: {list(checkpoint_param.keys())}, summary:'
        f' {param_summary}'
    )


def _get_sharding(
    sharding: Any,
    path: tuple[str, ...],
) -> jax.sharding.Sharding | None:
  """Resolves abstract mesh shardings into concrete device shardings.

  Args:
    sharding: The abstract or concrete sharding definition. If it is a
      NamedSharding with an AbstractMesh, it will be resolved to use the active
      mesh.
    path: The parameter path for error messages.

  Returns:
    A concrete `jax.sharding.Sharding` object, or None if the input was None.
  """
  if sharding is None:
    return None
  if isinstance(sharding, jax.sharding.NamedSharding) and isinstance(
      sharding.mesh, jax.sharding.AbstractMesh
  ):
    concrete_mesh = jax.sharding.get_mesh()
    if concrete_mesh.empty:
      raise ValueError(
          f'{path} requires an active mesh to place pre-quantized'
          ' arrays. Run process_prequantized_params inside the same'
          ' jax.set_mesh(...) context used for the sharded PTQ model.'
      )
    return jax.sharding.NamedSharding(concrete_mesh, sharding.spec)
  return sharding


def _sharding_from_template_metadata(
    boxed_template_value: Any,
) -> jax.sharding.NamedSharding | None:
  """Recovers a NamedSharding from an nnx template's partitioning metadata.

  On older jax versions, `nnx.eval_shape` may drop the sharding from template
  leaves even though the boxed nnx `Variable` still carries its partitioning
  metadata. When a concrete mesh is active, reconstruct the intended
  `NamedSharding` from that metadata, mirroring `nnx.get_named_sharding`.

  Args:
    boxed_template_value: The template leaf before `flax_util.unbox`.

  Returns:
    A concrete `NamedSharding`, or `None` when the template has no real
    partitioning metadata or no concrete mesh is active (callers then keep the
    existing un-sharded behavior).
  """
  if not isinstance(boxed_template_value, nnx.Variable):
    return None
  spec = nnx.spmd.get_var_pspec(boxed_template_value)
  concrete_mesh = jax.sharding.get_mesh()
  # Re-shard only when the template names a real non-replicated axis and a
  # concrete mesh is active. Empty/all-None specs keep the caller's default.
  if concrete_mesh.empty or spec is None or all(axis is None for axis in spec):
    return None
  return jax.sharding.NamedSharding(concrete_mesh, spec)


def _apply_sharding_and_dtype(
    checkpoint_value: Any,
    template_value: Any,
    path: tuple[str, ...],
    allow_broadcast: bool = False,
    use_checkpoint_sharding: bool = False,
) -> jax.Array:
  """Converts a host/device array-like value into the template's array shape."""
  boxed_template_value = template_value
  template_value = flax_util.unbox(template_value)
  if not isinstance(template_value, (jax.Array, jax.ShapeDtypeStruct)):
    raise TypeError(
        f'{path} does not resolve to a JAX array or ShapeDtypeStruct in the'
        ' template_params.'
    )

  target_dtype = template_value.dtype
  template_shape = template_value.shape
  if use_checkpoint_sharding:
    sharding = _get_sharding(getattr(checkpoint_value, 'sharding', None), path)
  else:
    sharding = _get_sharding(getattr(template_value, 'sharding', None), path)
    if sharding is None:
      sharding = _sharding_from_template_metadata(boxed_template_value)

  # Handle sharding.
  if sharding is not None:
    checkpoint_value = jax.device_put(checkpoint_value, sharding)
  else:
    checkpoint_value = jnp.asarray(checkpoint_value)

  # Handle dtype promotion to match template dtype.
  if checkpoint_value.dtype != target_dtype:
    checkpoint_value = checkpoint_value.astype(target_dtype)

  # Handle shape promotion.
  if checkpoint_value.shape != template_shape:
    # For scales and zero_point(allow_broadcast), broadcast 2d blocksize
    # scales/zero_points into 1d.
    if allow_broadcast:
      try:
        checkpoint_value = qarray.broadcast_to(checkpoint_value, template_shape)
      except Exception as e:
        raise ValueError(
            f'{path} has shape {checkpoint_value.shape}, expected'
            f' {template_shape}.'
        ) from e
    # For qvalue (not allow_broadcast), it should match the template shape.
    else:
      raise ValueError(
          f'{path} has shape {checkpoint_value.shape}, expected'
          f' {template_shape}.'
      )

  return checkpoint_value


def _process_quantized_param(
    checkpoint_param: Mapping[str, Any],
    template_param: Any,
    path: tuple[str, ...],
    *,
    use_checkpoint_sharding: bool,
) -> qarray.QArray:
  """Builds a QArray leaf from a quantized parameter dictionary and a quantized template.

  Args:
    checkpoint_param: A dictionary containing quantized leaves (`qvalue`,
      `scale`, and optional `zero_point`).
    template_param: An abstract template parameter containing the target shapes
      and types for coercion.
    path: The parameter path for error messages.
    use_checkpoint_sharding: Whether to use sharding from the checkpoint.

  Returns:
    A new `QArray` leaf whose contents have been coerced to the correct
    device placement, shape, and dtype.
  """
  _validate_prequantized_dict(checkpoint_param, path)

  qvalue = _apply_sharding_and_dtype(
      checkpoint_param['qvalue'],
      _get_template_field(template_param, 'qvalue'),
      path,
      use_checkpoint_sharding=use_checkpoint_sharding,
  )
  scale = _apply_sharding_and_dtype(
      checkpoint_param['scale'],
      _get_template_field(template_param, 'scale'),
      path,
      allow_broadcast=True,
      use_checkpoint_sharding=use_checkpoint_sharding,
  )

  template_zero_point = _get_template_field(template_param, 'zero_point')
  zero_point = checkpoint_param.get('zero_point')
  if template_zero_point is None and zero_point is not None:
    raise ValueError(
        f'{path} provided an unexpected "zero_point" for a symmetric'
        ' quantized param.'
    )
  if template_zero_point is not None and zero_point is None:
    raise ValueError(f'{path} is missing required quantized leaf "zero_point".')
  if template_zero_point is not None:
    zero_point = _apply_sharding_and_dtype(
        zero_point,
        template_zero_point,
        path,
        allow_broadcast=True,
        use_checkpoint_sharding=use_checkpoint_sharding,
    )

  qarray_leaf = qarray.QArray(
      qvalue=qvalue,
      scale=scale,
      zero_point=zero_point,
  )
  qarray.validate_qarray(qarray_leaf)
  return qarray_leaf


def _dequantize_quantized_param(
    checkpoint_param: Mapping[str, Any],
    template_param: Any,
    path: tuple[str, ...],
    *,
    use_checkpoint_sharding: bool,
) -> jax.Array:
  """Dequantizes a prequantized parameter dictionary to a JAX array.

  Args:
    checkpoint_param: A dictionary containing quantized leaves (`qvalue`,
      `scale`, and optional `zero_point`).
    template_param: An abstract template parameter containing the target shapes
      and types for coercion.
    path: The parameter path for error messages.
    use_checkpoint_sharding: Whether to use sharding from the checkpoint.

  Returns:
    A JAX array whose contents have been dequantized and coerced to the
    correct device placement, shape, and dtype.
  """
  _validate_prequantized_dict(checkpoint_param, path)
  ckpt_qvalue = checkpoint_param['qvalue']
  ckpt_scale = checkpoint_param['scale']
  ckpt_zero_point = checkpoint_param.get('zero_point')

  if use_checkpoint_sharding:
    sharding = _get_sharding(getattr(ckpt_qvalue, 'sharding', None), path)
  else:
    sharding = _get_sharding(getattr(template_param, 'sharding', None), path)
  if isinstance(sharding, jax.sharding.NamedSharding):
    # Handle multi-device sharding.
    qvalue = jax.device_put(ckpt_qvalue, sharding)
    scale_ndim = jnp.ndim(ckpt_scale)
    scale_sharding = jax.sharding.NamedSharding(
        sharding.mesh, jax.sharding.PartitionSpec(*[None] * scale_ndim)
    )
    scale = jax.device_put(ckpt_scale, scale_sharding)
    if ckpt_zero_point is not None:
      zp_ndim = jnp.ndim(ckpt_zero_point)
      zp_sharding = jax.sharding.NamedSharding(
          sharding.mesh, jax.sharding.PartitionSpec(*[None] * zp_ndim)
      )
      zero_point = jax.device_put(ckpt_zero_point, zp_sharding)
    else:
      zero_point = None
  elif sharding is not None and hasattr(sharding, 'device'):
    # Handle single-device sharding.
    qvalue = jax.device_put(ckpt_qvalue, sharding)
    scale = jax.device_put(ckpt_scale, sharding.device)
    if ckpt_zero_point is not None:
      zero_point = jax.device_put(ckpt_zero_point, sharding.device)
    else:
      zero_point = None
  else:
    qvalue = jnp.asarray(ckpt_qvalue)
    scale = jnp.asarray(ckpt_scale)
    if ckpt_zero_point is not None:
      zero_point = jnp.asarray(ckpt_zero_point)
    else:
      zero_point = None

  qarray_leaf = qarray.QArray(
      qvalue=qvalue,
      scale=scale,
      zero_point=zero_point,
  )
  dequantized = qarray.dequantize(qarray_leaf)
  return _apply_sharding_and_dtype(
      dequantized,
      template_param,
      path,
      use_checkpoint_sharding=use_checkpoint_sharding,
  )


def _resolve_template_param(
    path: tuple[str, ...],
    template_params: Any,
) -> tuple[tuple[str, ...], Any]:
  """Resolves the template parameter and its path.

  For QT models, Qwix doesn't convert JAX arrays to WithAux so the template
  parameters don't have the 'array' suffix in their paths.

  Args:
    path: The path from the checkpoint.
    template_params: The template parameters of the NNX PTQ/QT model.

  Returns:
    A tuple of (resolved_path, template_param).
  """
  template_param = flax_util.get_value_from_path(template_params, path)
  if template_param is not None or not path:
    return path, template_param

  *_, last_key = path
  if last_key == 'array':
    parent_path = path[:-1]
    parent_param = flax_util.get_value_from_path(template_params, parent_path)
    unboxed_parent = flax_util.unbox(parent_param)
    if isinstance(unboxed_parent, (jax.Array, jax.ShapeDtypeStruct)):
      return parent_path, parent_param

  return path, template_param


def process_prequantized_params(
    checkpoint_params: Mapping[str, Any],
    template_params: Any,
    *,
    allow_extra_params: bool = False,
    use_checkpoint_sharding: bool = False,
) -> Any:
  """Converts external pre-quantized params into an `nnx.update`-friendly pure dict.

  Args:
    checkpoint_params: A nested dict matching NNX state paths (without `.value`
      suffixes). Leaves must be either a `jax.Array` or a dict containing
      `'qvalue'`, `'scale'`, and optional `'zero_point'`.
    template_params: An NNX PTQ/QT model, possibly abstract (e.g., from
      `nnx.eval_shape`).
    allow_extra_params: If True, ignore payload entries not present in
      `template_params`.
    use_checkpoint_sharding: If True, use sharding from checkpoint_params
      instead of template_params.

  Returns:
    A nested pure dict consumable by `nnx.update`. The nested key paths will
    include `'value'` at the end, pointing to either a dict or a JAX array.
  """
  if not isinstance(template_params, nnx.Module):
    raise TypeError(
        'process_prequantized_params only supports NNX PTQ/QT models. Got'
        f' {type(template_params)}.'
    )

  # Value: QArray(Case 1) or a JAX array(Case 2 and 3).
  flat_processed = {}
  # Value: dict containing 'qvalue'(quantized) or a JAX array(fp).
  flat_checkpoint = flax.traverse_util.flatten_dict(
      checkpoint_params, is_leaf=_is_leaf
  )
  # tuple path example: ('layers', 0, 'mlp', 'experts_gate_up_proj_weight').
  for path, checkpoint_param in flat_checkpoint.items():
    # Value: WithAux(quantized) or jax.ShapeDtypeStruct / jax.Array (fp)
    resolved_path, template_param = _resolve_template_param(
        path, template_params
    )
    if template_param is None:
      if not allow_extra_params:
        raise ValueError(
            'Found extra parameters in prequantized_params not present in'
            f' template: {resolved_path}'
        )
      logging.info('Skipping parameter not in template: %s', resolved_path)
      continue

    # Gets the template array, which may be a dict or a QArray.
    # Qwix wraps intercepted operations in WithAux.QArray. For non-intercepted
    # operations, users will need to modify the abstract model state to match
    # their expectations. For example, Qwen 3.5 MoE overrides the original fp
    # template to be a dict like {'array': {'qvalue': ..., 'scale': ...}}.
    # Note: The consumer (e.g., Pallas kernel) must be implemented to handle
    # receiving a QArray/dict instead of a standard JAX array.
    template_array = _get_template_field(template_param, 'array')
    if template_array is not None:
      template_param = template_array

    # Case 1: checkpoint_param is prequantized (dict), template_param is
    # prequantized (dict or QArray).
    if isinstance(checkpoint_param, dict) and isinstance(
        template_param, (dict, qarray.QArray)
    ):
      processed = _process_quantized_param(
          checkpoint_param,
          template_param,
          resolved_path,
          use_checkpoint_sharding=use_checkpoint_sharding,
      )

    # Case 2: checkpoint_param is prequantized (dict), template_param is fp.
    elif isinstance(checkpoint_param, dict) and not isinstance(
        template_param, (dict, qarray.QArray)
    ):
      processed = _dequantize_quantized_param(
          checkpoint_param,
          template_param,
          resolved_path,
          use_checkpoint_sharding=use_checkpoint_sharding,
      )

    # Case 3: checkpoint_param is fp, template_param is fp.
    elif not isinstance(checkpoint_param, dict) and not isinstance(
        template_param, (dict, qarray.QArray)
    ):
      processed = _apply_sharding_and_dtype(
          checkpoint_param,
          template_param,
          resolved_path,
          use_checkpoint_sharding=use_checkpoint_sharding,
      )

    # Handle invalid combinations (e.g., checkpoint is full precision, but
    # template expects quantized).
    else:
      raise ValueError(
          f'Unhandled or invalid parameter combination for {resolved_path}. '
          f'checkpoint_param is {type(checkpoint_param)}, '
          f'template_param is {type(template_param)}.'
      )

    flat_processed[resolved_path] = processed

  # Key: nested key path.
  # Value: QArray(Case 1) or a JAX array(Case 2 and 3).
  nested_processed = flax.traverse_util.unflatten_dict(flat_processed)
  # Key: nested key path and ['value'].
  # Value: Dict(Case 1) or a JAX array(Case 2 and 3).
  #   - Case 1: dict {"qvalue": ..., "scale": ...}
  #   - Case 2: <jax.Array>
  #   - Case 3: <jax.Array>
  return nnx.to_pure_dict(nnx.state(nested_processed))


_DEFAULT_ACT_QTYPE = object()


def restore_quantization_rules(
    checkpoint_params: Mapping[str, Any],
    rule_type: type[qconfig.QuantizationRule] | type[qt.QtRule],
    *,
    tile_size: int,
    act_qtype: jax.typing.DTypeLike | None | object = _DEFAULT_ACT_QTYPE,
    **kwargs,
) -> list[qconfig.QuantizationRule | qt.QtRule]:
  """Restores quantization rules from pre-quantized checkpoint.

  Args:
    checkpoint_params: A nested dict matching NNX state paths (without `.value`
      suffixes). Leaves must be either a `jax.Array` or a dict containing
      `'qvalue'`, `'scale'`, and optional `'zero_point'`.
    rule_type: The type of quantization rule, either `qconfig.QuantizationRule`
      or `qt.QtRule`.
    tile_size: The tile size for subchannel quantization.
    act_qtype: The quantized type for activations. If not specified, it defaults
      to the quantized type for weights inferred from the checkpoint.
    **kwargs: Additional keyword arguments for the quantization rule.

  Returns:
    A list of `qconfig.QuantizationRule | qt.QtRule` objects.
  """
  rules = {}
  flat_checkpoint = flax.traverse_util.flatten_dict(
      checkpoint_params, is_leaf=_is_leaf
  )
  for path, checkpoint_param in flat_checkpoint.items():
    # Skip non-quantized parameters.
    if (
        not isinstance(checkpoint_param, dict)
        or 'qvalue' not in checkpoint_param
    ):
      continue

    # Remove optional 'array' suffix and parameter name from the path, and
    # replace numeric indices with wildcards.
    module_path_tuple = path
    if module_path_tuple and module_path_tuple[-1] == 'array':
      module_path_tuple = module_path_tuple[:-1]
    if module_path_tuple:
      module_path_tuple = module_path_tuple[:-1]
    module_path_parts = []
    for part in module_path_tuple:
      if isinstance(part, int):
        module_path_parts.append('[^/]+')
      else:
        module_path_parts.append(str(part))
    module_path = '/'.join(module_path_parts)

    # Infer weight quantized type and calibration method.
    qvalue = checkpoint_param['qvalue']
    weight_qtype = qvalue.dtype
    if act_qtype is _DEFAULT_ACT_QTYPE:
      resolved_act_qtype = weight_qtype
    else:
      resolved_act_qtype = cast(jax.typing.DTypeLike | None, act_qtype)
    zero_point = checkpoint_param.get('zero_point')
    if zero_point is not None:
      weight_calibration_method = 'minmax'
    else:
      weight_calibration_method = 'absmax'

    # Generate quantization rule.
    rule = rule_type(
        module_path=module_path,
        weight_qtype=weight_qtype,
        act_qtype=resolved_act_qtype,
        weight_calibration_method=weight_calibration_method,
        tile_size=tile_size,
        **kwargs,
    )
    if module_path in rules:
      if rules[module_path] != rule:
        logging.warning(
            'Conflicting quantization rules reconstructed for %s. Existing:'
            ' %s, New: %s. The existing rule will be overwritten.',
            module_path,
            rules[module_path],
            rule,
        )
    rules[module_path] = rule
  return list(rules.values())
