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

import dataclasses
from typing import Any, Mapping, cast

from absl import logging
import flax
from flax import linen as nn
from flax import nnx
import jax
from jax import numpy as jnp
from qwix._src import qconfig
from qwix._src.core import qarray
from qwix._src.providers import boxed_param
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
          ' jax.set_mesh(...) context used for the sharded model.'
      )
    return jax.sharding.NamedSharding(concrete_mesh, sharding.spec)
  return sharding


def _resolve_linen_partitioning(
    box: nn.Partitioned,
) -> tuple[jax.sharding.PartitionSpec | None, jax.sharding.Mesh]:
  """Resolves the partitioning metadata of a Linen box like Flax does.

  Args:
    box: A `nn.Partitioned` or `nn.LogicallyPartitioned` template leaf.

  Returns:
    A tuple of (spec, mesh). The spec is the mesh axes `PartitionSpec`, or
    `None` if the box names logical axes but no logical axis rules are set. The
    mesh is the box's mesh, or else the active mesh.
  """
  mesh = box.mesh if box.mesh is not None else jax.sharding.get_mesh()
  if not isinstance(box, nn.LogicallyPartitioned):
    return box.get_partition_spec(), mesh
  # Like Flax, fall back to the nn.logical_axis_rules context.
  rules = nn.get_logical_axis_rules() if box.rules is None else box.rules
  if not rules and any(isinstance(name, str) for name in box.names):
    return None, mesh
  return nn.logical_to_mesh_axes(box.names, rules), mesh


def _has_unresolved_linen_partitioning(template_param: Any) -> bool:
  """Checks if a template param has unresolved Linen partitioning metadata.

  The metadata of a box around an array without a sharding is unresolved if it
  names logical axes without logical axis rules, or mesh axes without a mesh.

  Args:
    template_param: The template param, which may contain Linen boxes.

  Returns:
    True if the partitioning metadata of any box is unresolved.
  """
  boxes = jax.tree.leaves(
      template_param, is_leaf=lambda x: isinstance(x, nn.Partitioned)
  )
  for box in boxes:
    # Arrays with a sharding are placed with it instead of the metadata.
    if (
        not isinstance(box, nn.Partitioned)
        or getattr(box.value, 'sharding', None) is not None
    ):
      continue
    spec, mesh = _resolve_linen_partitioning(box)
    if spec is None or (mesh.empty and any(axis is not None for axis in spec)):
      return True
  return False


def _sharding_from_template_metadata(
    boxed_template_value: Any,
) -> jax.sharding.NamedSharding | None:
  """Recovers a NamedSharding from a template's partitioning metadata.

  On older jax versions, `nnx.eval_shape` may drop the sharding from template
  leaves even though the boxed nnx `Variable` still carries its partitioning
  metadata. When a concrete mesh is active, reconstruct the intended
  `NamedSharding` from that metadata, mirroring `nnx.get_named_sharding`.
  Abstract Linen params never have a sharding, so it is recovered from their
  `nn.Partitioned` metadata in the same way.

  Args:
    boxed_template_value: The template leaf before `flax_util.unbox`.

  Returns:
    A concrete `NamedSharding`, or `None` when the template has no real
    partitioning metadata or no concrete mesh or logical axis rules are
    available (callers then keep the existing un-sharded behavior).
  """
  if isinstance(boxed_template_value, nnx.Variable):
    spec = nnx.spmd.get_var_pspec(boxed_template_value)
    concrete_mesh = jax.sharding.get_mesh()
  elif isinstance(boxed_template_value, nn.Partitioned):
    spec, concrete_mesh = _resolve_linen_partitioning(boxed_template_value)
  else:
    return None
  # Re-shard only when the template names a real non-replicated axis and a
  # concrete mesh is active. Empty/all-None specs keep the caller's default.
  if concrete_mesh.empty or spec is None or all(axis is None for axis in spec):
    return None
  return jax.sharding.NamedSharding(concrete_mesh, spec)


def _resolve_template_sharding(
    boxed_template_value: Any,
    path: tuple[str, ...],
) -> jax.sharding.Sharding | None:
  """Resolves the sharding for placing a checkpoint value like its template.

  Uses the sharding of the unboxed template array, resolving abstract meshes
  against the active mesh. If the template array has no sharding, falls back to
  the partitioning metadata of the boxed template leaf.

  Args:
    boxed_template_value: The template leaf before `flax_util.unbox`.
    path: The parameter path for error messages.

  Returns:
    A concrete `jax.sharding.Sharding`, or `None` if neither the template array
    nor its partitioning metadata provides one.
  """
  template_value = flax_util.unbox(boxed_template_value)
  sharding = _get_sharding(getattr(template_value, 'sharding', None), path)
  if sharding is None:
    sharding = _sharding_from_template_metadata(boxed_template_value)
  return sharding


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

  # Handle shape promotion. This must happen before device placement because
  # the template sharding is only valid for values that already have the
  # template shape.
  checkpoint_shape = jnp.shape(checkpoint_value)
  if checkpoint_shape != template_shape:
    # For scales and zero_point(allow_broadcast), broadcast 2d blocksize
    # scales/zero_points into 1d.
    if allow_broadcast:
      try:
        checkpoint_value = qarray.broadcast_to(checkpoint_value, template_shape)
      except Exception as e:
        raise ValueError(
            f'{path} has shape {checkpoint_shape}, expected {template_shape}.'
        ) from e
    # For qvalue (not allow_broadcast), it should match the template shape.
    else:
      raise ValueError(
          f'{path} has shape {checkpoint_shape}, expected {template_shape}.'
      )

  if use_checkpoint_sharding:
    sharding = _get_sharding(getattr(checkpoint_value, 'sharding', None), path)
  else:
    sharding = _resolve_template_sharding(boxed_template_value, path)

  # Handle sharding.
  if sharding is not None:
    checkpoint_value = jax.device_put(checkpoint_value, sharding)
  else:
    checkpoint_value = jnp.asarray(checkpoint_value)

  # Handle dtype promotion to match template dtype.
  if checkpoint_value.dtype != target_dtype:
    checkpoint_value = checkpoint_value.astype(target_dtype)

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


def _is_same_quantization_schema(
    checkpoint_param: Mapping[str, Any], template_param: Any
) -> bool:
  """Checks if the checkpoint and template have or can be coerced to the same quantization schema.

  Args:
    checkpoint_param: A dictionary containing quantized leaves (`qvalue`,
      `scale`, and optional `zero_point`).
    template_param: An abstract template parameter containing the target shapes
      and types for coercion.

  Returns:
    True if the checkpoint and template have or can be coerced to the same
    quantization schema, False otherwise.
  """
  if 'qvalue' not in checkpoint_param or 'scale' not in checkpoint_param:
    return False

  ckpt_qtype = checkpoint_param['qvalue'].dtype
  template_qvalue = flax_util.unbox(
      _get_template_field(template_param, 'qvalue')
  )
  if (
      isinstance(template_param, qarray.QArray)
      and template_param.qtype is not None
  ):
    template_qtype = template_param.qtype
  else:
    template_qtype = template_qvalue.dtype
  if jnp.dtype(ckpt_qtype) != jnp.dtype(template_qtype):
    return False

  # Check if the checkpoint scale can be broadcast up onto the template scale.
  # qarray.broadcast_to left-pads a lower-rank scale with 1s so ranks need not
  # match exactly but a higher-rank checkpoint scale cannot be reduced. Compare
  # dimensions from the right to match that padding.
  template_scale = flax_util.unbox(_get_template_field(template_param, 'scale'))
  template_scale_shape = template_scale.shape
  ckpt_scale_shape = checkpoint_param['scale'].shape
  if len(ckpt_scale_shape) > len(template_scale_shape):
    return False
  for src, dst in zip(
      reversed(ckpt_scale_shape), reversed(template_scale_shape)
  ):
    if src == 0 or dst % src != 0:
      return False

  ckpt_has_zero_point = checkpoint_param.get('zero_point') is not None
  template_has_zero_point = (
      _get_template_field(template_param, 'zero_point') is not None
  )
  return ckpt_has_zero_point == template_has_zero_point


def _place_quantized_leaves(
    qvalue: Any,
    scale: Any,
    zero_point: Any,
    sharding: Any,
) -> qarray.QArray:
  """Places quantized leaves onto a device sharding without any dtype/shape coercion.

  Args:
    qvalue: The quantized value array.
    scale: The scale array.
    zero_point: The optional zero point array, or None.
    sharding: The resolved sharding for the quantized value, or None.

  Returns:
    A `QArray` leaf with the given quantized leaves placed onto the given
    sharding.
  """
  if isinstance(sharding, jax.sharding.NamedSharding):
    # Handle multi-device sharding.
    qvalue = jax.device_put(qvalue, sharding)
    scale_ndim = jnp.ndim(scale)
    scale_sharding = jax.sharding.NamedSharding(
        sharding.mesh, jax.sharding.PartitionSpec(*[None] * scale_ndim)
    )
    scale = jax.device_put(scale, scale_sharding)
    if zero_point is not None:
      zp_ndim = jnp.ndim(zero_point)
      zp_sharding = jax.sharding.NamedSharding(
          sharding.mesh, jax.sharding.PartitionSpec(*[None] * zp_ndim)
      )
      zero_point = jax.device_put(zero_point, zp_sharding)
    else:
      zero_point = None
  elif sharding is not None and hasattr(sharding, 'device'):
    # Handle single-device sharding.
    qvalue = jax.device_put(qvalue, sharding)
    scale = jax.device_put(scale, sharding.device)
    if zero_point is not None:
      zero_point = jax.device_put(zero_point, sharding.device)
    else:
      zero_point = None
  else:
    qvalue = jnp.asarray(qvalue)
    scale = jnp.asarray(scale)
    if zero_point is not None:
      zero_point = jnp.asarray(zero_point)
    else:
      zero_point = None

  return qarray.QArray(
      qvalue=qvalue,
      scale=scale,
      zero_point=zero_point,
  )


def _reconstruct_how_from_template(template_param: Any) -> qarray.HowToQuantize:
  """Reconstructs a HowToQuantize from an abstract quantized template leaf.

  Args:
    template_param: An abstract template parameter containing the target shapes
      and types for coercion.

  Returns:
    A `qarray.HowToQuantize` object.
  """
  qvalue = flax_util.unbox(_get_template_field(template_param, 'qvalue'))
  scale = flax_util.unbox(_get_template_field(template_param, 'scale'))
  zero_point = _get_template_field(template_param, 'zero_point')
  if (
      isinstance(template_param, qarray.QArray)
      and template_param.qtype is not None
  ):
    qtype = template_param.qtype
  else:
    qtype = qvalue.dtype
  # Reconstruct channelwise and tiled axes from the scale shape, per
  # qarray.get_scale_shape:
  channelwise_axes = []
  tiled_axes = {}
  for axis, (qdim, sdim) in enumerate(zip(qvalue.shape, scale.shape)):
    # scale_dim == qvalue_dim -> channelwise (per-channel scale)
    if sdim == qdim:
      channelwise_axes.append(axis)
    # scale_dim == 1 -> neither (single shared scale)
    elif sdim == 1:
      continue
    # otherwise -> tiled with tile_size = qvalue_dim // scale_dim
    else:
      tiled_axes[axis] = qdim // sdim
  calibration_method = 'minmax' if zero_point is not None else 'absmax'
  return qarray.HowToQuantize(
      qtype=qtype,
      channelwise_axes=tuple(channelwise_axes),
      tiled_axes=tiled_axes,
      calibration_method=calibration_method,
  )


def _requantize_quantized_param(
    checkpoint_param: Mapping[str, Any],
    template_param: Any,
    path: tuple[str, ...],
    *,
    use_checkpoint_sharding: bool,
) -> qarray.QArray:
  """Requantizes a prequantized parameter dictionary to a QArray leaf.

  Args:
    checkpoint_param: A dictionary containing quantized leaves (`qvalue`,
      `scale`, and optional `zero_point`).
    template_param: An abstract template parameter containing the target shapes
      and types for coercion.
    path: The parameter path for error messages.
    use_checkpoint_sharding: Whether to use sharding from the checkpoint.

  Returns:
    A new `QArray` leaf whose contents have been requantized to the correct
    device placement, shape, and dtype.
  """
  _validate_prequantized_dict(checkpoint_param, path)
  ckpt_qvalue = checkpoint_param['qvalue']
  ckpt_scale = checkpoint_param['scale']
  ckpt_zero_point = checkpoint_param.get('zero_point')

  if use_checkpoint_sharding:
    sharding = _get_sharding(getattr(ckpt_qvalue, 'sharding', None), path)
  else:
    sharding = _resolve_template_sharding(
        _get_template_field(template_param, 'qvalue'), path
    )

  ckpt_qarray_leaf = _place_quantized_leaves(
      ckpt_qvalue, ckpt_scale, ckpt_zero_point, sharding
  )
  dequantized = qarray.dequantize(ckpt_qarray_leaf)
  how = _reconstruct_how_from_template(template_param)
  requantized = qarray.quantize(dequantized, how)

  scale = _apply_sharding_and_dtype(
      requantized.scale,
      _get_template_field(template_param, 'scale'),
      path,
      allow_broadcast=True,
      use_checkpoint_sharding=use_checkpoint_sharding,
  )
  template_zero_point = _get_template_field(template_param, 'zero_point')
  zero_point = requantized.zero_point
  if template_zero_point is not None and zero_point is not None:
    zero_point = _apply_sharding_and_dtype(
        zero_point,
        template_zero_point,
        path,
        allow_broadcast=True,
        use_checkpoint_sharding=use_checkpoint_sharding,
    )
  qarray_leaf = qarray.QArray(
      qvalue=requantized.qvalue,
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
    sharding = _resolve_template_sharding(template_param, path)

  ckpt_qarray_leaf = _place_quantized_leaves(
      ckpt_qvalue, ckpt_scale, ckpt_zero_point, sharding
  )
  dequantized = qarray.dequantize(ckpt_qarray_leaf)
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

  For QT and full-precision models, Qwix doesn't convert JAX arrays to WithAux
  so the template parameters don't have the 'array' suffix in their paths.

  Args:
    path: The path from the checkpoint.
    template_params: The template parameters of the NNX model or Linen params
      dict.

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


def _rebuild_linen_param(template_param: Any, processed: Any) -> Any:
  """Places a processed value into the containers of its Linen template param.

  Linen consumes params as given, so a quantized param must keep the template's
  `WithAux` and `QArray` containers along with their static fields, such as
  `WithAux.how` and `QArray.qtype`. Partitioning metadata boxes are dropped.

  Args:
    template_param: The Linen template param that owns the processed value.
    processed: A `QArray` (Case 1) or a JAX array (Case 2 and 3).

  Returns:
    The processed value in the containers of `template_param`.
  """
  if isinstance(template_param, boxed_param.WithAux):
    return dataclasses.replace(
        template_param,
        array=_rebuild_linen_param(template_param.array, processed),
    )
  if isinstance(template_param, qarray.QArray):
    return dataclasses.replace(
        template_param,
        qvalue=processed.qvalue,
        scale=processed.scale,
        zero_point=processed.zero_point,
    )
  if isinstance(template_param, dict) and isinstance(processed, qarray.QArray):
    # Manually overridden dict templates get the same dict as in NNX.
    rebuilt = {'qvalue': processed.qvalue, 'scale': processed.scale}
    if processed.zero_point is not None:
      rebuilt['zero_point'] = processed.zero_point
    return rebuilt
  return processed


def _to_linen_params(
    flat_processed: Mapping[tuple[Any, ...], Any],
    template_params: dict[str, Any],
) -> dict[str, Any]:
  """Nests processed values into a Linen params dict shaped like the template.

  Args:
    flat_processed: The processed values keyed by their resolved paths.
    template_params: The Linen params dict used as the template.

  Returns:
    A nested dict consumable by `model.apply({'params': ...}, ...)`.
  """
  flat_params = {}
  for path, processed in flat_processed.items():
    # Checkpoint paths address the array inside a WithAux, but Linen needs the
    # whole WithAux.
    if path[-1:] == ('array',) and isinstance(
        flax_util.get_value_from_path(template_params, path[:-1]),
        boxed_param.WithAux,
    ):
      path = path[:-1]
    template_param = flax_util.get_value_from_path(template_params, path)
    flat_params[path] = _rebuild_linen_param(template_param, processed)
  return flax.traverse_util.unflatten_dict(flat_params)


def process_prequantized_params(
    checkpoint_params: Mapping[str, Any],
    template_params: Any,
    *,
    allow_extra_params: bool = False,
    use_checkpoint_sharding: bool = False,
) -> Any:
  """Converts external pre-quantized params to match an NNX or Linen template.

  Args:
    checkpoint_params: A nested dict matching NNX state paths (without `.value`
      suffixes) or Linen paths (within the `'params'` collection). Leaves must
      be either a `jax.Array` or a dict containing `'qvalue'`, `'scale'`, and
      optional `'zero_point'`.
    template_params: An NNX model, possibly abstract (e.g. from
      `nnx.eval_shape`), or a Linen params dict, possibly abstract (e.g.
      `jax.eval_shape(model.init, ...)['params']`). Linen partitioning metadata
      is resolved with the box's mesh and logical axis rules, or else the active
      ones. Params whose metadata can't be resolved are placed unsharded with a
      warning.
    allow_extra_params: If True, ignore payload entries not present in
      `template_params`.
    use_checkpoint_sharding: If True, use sharding from checkpoint_params
      instead of template_params.

  Returns:
    For an NNX model, a nested pure dict consumable by `nnx.update`. The nested
    key paths will include `'value'` at the end, pointing to either a dict or a
    JAX array. For a Linen params dict, a nested dict of the processed params
    in the `WithAux` and `QArray` containers of `template_params`, consumable
    by `model.apply({'params': ...}, ...)`.
  """
  if isinstance(template_params, (dict, flax.core.FrozenDict)):
    # A Linen params dict, which is a plain dict from here on.
    template_params = flax.core.unfreeze(template_params)
  elif not isinstance(template_params, nnx.Module):
    raise TypeError(
        'process_prequantized_params expects an NNX model or a Linen params'
        f' dict as template_params. Got {type(template_params)}.'
    )

  # Value: QArray(Case 1) or a JAX array(Case 2 and 3).
  flat_processed = {}
  # Paths of Linen params that can't be sharded like the template.
  unresolved_paths = []
  # Value: dict containing 'qvalue'(quantized) or a JAX array(fp). Unfreeze so
  # that _is_leaf recognizes the quantized leaves of FrozenDict payloads.
  flat_checkpoint = flax.traverse_util.flatten_dict(
      flax.core.unfreeze(dict(checkpoint_params)), is_leaf=_is_leaf
  )
  # tuple path example: ('layers', 0, 'mlp', 'experts_gate_up_proj_weight').
  for path, checkpoint_param in flat_checkpoint.items():
    # Value: WithAux(quantized) or jax.ShapeDtypeStruct / jax.Array (fp)
    resolved_path, template_param = _resolve_template_param(
        path, template_params
    )
    if template_param is None:
      if not allow_extra_params:
        message = (
            'Found extra parameters in prequantized_params not present in'
            f' template: {resolved_path}'
        )
        if isinstance(template_params, dict) and 'params' in template_params:
          message += (
              ". template_params has a top-level 'params' key; pass the Linen"
              " params collection, e.g. variables['params']."
          )
        raise ValueError(message)
      logging.info('Skipping parameter not in template: %s', resolved_path)
      continue
    if not use_checkpoint_sharding and _has_unresolved_linen_partitioning(
        template_param
    ):
      unresolved_paths.append(resolved_path)

    # Gets the template array, which may be a dict or a QArray.
    # Qwix wraps intercepted operations in WithAux.QArray. For non-intercepted
    # operations, users will need to modify the abstract model state to match
    # their expectations. For example, Qwen 3.5 MoE overrides the original fp
    # template to be a dict like {'array': {'qvalue': ..., 'scale': ...}}.
    # Note: The consumer (e.g. Pallas kernel) must be implemented to handle
    # receiving a QArray/dict instead of a standard JAX array.
    template_array = _get_template_field(template_param, 'array')
    if template_array is not None:
      template_param = template_array

    # Case 1: checkpoint_param is prequantized (dict), template_param is
    # prequantized (dict or QArray).
    if isinstance(checkpoint_param, dict) and isinstance(
        template_param, (dict, qarray.QArray)
    ):
      # Case 1a: checkpoint_param and template_param have the same
      # quantization schema and can be coerced directly into the target
      # template schema.
      if _is_same_quantization_schema(checkpoint_param, template_param):
        processed = _process_quantized_param(
            checkpoint_param,
            template_param,
            resolved_path,
            use_checkpoint_sharding=use_checkpoint_sharding,
        )
      # Case 1b: checkpoint_param and template_param have different
      # quantization schemas and need to be requantized to match the target
      # template schema.
      else:
        processed = _requantize_quantized_param(
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

    # Handle invalid combinations (e.g. checkpoint is full precision, but
    # template expects quantized).
    else:
      raise ValueError(
          f'Unhandled or invalid parameter combination for {resolved_path}. '
          f'checkpoint_param is {type(checkpoint_param)}, '
          f'template_param is {type(template_param)}.'
      )

    flat_processed[resolved_path] = processed

  if unresolved_paths:
    logging.warning(
        'Could not shard %s because their partitioning metadata names logical'
        ' axes without logical axis rules or mesh axes without a mesh. Run'
        ' process_prequantized_params inside nn.logical_axis_rules(...) and'
        ' jax.set_mesh(...) to shard them.',
        unresolved_paths,
    )

  if isinstance(template_params, dict):
    return _to_linen_params(flat_processed, template_params)

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
      suffixes) or Linen paths (within the `'params'` collection). Leaves must
      be either a `jax.Array` or a dict containing `'qvalue'`, `'scale'`, and
      optional `'zero_point'`.
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
  # Unfreeze so that _is_leaf recognizes the quantized leaves of FrozenDict
  # payloads.
  flat_checkpoint = flax.traverse_util.flatten_dict(
      flax.core.unfreeze(dict(checkpoint_params)), is_leaf=_is_leaf
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
