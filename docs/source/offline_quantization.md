# Offline Quantization

Qwix's Offline Quantization feature enables loading pre-quantized checkpoints
and continuing training or inference workloads in the desired target numeric.

*   **Quantized Training/Inference**: Keep weights in the original quantization
    schema or requantize weights to another quantization schema for quantized
    training or optimized serving.
*   **Full-Precision Training/Inference**: Dequantize weights back to
    full-precision (e.g. BF16) for continued training or full-precision
    inference.

Both Flax Linen and NNX models are supported. The examples below use NNX; see
[Flax Linen Models](offline_quantization.md#offline-quantization-linen) for the Linen differences.

## Overview

### Quantized Training/Inference

1.  **Load and Transform Checkpoint**: Load and transform your pre-quantized
    checkpoint into the expected Qwix structure with Orbax's model surgery
    transformation utilities.
2.  **Generate Abstract Model**: Initialize your model with
    `qwix.quantize_model` and `nnx.eval_shape`. Note that opaque layers will
    remain as full-precision `nnx.Param` objects.
3.  **Override Opaque Layers**: If you have pre-quantized weights inside opaque
    layers, replace the `nnx.Param` nodes with nested dictionaries containing
    `nnx.Param` objects for `qvalue`, `scale`, and optionally `zero_point`,
    tailored to your kernel's specific block quantization shapes and numerics.
4.  **Process and Update State**: Process the loaded parameters with
    `qwix.process_prequantized_params` and update model state with `nnx.update`.
5.  **Ready for continued training or inference!**

```py
from flax import nnx
import jax
from orbax.checkpoint.experimental import v1 as ocp
import qwix

checkpoint_dir = '/path/to/checkpoint'
mesh = create_mesh()

# Step 1: Load and Transform Checkpoint
with ocp.Context(checkpoint_layout=ocp.options.CheckpointLayout.SAFETENSORS):
  meta = ocp.metadata(checkpoint_dir)
  sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
  flat_abstract = {
      name: jax.ShapeDtypeStruct(
          shape=m.shape, dtype=m.dtype, sharding=sharding
      )
      for name, m in meta.metadata.items()
  }
  restored_tree = ocp.load(checkpoint_dir, abstract_state=flat_abstract)

# See below section on how to implement this helper.
restored_tree = transform_checkpoint_tree(restored_tree)

# Step 2: Generate Abstract Model
quantization_rules = qwix.restore_quantization_rules(
    restored_tree,
    qwix.QuantizationRule,
    tile_size=64,
)
# If applicable, replace quantization rules as needed for the target
# quantization schema.


def build_model():
  model = MyModel()
  ptq_provider = qwix.PtqProvider(quantization_rules)
  dummy_input = get_dummy_input()
  return qwix.quantize_model(model, ptq_provider, dummy_input)


with jax.set_mesh(mesh):
  abstract_model = nnx.eval_shape(build_model)

  # Step 3: Override Opaque Layers
  # If applicable, see below section on how to implement this helper.
  override_opaque_layers(abstract_model, mesh)

  # Step 4: Process and Update State
  processed_params = qwix.process_prequantized_params(
      restored_tree, abstract_model
  )
  nnx.update(abstract_model, processed_params)

# Step 5: Ready for continued training or inference!
```

### Full-Precision Training/Inference

```{note}
The full-precision path omits the `restore_quantization_rules` and
`override_opaque_layers` steps which are only needed to initialize a quantized
abstract model with intercepted and opaque weights.
```

1.  **Load and Transform Checkpoint**: Load and transform your pre-quantized
    checkpoint into the expected Qwix structure with Orbax's model surgery
    transformation utilities.
2.  **Generate Abstract Model**: Initialize your model with `nnx.eval_shape`.
3.  **Process and Update State**: Process the loaded parameters with
    `qwix.process_prequantized_params` and update model state with `nnx.update`.
4.  **Ready for continued training or inference!**

```py
from flax import nnx
import jax
from orbax.checkpoint.experimental import v1 as ocp
import qwix

checkpoint_dir = '/path/to/checkpoint'
mesh = create_mesh()

# Step 1: Load and Transform Checkpoint
with ocp.Context(checkpoint_layout=ocp.options.CheckpointLayout.SAFETENSORS):
  meta = ocp.metadata(checkpoint_dir)
  sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
  flat_abstract = {
      name: jax.ShapeDtypeStruct(
          shape=m.shape, dtype=m.dtype, sharding=sharding
      )
      for name, m in meta.metadata.items()
  }
  restored_tree = ocp.load(checkpoint_dir, abstract_state=flat_abstract)

# See below section on how to implement this helper.
restored_tree = transform_checkpoint_tree(restored_tree)

# Step 2: Generate Abstract Model
with jax.set_mesh(mesh):
  abstract_model = nnx.eval_shape(lambda: MyModel())

  # Step 3: Process and Update State
  processed_params = qwix.process_prequantized_params(
      restored_tree, abstract_model
  )
  nnx.update(abstract_model, processed_params)

# Step 4: Ready for continued training or inference!
```

(offline-quantization-linen)=

## Flax Linen Models

The Linen flows mirror the NNX flows above, with the following differences:

*   **Generate Abstract Params**: Call `qwix.quantize_model` without model
    inputs, then generate abstract params with `jax.eval_shape(model.init,
    ...)['params']`.
*   **Override Opaque Layers**: Replace the opaque layer leaves in the abstract
    params dict. See
    [Overriding Custom Parameters in Opaque Layers](offline_quantization.md#offline-quantization-opaque-layers).
*   **Process Params**: Pass the abstract `'params'` collection, not the whole
    variables dict, to `qwix.process_prequantized_params`, then run the model
    with `model.apply({'params': params}, ...)` instead of calling `nnx.update`.

For the full-precision flow, skip `restore_quantization_rules`,
`qwix.quantize_model`, and the opaque layer overrides.

```py
from flax import linen as nn
import jax
import qwix

# Step 1: Load and Transform Checkpoint to Linen paths (as above).
restored_tree = transform_checkpoint_tree(restored_tree)

# Step 2: Generate Abstract Params
quantization_rules = qwix.restore_quantization_rules(
    restored_tree,
    qwix.QuantizationRule,
    tile_size=64,
)
# Linen models are quantized without model inputs.
model = qwix.quantize_model(MyModel(), qwix.PtqProvider(quantization_rules))

# nn.logical_axis_rules is only needed for nn.LogicallyPartitioned params.
with jax.set_mesh(mesh), nn.logical_axis_rules(logical_axis_rules):
  abstract_params = jax.eval_shape(
      model.init, jax.random.key(0), get_dummy_input()
  )['params']

  # Step 3: Override Opaque Layers
  # If applicable, see below section on how to implement this helper.
  override_opaque_layers(abstract_params)

  # Step 4: Process Params
  params = qwix.process_prequantized_params(restored_tree, abstract_params)

# Step 5: Ready for continued training or inference!
outputs = model.apply({'params': params}, inputs)
```

### Output Params

For Linen, `qwix.process_prequantized_params` returns a nested dict of only the
checkpoint's params, placed on device inside the template's `WithAux` and
`QArray` containers with partitioning metadata boxes removed. Template params
missing from the checkpoint aren't included, so merge those in (e.g. from
`model.init`) before calling `model.apply`.

(offline-quantization-linen-sharding)=

### Sharding

Linen abstract params carry no sharding, so Qwix shards each param from its
`nn.Partitioned` or `nn.LogicallyPartitioned` metadata like Flax does: with the
box's mesh or else the mesh set by `jax.set_mesh`, and with the box's rules or
else the rules set by `nn.logical_axis_rules`. If the metadata can't be resolved
this way, Qwix logs a warning and places the param unsharded.

```{note}
The deprecated `with mesh:` context manager doesn't set the mesh read by Qwix.
Use `with jax.set_mesh(mesh):` instead.
```

## Transforming Checkpoints to Qwix Structure

When loading checkpoints from external formats like Hugging Face SafeTensors,
the checkpoint must be transformed to match the expected Qwix structure.

Orbax provides standardized model surgery transformation utilities in
`orbax.checkpoint.experimental.model_surgery.transformations` to facilitate the
conversion process.

### Expected Qwix Structure

Qwix expects checkpoints to be processed into a nested PyTree dictionary
matching the target model's NNX state paths without `.value` suffixes. The
quantized parameter leaves must be represented as dictionaries containing
`qvalue`, `scale`, and optionally `zero_point`.

For Linen models, the checkpoint must instead match the target model's paths
within the `'params'` collection. Linen names submodules assigned as a list in
`setup()` as `layers_0`, `layers_1`, etc., so rename the checkpoint's layer keys
to match (e.g. `layers.0` to `layers_0`) and skip `nnx.restore_int_paths`.

### Example: Pre-Quantized SafeTensors Checkpoint

The following example demonstrates how to load and transform a pre-quantized
SafeTensors checkpoint into the expected Qwix structure using Orbax
transformation utilities.

1.  **Prefix Transform**: Remove top-level wrapper prefixes to match JAX model
    structure.
2.  **Fusing Transform**: Fuse separate gate and up projections for optimized
    SwiGLU execution.
3.  **Renaming Transform**: Rename to match NNX naming conventions (standard
    intercepted layers wrap weights under `.kernel.array`, while custom
    parameters in opaque layers use `.array`).
4.  **Transpose Layouts**: Transpose PyTorch layouts ([out, in]) to match JAX
    layouts ([in, out]).
5.  **Unflatten Transform**: Unflatten to match NNX structure.
6.  **Restore Int Paths**: Convert string digit keys back to integers (NNX
    only).

```py
from typing import Any, Dict
from flax import nnx
import jax.numpy as jnp
from orbax.checkpoint.experimental.model_surgery.transformations import (
    fusing,
    nesting,
    renaming,
)


def transform_checkpoint_tree(tree: Dict[str, Any]) -> Dict[str, Any]:
  """Transforms SafeTensors checkpoints into Qwix's expected structure."""

  def _transpose_layouts(flat_tree: Dict[str, Any]) -> Dict[str, Any]:
    for key in list(flat_tree):
      val = flat_tree[key]
      if "embed_tokens" in key or "lm_head" in key:
        continue
      if val.ndim == 2:
        flat_tree[key] = jnp.transpose(val, (1, 0))
      elif val.ndim == 3:
        flat_tree[key] = jnp.transpose(val, (0, 2, 1))
    return flat_tree

  transforms = [
      # 1. Remove top-level wrapper prefixes to match JAX model structure.
      renaming.rename_by_regex([(r"^(model\.language_model\.|model\.)", "")]),
      # 2. Fuse separate gate and up projections for optimized SwiGLU
      #    execution.
      fusing.fuse_by_pattern(
          pattern=r"^(layers\.\d+\.mlp)\.(gate_proj|up_proj)\.(.+)$",
          unique_parts=["gate_proj", "up_proj"],
          fused_unique_part="gate_up_proj",
          axis=0,
      ),
      # 3. Rename to match NNX naming conventions.
      renaming.rename_by_regex([
          # Standard intercepted layers use `.kernel.array`.
          (
              r"self_attn\.(q_proj|k_proj|v_proj|o_proj)\.weight$",
              r"self_attn.\1.kernel.array.qvalue",
          ),
          (
              r"self_attn\.(q_proj|k_proj|v_proj|o_proj)\.weight_scale_inv$",
              r"self_attn.\1.kernel.array.scale",
          ),
          # Custom opaque layers use `.array` and may need renaming to match
          # the model's parameter names (e.g. adding '_weight').
          (
              r"((?:mlp\.)?experts)\.(gate_up_proj|down_proj)(?:\.weight)?$",
              r"\1.\2_weight.array.qvalue",
          ),
          (
              r"((?:mlp\.)?experts)\.(gate_up_proj|down_proj)"
              + r"(?:\.weight_scale_inv|_scale_inv)$",
              r"\1.\2_weight.array.scale",
          ),
      ]),
      # 4. Transpose PyTorch layouts ([out, in]) to match JAX layouts
      #    ([in, out]).
      _transpose_layouts,
      # 5. Unflatten to match NNX structure.
      nesting.unflatten(separator="."),
      # 6. Convert string digit keys back to integers (NNX only).
      nnx.restore_int_paths,
  ]

  for transform in transforms:
    tree = transform(tree)

  return tree
```

(offline-quantization-opaque-layers)=

## Overriding Custom Parameters in Opaque Layers

Qwix intentionally disables interception for opaque layers like custom Pallas
kernels. Consequently, when constructing your abstract model, opaque layers
remain untouched as full-precision parameters (e.g. `jnp.bfloat16`).

If the checkpoint contains pre-quantized weights for these opaque layers, you
must manually override their abstract PyTree nodes to be quantized dictionaries
containing `qvalue`, `scale`, and optionally `zero_point` before loading the
checkpoint state.

### Example: Pallas MoE Expert Projections

The following example demonstrates how to manually override MoE expert
projection weights inside an opaque Pallas layer. Depending on the model
architecture, the MoE experts may be located directly under the decoder layers
(e.g. Gemma 4) or nested under the MLP block (e.g. Qwen 3.5).

```py
from typing import Any, Dict
from flax import nnx
import jax
import jax.numpy as jnp


def override_opaque_layers(
    model: nnx.Module,
    mesh: jax.sharding.Mesh,
    block_size: int = 64,
):
  """Overrides parameters in opaque layers to be quantized dictionaries."""

  def _to_quantized_dict(
      param: Any, target_sharding: jax.sharding.NamedSharding
  ) -> Dict[str, Dict[str, nnx.Param]]:
    shape = param.shape
    # Adjust scale shape for 2D block quantization.
    scale_shape = shape[:-2] + (
        shape[-2] // block_size,
        shape[-1] // block_size,
    )
    # Override to the target qtype.
    return {
        "array": {
            "qvalue": nnx.Param(
                jax.ShapeDtypeStruct(
                    shape, jnp.float8_e4m3fn, sharding=target_sharding
                )
            ),
            "scale": nnx.Param(
                jax.ShapeDtypeStruct(
                    scale_shape, jnp.bfloat16, sharding=target_sharding
                )
            ),
        }
    }

  # Replace parameters in opaque layer paths with quantized dictionaries.
  # The exact paths and shardings will depend on your model's architecture.
  sharding = jax.sharding.NamedSharding(
      mesh, jax.sharding.PartitionSpec("expert", None, None)
  )

  # Option A: If MoE experts are directly under decoder layers (e.g. Gemma 4)
  moe_modules = [
      (layer.experts, sharding)
      for layer in model.layers
      if hasattr(layer, "experts") and layer.experts is not None
  ]

  # Option B: If MoE experts are nested under the MLP block (e.g. Qwen 3.5)
  # moe_modules = [
  #     (model.blocks.linear_layers.layers.mlp.experts, sharding),
  #     (model.blocks.full_attn_layer.mlp.experts, sharding),
  # ]

  for moe, sharding in moe_modules:
    moe.gate_up_proj_weight = _to_quantized_dict(
        moe.gate_up_proj_weight, sharding
    )
    moe.down_proj_weight = _to_quantized_dict(
        moe.down_proj_weight, sharding
    )
```

For Linen models, override the leaves of the abstract params dict instead, and
box them with `nn.Partitioned` or `nn.LogicallyPartitioned` to shard them (see
[Sharding](offline_quantization.md#offline-quantization-linen-sharding)):

```py
from typing import Any, Dict
from flax import linen as nn
import jax
import jax.numpy as jnp


def override_opaque_layers(params: Dict[str, Any], block_size: int = 64):
  """Overrides params in opaque layers to be quantized dictionaries."""

  def _to_quantized_dict(param: Any) -> Dict[str, Dict[str, nn.Partitioned]]:
    shape = nn.unbox(param).shape
    # Adjust scale shape for 2D block quantization.
    scale_shape = shape[:-2] + (
        shape[-2] // block_size,
        shape[-1] // block_size,
    )
    names = ("expert", None, None)
    # Override to the target qtype.
    return {
        "array": {
            "qvalue": nn.Partitioned(
                jax.ShapeDtypeStruct(shape, jnp.float8_e4m3fn), names=names
            ),
            "scale": nn.Partitioned(
                jax.ShapeDtypeStruct(scale_shape, jnp.bfloat16), names=names
            ),
        }
    }

  # Replace params in opaque layer paths with quantized dictionaries.
  # The exact paths and partitioning will depend on your model's architecture.
  for name, layer in params.items():
    if name.startswith("layers_") and "experts" in layer:
      experts = layer["experts"]
      for weight in ("gate_up_proj_weight", "down_proj_weight"):
        experts[weight] = _to_quantized_dict(experts[weight])
```

```{note}
Linen converts dicts assigned as module attributes in `setup()` to
`FrozenDict`s, so opaque layers that consume these params should accept any
`Mapping`, not only `dict`.
```
