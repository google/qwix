# Offline Quantization

Qwix's Offline Quantization feature enables loading pre-quantized checkpoints
and continuing training or inference workloads in the desired target numeric.

## Overview

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
6.  **Restore Int Paths**: Convert string digit keys back to integers.

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
      # 6. Convert string digit keys back to integers.
      nnx.restore_int_paths,
  ]

  for transform in transforms:
    tree = transform(tree)

  return tree
```

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
