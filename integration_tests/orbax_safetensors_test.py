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

"""Integration test for Orbax + Qwix Safetensors loading."""

from typing import Any

from absl.testing import absltest
from etils import epath
from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
from orbax.checkpoint import v1 as ocp
from orbax.checkpoint.experimental.model_surgery.transformations import nesting
import qwix
from safetensors import numpy as snp


def _flatten_dict(
    d: dict[str, Any], parent_key: str = "", sep: str = "."
) -> dict[str, Any]:
  """Flattens a nested dictionary into a single-level dictionary."""
  items = []
  for k, v in d.items():
    new_key = f"{parent_key}{sep}{k}" if parent_key else k
    if isinstance(v, dict):
      items.extend(_flatten_dict(v, new_key, sep=sep).items())
    else:
      items.append((new_key, v))
  return dict(items)


def load_nested_safetensors(
    directory: str | epath.Path,
    abstract_pytree: Any | None = None,
    mesh: jax.sharding.Mesh | None = None,
) -> Any:
  """Loads a safetensors checkpoint as a nested PyTree."""
  path = epath.Path(directory)

  flat_abstract = None
  if abstract_pytree is not None:
    flat_abstract = _flatten_dict(abstract_pytree)
  elif mesh is not None:
    with ocp.Context(
        checkpoint_layout=ocp.options.CheckpointLayout.SAFETENSORS
    ):
      meta = ocp.pytree_metadata(path)
    sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    flat_abstract = {
        k: jax.ShapeDtypeStruct(shape=v.shape, dtype=v.dtype, sharding=sharding)
        for k, v in meta.metadata.items()
    }

  with ocp.Context(checkpoint_layout=ocp.options.CheckpointLayout.SAFETENSORS):
    flat_tree = ocp.load_pytree(path, abstract_pytree=flat_abstract)

  unflatten_transform = nesting.unflatten()
  nested_tree = unflatten_transform(flat_tree)

  return nested_tree


class QProj(nnx.Module):

  def __init__(self, rngs: nnx.Rngs):
    self.weight = nnx.Param(jnp.ones((128, 128), dtype=jnp.float32))

  def __call__(self, x):
    return jnp.dot(x, self.weight.value)


class LinearAttn(nnx.Module):

  def __init__(self, rngs: nnx.Rngs):
    self.q_proj = QProj(rngs)

  def __call__(self, x):
    return self.q_proj(x)


class Layer(nnx.Module):

  def __init__(self, rngs: nnx.Rngs):
    self.linear_attn = LinearAttn(rngs)

  def __call__(self, x):
    return self.linear_attn(x)


class EmbedTokens(nnx.Module):

  def __init__(self, rngs: nnx.Rngs):
    self.weight = nnx.Param(jnp.ones((128, 128), dtype=jnp.float32))

  def __call__(self, x):
    return jnp.dot(x, self.weight.value)


class LanguageModel(nnx.Module):

  def __init__(self, rngs: nnx.Rngs):
    self.embed_tokens = EmbedTokens(rngs)
    self.layers = nnx.Dict({"0": Layer(rngs)})

  def __call__(self, x):
    x = self.embed_tokens(x)
    return self.layers["0"](x)


class InnerModel(nnx.Module):

  def __init__(self, rngs: nnx.Rngs):
    self.language_model = LanguageModel(rngs)

  def __call__(self, x):
    return self.language_model(x)


class CustomTestModel(nnx.Module):
  """Dummy custom test model."""

  def __init__(self, rngs: nnx.Rngs):
    self.model = InnerModel(rngs)

  def __call__(self, x):
    return self.model(x)


class OrbaxSafetensorsIntegrationTest(absltest.TestCase):

  def test_load_full_precision_weights(self):
    directory = self.create_tempdir().full_path
    path = epath.Path(directory)

    # Save a fake flat model
    flat_tree = {
        "model.language_model.embed_tokens.weight": np.ones(
            (128, 128), dtype=np.float32
        ),
        "model.language_model.layers.0.linear_attn.q_proj.weight": np.ones(
            (128, 128), dtype=np.float32
        ),
    }

    snp.save_file(flat_tree, path / "model.safetensors")

    devices = jax.devices()
    mesh = jax.sharding.Mesh(np.array(devices), ("x",))

    loaded_fp_params = load_nested_safetensors(str(path), mesh=mesh)

    q_rule = qwix.QuantizationRule(
        module_path=".*",
        weight_qtype="float8_e4m3fn",
        act_qtype="float8_e4m3fn",
        act_static_scale=False,
        tile_size=128,
    )

    model_input = jnp.ones((1, 128))

    with jax.set_mesh(mesh):
      abs_ptq_model = nnx.eval_shape(
          lambda: qwix.quantize_model(
              CustomTestModel(rngs=nnx.Rngs(0)),
              qwix.PtqProvider([q_rule]),
              model_input,
          )
      )

      quantized_params = qwix.quantize_params(loaded_fp_params, abs_ptq_model)
      nnx.update(abs_ptq_model, quantized_params)

    # Verify that it loaded cleanly into an nnx module's structure
    self.assertIsNotNone(abs_ptq_model)

    embed_weight = abs_ptq_model.model.language_model.embed_tokens.weight
    self.assertEqual(embed_weight.shape, (128, 128))
    self.assertTrue(hasattr(embed_weight, "array"))
    self.assertTrue(hasattr(embed_weight.array, "qvalue"))
    self.assertEqual(embed_weight.array.qvalue.dtype, jnp.float8_e4m3fn)

    q_proj_weight = abs_ptq_model.model.language_model.layers[
        "0"
    ].linear_attn.q_proj.weight
    self.assertEqual(q_proj_weight.shape, (128, 128))
    self.assertTrue(hasattr(q_proj_weight, "array"))
    self.assertTrue(hasattr(q_proj_weight.array, "qvalue"))
    self.assertEqual(q_proj_weight.array.qvalue.dtype, jnp.float8_e4m3fn)

  def test_load_prequantized_checkpoints(self):
    directory = self.create_tempdir().full_path
    path = epath.Path(directory)

    # Save fake flat model for prequantized state (CustomTestModel shape)
    flat_tree = {
        "model.language_model.embed_tokens.weight.qvalue": np.ones(
            (128, 128), dtype=np.int8
        ),
        "model.language_model.embed_tokens.weight.scale": np.ones(
            (1, 128), dtype=np.float32
        ),
        "model.language_model.layers.0.linear_attn.q_proj.weight.qvalue": (
            np.ones((128, 128), dtype=np.int8)
        ),
        "model.language_model.layers.0.linear_attn.q_proj.weight.scale": (
            np.ones((1, 128), dtype=np.float32)
        ),
    }
    snp.save_file(flat_tree, path / "model.safetensors")

    devices = jax.devices()
    mesh = jax.sharding.Mesh(np.array(devices), ("x",))

    loaded_quant_params = load_nested_safetensors(str(path), mesh=mesh)

    q_rule = qwix.QuantizationRule(
        module_path=".*",
        weight_qtype="float8_e4m3fn",
        act_qtype="float8_e4m3fn",
        act_static_scale=False,
        tile_size=128,
    )

    model_input = jnp.ones((1, 128))

    with jax.set_mesh(mesh):
      abs_ptq_model = nnx.eval_shape(
          lambda: qwix.quantize_model(
              CustomTestModel(rngs=nnx.Rngs(0)),
              qwix.PtqProvider([q_rule]),
              model_input,
          )
      )

      processed_params = qwix.process_prequantized_params(
          loaded_quant_params, abs_ptq_model
      )
      nnx.update(abs_ptq_model, processed_params)

    self.assertIsNotNone(abs_ptq_model)

    embed_weight = abs_ptq_model.model.language_model.embed_tokens.weight
    self.assertEqual(embed_weight.shape, (128, 128))
    self.assertTrue(hasattr(embed_weight, "array"))
    self.assertTrue(hasattr(embed_weight.array, "qvalue"))
    self.assertEqual(embed_weight.array.qvalue.dtype, jnp.float8_e4m3fn)

    q_proj_weight = abs_ptq_model.model.language_model.layers[
        "0"
    ].linear_attn.q_proj.weight
    self.assertEqual(q_proj_weight.shape, (128, 128))
    self.assertTrue(hasattr(q_proj_weight, "array"))
    self.assertTrue(hasattr(q_proj_weight.array, "qvalue"))
    self.assertEqual(q_proj_weight.array.qvalue.dtype, jnp.float8_e4m3fn)


if __name__ == "__main__":
  absltest.main()
