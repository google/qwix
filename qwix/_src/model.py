# Copyright 2024 Google LLC
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
"""Flax integration for QWIX."""

from collections.abc import Collection
import functools
import inspect
from typing import Any, TypeVar

from flax import linen as nn
from flax import nnx
from qwix._src import interception
from qwix._src import qconfig


ModelType = TypeVar("ModelType")


def quantize_model(
    model: ModelType,
    provider: qconfig.QuantizationProvider,
    *model_inputs: Any,
    methods: Collection[str] = ("__call__",),
    **model_inputs_kwargs: Any,
) -> ModelType:
  """Quantize a flax model.

  Args:
    model: The model to quantize, which can be previously quantized.
    provider: The quantization provider.
    *model_inputs: The inputs to the model, which should be provided if and only
      if the model is an NNX model.
    methods: The methods to quantize.
    **model_inputs_kwargs: The keyword arguments to the model, which should be
      provided if and only if the model is an NNX model.

  Returns:
    A quantized model instance.
  """
  if isinstance(model, nn.Module):
    if model_inputs or model_inputs_kwargs:
      raise ValueError("Model inputs must not be provided for linen models.")
    return quantize_linen_model(model, provider, methods)
  elif isinstance(model, nnx.Module):
    if not model_inputs and not model_inputs_kwargs:
      raise ValueError("Model inputs must be provided for nnx models.")
    if len(methods) != 1:
      raise ValueError("Only one method is supported for nnx models.")
    return quantize_nnx_model(
        model,
        provider,
        *model_inputs,
        call_method=next(iter(methods)),
        **model_inputs_kwargs,
    )
  else:
    raise ValueError(f"Unsupported model type: {type(model)}")


def quantize_linen_model(
    model: nn.Module,
    provider: qconfig.QuantizationProvider,
    methods: Collection[str],
) -> nn.Module:
  """Quantize a linen model."""

  def _is_in_nn_module() -> bool:
    nn_module: nn.Module = nn.module._context.module_stack[-1]  # pylint: disable=protected-access
    return nn_module and nn_module.scope is not None

  # Make a copy of the model to avoid modifying the original model.
  model = model.copy()

  # We need to modify the model class's methods because nn.Module.apply will
  # invoke a copy of the model rather than the original one, so setting the
  # attributes of the model doesn't always work. For example, in a VAE model,
  # we intercept the encode and decode methods, but invoke the __call__ method
  # during training. The interception on encode and decode methods will be
  # unset when the model is copied.
  model_class = model.__class__
  # If the model class is already quantized, get the unquantized type.
  if hasattr(model_class, "_unquantized_type"):
    model_class = model_class._unquantized_type  # pylint: disable=protected-access

  new_fields = {"_unquantized_type": model_class}
  for method_name in methods:
    method = getattr(model_class, method_name)

    # Step 1: unwrap if needed.
    wrapped = False
    # This is a while loop because the method may be wrapped e.g. by nn.jit as
    # jit(wrap_method_once(method)). We discard the jit decorator for now.
    while hasattr(method, "method_handler_wrapped"):
      # Apply interception to the unwrapped method so that input/output
      # transformations are called inside a proper linen scope.
      method = method.__wrapped__
      wrapped = True

    # Step 2: intercept the method.
    method = interception.wrap_func_intercepted(
        method,
        provider.get_intercept_map,
        input_transform=functools.partial(_input_transform, provider),
        output_transform=functools.partial(
            provider.process_model_output, method_name
        ),
        should_intercept=_is_in_nn_module,
    )

    # Step 3: wrap if needed.
    if wrapped:
      method = nn.module.wrap_method_once(method)

    new_fields[method_name] = method

  # Create a new class for the model.
  model.__class__ = type(model_class.__name__, (model_class,), new_fields)
  return model


def quantize_nnx_model(
    model: nnx.Module,
    provider: qconfig.QuantizationProvider,
    *model_inputs: Any,
    call_method: str = "__call__",
    **model_inputs_kwargs: Any,
) -> nnx.Module:
  """Quantize an NNX model.

  To fully quantize an NNX model, Qwix needs to run the model at least once.
  Thus model inputs are required. The actual values do not matter.

  If the model already contains the correct original weights, this function will
  quantize them correctly.

  Args:
    model: An NNX model instance, which can be previously quantized.
    provider: The quantization provider.
    *model_inputs: The inputs to the model.
    call_method: The name of the method to call, default to __call__.
    **model_inputs_kwargs: The keyword arguments to the model.

  Returns:
    A quantized model instance.
  """
  # Make a copy of the model to avoid modifying the original model.
  model = nnx.clone(model)

  model_class = model.__class__
  # If the model class is already quantized, get the unquantized type.
  if hasattr(model_class, "_unquantized_type"):
    model_class = model_class._unquantized_type  # pylint: disable=protected-access

  # Only intercepts the `call_method` of the nnx module for now.
  #
  # The class method type(model).__call__ will be invoked when calling the
  # instance of the class so we create a new class with the intercepted __call__
  # and the update the class of the model.
  new_fields = {
      "_unquantized_type": model_class,
      call_method: interception.wrap_func_intercepted(
          getattr(model_class, call_method),
          provider.get_intercept_map,
          input_transform=functools.partial(_input_transform, provider),
          output_transform=functools.partial(
              _output_transform_nnx, provider, call_method
          ),
      ),
  }
  model.__class__ = type(model_class.__name__, (model_class,), new_fields)

  rngs = model_inputs_kwargs.pop("rngs", None)
  if rngs is not None and not isinstance(rngs, nnx.Rngs):
    raise ValueError("Rngs must be an nnx.Rngs instance.")

  # Unlike linen module, nnx module does not have scope or path attribute, we
  # need to iterate over all modules and set the path for them.
  for path, module in model.iter_modules():
    module.qwix_path = path
    # Disable quant_stats update for the first call.
    module.disable_quant_stats_update = True
    # Set the rngs, which is shared by all modules and useful for lora weights
    # initialization.
    module.qwix_rngs = rngs

  # Because nnx modules are stateful, we need to call them once to initialize
  # them (convert weights, create quant_stats) unless users explicitly opt out.
  if not model_inputs_kwargs.pop("skip_nnx_init", False):
    getattr(model, call_method)(*model_inputs, **model_inputs_kwargs)

  # Enable quant_stats update for subsequent calls and unset the rngs.
  # If users want to use the rngs for other purposes, e.g. stochastic rounding,
  # lora dropout, they need to set it themselves.
  model.set_attributes(disable_quant_stats_update=False, qwix_rngs=None)

  return model


def _input_transform(provider: qconfig.QuantizationProvider, args, kwargs):
  model, *args = args
  model, args, kwargs = provider.process_model_inputs(model, args, kwargs)
  return (model, *args), kwargs


def _output_transform_nnx(
    provider: qconfig.QuantizationProvider, method_name: str, output: Any
):
  # Create a frame with a variable named `self` pointing to the model so that
  # flax_util.get_current_module() can work inside the output transform.
  # We cannot use the model in quantize_nnx_model because users may choose to
  # clone the model.
  args = inspect.currentframe().f_back.f_locals["args"]  # pytype: disable=attribute-error
  self = args[0]  # pylint: disable=unused-variable
  return provider.process_model_output(method_name, output)
