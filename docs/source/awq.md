# AWQ

[AWQ](https://arxiv.org/abs/2306.00978) (Activation-aware Weight Quantization)
is a post-training quantization method that identifies salient weight channels
based on activation magnitudes and applies per-channel scaling to improve
quantization accuracy.

## AWQ with Qwix

In Qwix, `AwqCalibrationProvider` collects the activation statistics and
`AwqInferenceProvider` runs the model for AWQ inference.

`````{tabs}
````{tab} Linen
```py
model = SomeLinenModel()

# Collect the activation statistics for AWQ calibration.
rules = [
  awq.AwqRule(
    module_path='Dense_0',
    weight_qtype=jnp.int4,
    tile_size=64
  )
]
awq_calibration_provider = awq.AwqCalibrationProvider(rules)
cal_model = qwix.quantize_model(model, awq_calibration_provider)
_, new_variables = cal_model.apply(variables, model_input, mutable='quant_stats')
variables.update(new_variables)

# Use PtqProvider to get the abstract quantized params tree.
ptq_provider = qwix.PtqProvider(rules)
ptq_model = qwix.quantize_model(model, ptq_provider)
abs_variables = jax.eval_shape(ptq_model.init, jax.random.key(0), model_input)

# Use AwqInferenceProvider for inference and apply AWQ params to the model.
awq_params = awq.quantize_params(
    variables['params'], abs_variables['params'], variables['quant_stats']
)
awq_inference_provider = awq.AwqInferenceProvider(rules)
awq_model = qwix.quantize_model(model, awq_inference_provider)
awq_output = awq_model.apply({'params': awq_params}, model_input)
```
````
````{tab} NNX
```py
model = SomeNnxModel()

# Collect the activation statistics for AWQ calibration.
rules = [
  awq.AwqRule(
    module_path='dense1',
    weight_qtype=jnp.int4,
    tile_size=64
  )
]
awq_calibration_provider = awq.AwqCalibrationProvider(rules)
cal_model = qwix.quantize_model(model, awq_calibration_provider, model_input)
_ = cal_model(model_input)

# Use AwqInferenceProvider for inference.
awq_inference_provider = awq.AwqInferenceProvider(rules)
awq_model = qwix.quantize_model(model, awq_inference_provider, model_input)

# Apply AWQ params to the model.
state = nnx.to_pure_dict(nnx.state(cal_model, nnx.Param))
quant_stats = nnx.to_pure_dict(nnx.state(cal_model, qwix.QuantStat))
awq_params = awq.quantize_params(state, awq_model, quant_stats)
nnx.update(awq_model, awq_params)
awq_output = awq_model(model_input)
```
````
`````
