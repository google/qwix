# SmoothQuant

[SmoothQuant](https://arxiv.org/abs/2211.10438) is a post-training quantization
method that smooths the activation outliers by offline migrating the
quantization difficulty from activations to weights with a mathematically
equivalent transformation.

## SmoothQuant with Qwix

In Qwix, `SqCalibrationProvider` collects activation scale statistics and
`SqInferenceProvider` runs the model for SmoothQuant inference.

`````{tabs}
````{tab} Linen
```py
model = SomeLinenModel()

# Collect activation scale statistics for SmoothQuant calibration.
rules = [
  sq.SqRule(
    module_path='Dense_0',
    weight_qtype=jnp.int4,
    act_qtype=jnp.int4,
    alpha=0.5
  )
]
sq_calibration_provider = sq.SqCalibrationProvider(rules)
cal_model = qwix.quantize_model(model, sq_calibration_provider)
_, new_variables = cal_model.apply(variables, model_input, mutable='quant_stats')
variables.update(new_variables)

# Use PtqProvider to get the abstract quantized params tree.
ptq_provider = qwix.PtqProvider(rules)
ptq_model = qwix.quantize_model(model, ptq_provider)
abs_variables = jax.eval_shape(ptq_model.init, jax.random.key(0), model_input)

# Use SqInferenceProvider for inference and apply SmoothQuant params to the model.
sq_params = sq.quantize_params(
    variables['params'], abs_variables['params'], variables['quant_stats']
)
sq_inference_provider = sq.SqInferenceProvider(rules)
sq_model = qwix.quantize_model(model, sq_inference_provider)
sq_output = sq_model.apply({'params': sq_params}, model_input)
```
````
````{tab} NNX
```py
model = SomeNnxModel()

# Collect activation scale statistics for SmoothQuant calibration.
rules = [
  sq.SqRule(
    module_path='dense1',
    weight_qtype=jnp.int4,
    act_qtype=jnp.int4,
    alpha=0.5
  )
]
sq_calibration_provider = sq.SqCalibrationProvider(rules)
cal_model = qwix.quantize_model(model, sq_calibration_provider, model_input)
_ = cal_model(model_input)

# Use SqInferenceProvider for inference.
sq_inference_provider = sq.SqInferenceProvider(rules)
sq_model = qwix.quantize_model(model, sq_inference_provider, model_input)

# Apply SmoothQuant params to the model.
state = nnx.to_pure_dict(nnx.state(cal_model, nnx.Param))
quant_stats = nnx.to_pure_dict(nnx.state(cal_model, qwix.QuantStat))
sq_params = sq.quantize_params(state, sq_model, quant_stats)
nnx.update(sq_model, sq_params)
sq_output = sq_model(model_input)
```
````
`````
