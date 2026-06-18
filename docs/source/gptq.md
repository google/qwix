# GPTQ

[GPTQ](https://arxiv.org/abs/2210.17323) is a post-training quantization method
based on approximate second-order information.

## GPTQ with Qwix

In Qwix, `GptqCalibrationProvider` collects the Hessian statistics and
`PtqProvider` runs the model for GPTQ inference.

`````{tabs}
````{tab} Linen
```py
model = SomeLinenModel()

# Collect the Hessian statistics for GPTQ calibration.
rules = [
  gptq.GptqRule(
    module_path='Dense_0',
    weight_qtype=jnp.int8
  )
]
gptq_calibration_provider = gptq.GptqCalibrationProvider(rules)
cal_model = qwix.quantize_model(model, gptq_calibration_provider)
for batch in calibration_dataset:
    # Pass the batch to the model and capture the mutated quant_stats.
    _, new_variables = cal_model.apply(variables, batch, mutable='quant_stats')
    # Update your variables dictionary so the stats accumulate.
    variables.update(new_variables)

# Use PtqProvider for inference.
ptq_provider = qwix.PtqProvider(rules)
gptq_model = qwix.quantize_model(model, ptq_provider)
abs_variables = jax.eval_shape(gptq_model.init, jax.random.key(0), model_input)

# Apply GPTQ params to the model.
gptq_params = gptq.quantize_params(
    variables['params'], abs_variables['params'], variables['quant_stats']
)
gptq_output = gptq_model.apply({'params': gptq_params}, model_input)
```
````
````{tab} NNX
```py
model = SomeNnxModel()

# Collect the Hessian statistics for GPTQ calibration.
rules = [
  gptq.GptqRule(
    module_path='dense1',
    weight_qtype=jnp.int8
  )
]
gptq_calibration_provider = gptq.GptqCalibrationProvider(rules)
cal_model = qwix.quantize_model(model, gptq_calibration_provider, model_input)
# Iterate over your calibration dataset.
for batch in calibration_dataset:
    # The quant_stats state updates automatically inside the NNX module.
    _ = cal_model(batch)

# Use PtqProvider for inference.
ptq_provider = qwix.PtqProvider(rules)
gptq_model = qwix.quantize_model(model, ptq_provider, model_input)

# Apply GPTQ params to the model.
state = nnx.to_pure_dict(nnx.state(cal_model, nnx.Param))
quant_stats = nnx.to_pure_dict(nnx.state(cal_model, qwix.QuantStat))
gptq_params = gptq.quantize_params(state, gptq_model, quant_stats)
nnx.update(gptq_model, gptq_params)
gptq_output = gptq_model(model_input)
```
````
`````
