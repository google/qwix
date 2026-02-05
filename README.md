Qwix: A Quantization Library for JAX
Qwix is a JAX quantization library that supports Quantization-Aware Training (QAT) and Post-Training Quantization (PTQ) for both XLA targets (CPU/GPU/TPU) and ODML targets (LiteRT).

Features
Supported Schemas
Weight-only quantization

Dynamic-range quantization

Static-range quantization

Supported Modes
QAT: Emulates quantized inference behavior during training using fake quantization.

PTQ: Provides the best inference performance on XLA devices such as TPU and GPU.

ODML: Adds the required annotations so that the LiteRT converter can generate full-integer models.

LoRA/QLoRA: Enables LoRA and QLoRA on a model.

Supported Numerics
Native: int4, int8, fp8

Emulated: int1 to int7, nf4

Supported Array Calibration Methods
absmax: Symmetric quantization using the maximum absolute value

minmax: Asymmetric quantization using the minimum and maximum values

rms: Symmetric quantization using root mean square

fixed: Fixed quantization range

Supported JAX Ops and Quantization Granularity
XLA
conv_general_dilated: per-channel

dot_general and einsum: per-channel and sub-channel

LiteRT
conv, matmul, and fully_connected: per-channel

Other LiteRT-supported ops: per-tensor

Model Integration
Works with any Flax Linen or NNX model using a single function call.

Usage
Qwix is not available on PyPI yet. To use it, install directly from GitHub:

pip install git+https://github.com/google/qwix
Model Definition
In this example, we use a simple MLP model. Since Qwix integrates without requiring changes to model code, any model can be used.

import jax
from flax import linen as nn

class MLP(nn.Module):

dhidden: int
dout: int

@nn.compact
def **call**(self, x):
x = nn.Dense(self.dhidden, use_bias=False)(x)
x = nn.relu(x)
x = nn.Dense(self.dout, use_bias=False)(x)
return x

model = MLP(64, 16)
model_input = jax.random.uniform(jax.random.key(0), (8, 16))
Quantization Config
Qwix uses a regex-based configuration system to define how a JAX model should be quantized. Configurations are specified as a list of QuantizationRule. Each rule contains:

A key that matches Flax modules

A set of values that control quantization behavior

For example, to quantize the model above using int8 (w8a8), define the rules as follows:

import qwix

rules = [
qwix.QuantizationRule(
module_path='.*', # matches all modules
weight_qtype='int8', # quantizes weights to int8
act_qtype='int8', # quantizes activations to int8
)
]
Unlike some libraries that provide only a limited set of quantization recipes, Qwix does not use presets. Instead, different quantization schemas are achieved by combining configuration options.

Post-Training Quantization (PTQ)
To apply PTQ to the model above, simply call qwix.quantize_model:

ptq_model = qwix.quantize_model(model, qwix.PtqProvider(rules))
The resulting ptq_model contains quantized weights. This can be verified as shown below:

> > > jax.eval_shape(ptq_model.init, jax.random.key(0), model_input)['params']
> > > {
> > > 'Dense_0': {

    'kernel': WithAux(
        array=QArray(
            qvalue=ShapeDtypeStruct(shape=(16, 64), dtype=int8),
            scale=ShapeDtypeStruct(shape=(1, 64), dtype=float32),
            ...
        ),
        ...
    )

},
'Dense_1': {
'kernel': WithAux(
array=QArray(
qvalue=ShapeDtypeStruct(shape=(64, 16), dtype=int8),
scale=ShapeDtypeStruct(shape=(1, 16), dtype=float32),
...
),
...
)
}
}
Weight Quantization
Because Flax Linen modules are pure-functional, weight quantization is handled separately from model quantization. To quantize weights for the ptq_model, use qwix.quantize_params.

# Floating-point params, typically loaded from checkpoints.

fp_params = ...

# Abstract quantized params used as a template for quantize_params.

abs_ptq_params = jax.eval_shape(ptq_model.init, jax.random.key(0), model_input)['params']

# Weight quantization.

ptq_params = qwix.quantize_params(fp_params, abs_ptq_params)

# ptq_params now contains quantized weights and can be used with ptq_model.

quantized_model_output = ptq_model.apply({'params': ptq_params}, model_input)
Relation with AQT
The design of Qwix was inspired by AQT and borrows many of its ideas. Below is a summary of similarities and differences:

Similarities
Qwix’s QArray is similar to AQT’s QTensor, and both support sub-channel quantization.

Differences
AQT supports quantized training (quantized forward and backward passes), while Qwix’s QAT is based on fake quantization and does not improve training performance.

AQT provides drop-in replacements for einsum and dot_general, which must be configured separately. Qwix provides additional mechanisms to integrate quantization across the entire model implicitly.

Applying static-range quantization is easier in Qwix due to deeper integration with Flax.

Citing Qwix
To cite Qwix, please use the following citation:

bibtex
Copy code
@software{Qwix,
title = {Qwix: A Quantization Library for Jax},
author={Dangyi Liu, Jiwon Shin, et al.},
year = {2024},
howpublished = {\url{https://github.com/google/qwix}},
}
