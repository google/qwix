# Qwix: a quantization library for Jax.

Qwix is a Jax quantization library supporting Quantization-Aware Training (QAT)
and Post-Training Quantization (PTQ) for both XLA targets (CPU/GPU/TPU) and ODML
targets (LiteRT).

## Features

*   Supported schemas:
    * Weight-only quantization.
    * Dynamic-range quantization.
    * Static-range quantization.
*   Supported modes:
    *   QAT: this mode emulates quantized behavior during serving with fake
        quantization.
    *   PTQ: this mode achieves the best serving performance on XLA devices such
        as TPU and GPU.
    *   ODML: this mode adds proper annotation to the model so that the LiteRT
        converter could produce full integer models.
    *   LoRA/QLoRA: this mode enables LoRA and QLoRA on a model.
*   Supported numerics:
    *   Native: `int4`, `int8`, `fp8`.
    *   Emulated: `int1` to `int7`, `nf4`.
*   Supported array calibration methods:
    *   `absmax`: symmetric quantization using maximum absolute value.
    *   `minmax`: asymmetric quantization using minimum and maximum values.
    *   `rms`: symmetric quantization using root mean square.
    *   `fixed`: fixed range.
*   Supported Jax ops and their quantization granularity:
    *   XLA:
        *   `conv_general_dilated`: per-channel.
        *   `dot_general` and `einsum`: per-channel and sub-channel.
    *   LiteRT:
        *   `conv`, `matmul`, and `fully_connected`: per-channel.
        *   Other ops available in LiteRT: per-tensor.
*   Integration with any Flax Linen or NNX models via a single function call.

## Usage

Qwix doesn't provide a PyPI package yet. To use Qwix, you need to install from
GitHub directly.

```sh
pip install git+https://github.com/google/qwix
```

### Model definition

We're going to use a simple MLP model in the example. Qwix integrates with
models without need to modify their code, so any model can be used below.

```py
import jax
from flax import linen as nn

class MLP(nn.Module):

  dhidden: int
  dout: int

  @nn.compact
  def __call__(self, x):
    x = nn.Dense(self.dhidden, use_bias=False)(x)
    x = nn.relu(x)
    x = nn.Dense(self.dout, use_bias=False)(x)
    return x

model = MLP(64, 16)
model_input = jax.random.uniform(jax.random.key(0), (8, 16))
```

## Quantization config

Qwix uses a regex-based configuration system to instruct how to quantize a Jax
model. Configurations are defined as a list of `QuantizationRule`. Each rule
consists of a key that matches Flax modules, and a set of values that control
quantization behavior.

For example, to quantize the above model in int8 (w8a8), we need to define the
rules as below.

```py
import qwix

rules = [
    qwix.QuantizationRule(
        module_path='.*',  # this rule matches all modules.
        weight_qtype='int8',  # quantizes weights in int8.
        act_qtype='int8',  # quantizes activations in int8.
    )
]
```

Unlike some other libraries that provides limited number of **quantization
recipes**, Qwix doesn't have a list of presets. Instead, different quantization
schemas are achieved by combinations of quantization configs.

### Post-Training Quantization

To apply PTQ to the above model, we only need to call `qwix.quantize_model`.

```py
ptq_model = qwix.quantize_model(model, qwix.PtqProvider(rules))
```

Now the `ptq_model` will contain quantized weights. We could verify that.

```py
>>> jax.eval_shape(ptq_model.init, jax.random.key(0), model_input)['params']
{
  'Dense_0': {
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
```

### Weight quantization

Since Flax Linen modules are pure-functional, weights quantization are separate
from model quantization. To quantize weights for the above `ptq_model`, we
need to call `qwix.quantize_params`.

```py
# Floating-point params, usually loaded from checkpoints.
fp_params = ...

# Abstract quantized params, which serve as a template for quantize_params.
abs_ptq_params = jax.eval_shape(ptq_model.init, jax.random.key(0), model_input)['params']

# Weight quantization.
ptq_params = qwix.quantize_params(fp_params, abs_ptq_params)

# ptq_params contains the quantized weights and can be consumed by ptq_model.
quantized_model_output = ptq_model.apply({'params': ptq_params}, model_input)
```

## Relation with AQT

The design of Qwix was inspired by [AQT](https://github.com/google/aqt) and
borrowed many great ideas from it. Here's a brief list of the similarities and
the differences.

*   Qwix's `QArray` is similar to AQT's `QTensor`, both supporting sub-channel
    quantization.
*   AQT has quantized training support (quantized forwards and quantized
    backwards), while Qwix's QAT is based on fake quantization, which doesn't
    improve the training performance.
*   AQT provides drop-in replacements for `einsum` and `dot_general`, each of
    these having to be configured separately. Qwix provides addtional mechanisms
    to integrate with a whole model implicitly.
*   Applying static-range quantization is easier in Qwix as it has more in-depth
    support with Flax.
