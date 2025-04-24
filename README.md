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
    *   Native: `int4`, `int8`, `fp8`, or any types available in Jax.
    *   Emulated: `int1` to `int7`, `nf4`, or any other LUT-based types.
    *   Asymmetric quantization for native types.
*   Supported Jax ops and their quantization:
    *   XLA:
        *   `conv_general_dilated`: per-channel quantization.
        *   `dot_general` and `einsum`: per-channel and subchannel quantization.
    *   LiteRT:
        *   `conv`, `matmul`, and `fully_connected`: per-channel quantization.
        *   Other ops available in LiteRT: per-tensor quantization.
*   Integration with any Flax Linen or NNX models via a single function call.

## Credits

The development of Qwix was based on the design of
[AQT](http://github.com/google/aqt).
