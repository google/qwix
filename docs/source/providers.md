# Standard Providers

This section details the built-in quantization providers available in Qwix.

## Quantized Training (QT)

The `QtProvider` implements Quantized Training (QT). This provider performs
quantization on both the forward and backward passes during training.

```{eval-rst}
.. autoclass:: qwix.QtProvider
  :members:
  :undoc-members:

.. autoclass:: qwix.QtRule
  :members:
  :undoc-members:
```

## Post-Training Quantization (PTQ)

These APIs handle Post-Training Quantization, which quantizes a pre-trained
model without requiring a full retraining loop.

```{eval-rst}
.. autoclass:: qwix.PtqProvider
  :members:
  :undoc-members:

.. autofunction:: qwix.quantize_params
```

## On-Device ML (ODML)

These APIs are specialized for converting JAX models to run on edge devices via
LiteRT (formerly TensorFlow Lite). They handle specific constraints required by
mobile hardware.

```{eval-rst}
.. autoclass:: qwix.OdmlConversionProvider
  :members:
  :undoc-members:

.. autoclass:: qwix.OdmlQatProvider
  :members:
  :undoc-members:
```

## LoRA Quantization

These APIs combine Low-Rank Adaptation (LoRA) with quantization, allowing for
memory-efficient fine-tuning of large models.

```{eval-rst}
.. autoclass:: qwix.LoraProvider
  :members:
  :undoc-members:

.. autoclass:: qwix.LoraRule
  :members:
  :undoc-members:

.. autofunction:: qwix.apply_lora_to_model
```
