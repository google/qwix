# Provider Interface

This module defines the abstract base classes that all quantization providers
must implement.

If you are building a custom quantization strategy or extending Qwix, you will
implement the `QuantizationProvider` interface.

`BoxedParamProvider` is the shared base for providers that need runtime JAX
operation interception plus `WithAux` parameter substitution. Use it when a
provider needs boxed frozen weights, such as PTQ inference or QLoRA training,
without implying that the provider itself is a PTQ inference provider.

```{eval-rst}
.. autoclass:: qwix.QuantizationProvider
  :members:

.. autoclass:: qwix.BoxedParamProvider
  :members:

.. autoclass:: qwix.QuantizationRule
  :members:
```
