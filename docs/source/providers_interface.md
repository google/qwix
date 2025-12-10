# Provider Interface

This module defines the abstract base classes that all quantization providers
must implement.

If you are building a custom quantization strategy or extending Qwix, you will
implement the `QuantizationProvider` interface.

```{eval-rst}
.. autoclass:: qwix.QuantizationProvider
  :members:

.. autoclass:: qwix.QuantizationRule
  :members:
```
