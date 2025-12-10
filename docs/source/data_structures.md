# Data Structures

The `QArray` is the backbone of the Qwix library. It serves as the fundamental
data structure that encapsulates quantized data (integers) alongside its
quantization parameters (scale and zero-point).

By wrapping these elements together, `QArray` allows quantized tensors to be
passed around and manipulated seamlessly, ensuring that the quantization
context is preserved throughout the computation graph.

```{eval-rst}
.. autoclass:: qwix.QArray
  :members:
  :undoc-members:
  :exclude-members: qvalue, scale, zero_point, qtype

.. autofunction:: qwix.quantize
.. autofunction:: qwix.dequantize
```
