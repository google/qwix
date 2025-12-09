# Core Operations

This module provides the low-level mathematical operations for Qwix.

These functions are designed to be **quantized equivalents of standard JAX
operations**. For example, `qwix.dot_general` mirrors `jax.lax.dot_general`,
but is specifically engineered to accept `QArray` inputs. It performs the
underlying quantized arithmetic (such as integer matrix multiplication) and
correctly handles the propagation of scales and zero-points.

```{eval-rst}
.. autofunction:: qwix.conv_general_dilated
.. autofunction:: qwix.dot
.. autofunction:: qwix.dot_general
.. autofunction:: qwix.einsum
.. autofunction:: qwix.ragged_dot
.. autofunction:: qwix.ragged_dot_general
```
