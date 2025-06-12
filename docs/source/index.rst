.. Qwix documentation master file, created by
   sphinx-quickstart on Wed Jun 12 05:46:19 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Introduction to Qwix
====================

`Qwix <https://github.com/google/qwix>`_ is a Jax quantization library for both research and production.

.. toctree::
   :maxdepth: 1
   :caption: Contents:

   get_started
   basics
   qat
   ptq
   odml
   lora
   extend

Why Qwix
--------

* Qwix is the only Jax quantization solution that supports QAT, PTQ on TPU/GPU, and ODML quantization in one library.
* Qwix has well tested static range quantization (SRQ) support.
* In ODML quantization mode, Qwix can quantize every op in the model, and export a full integer network.
* Qwix integrates with models without need to modify the model code.
* Qwix is open and extensible. Building new algorithms on Qwix is easy.

Features
--------

* Supported schemas:

  * Weight-only quantization.
  * Dynamic-range quantization.
  * Static-range quantization.

* Supported modes:

  * **QAT**: this mode emulates quantized behavior during serving with fake quantization.
  * **PTQ**: this mode achieves the best serving performance on XLA devices such as TPU and GPU.
  * **ODML**: this mode adds proper annotation to the model so that the LiteRT converter could produce full integer models.
  * **LoRA/QLoRA**: this mode enables LoRA and QLoRA on a model.

* Supported numerics:

  * Native: ``int4``, ``int8``, ``fp8``.
  * Emulated: ``int1`` to ``int7``, ``nf4``.

* Supported array calibration methods:

  * ``absmax``: symmetric quantization using maximum absolute value.
  * ``minmax``: asymmetric quantization using minimum and maximum values.
  * ``rms``: symmetric quantization using root mean square, also known as "MSE Quant".
  * ``fixed``: fixed range.

* Supported Jax ops and their quantization granularity:

  * XLA:

    * ``conv_general_dilated``: per-channel.
    * ``dot_general`` and ``einsum``: per-channel and sub-channel.

  * LiteRT:

    * ``conv``, ``matmul``, and ``fully_connected``: per-channel.
    * Other ops available in LiteRT: per-tensor.

* Integration with any Flax Linen or NNX models via a single function call.

Relation with AQT
-----------------

The design of Qwix was inspired by `AQT <https://github.com/google/aqt>`_ and borrowed many great ideas from it. Here's a brief list of the similarities and the differences.

* Qwix's ``QArray`` is similar to AQT's ``QTensor``, both supporting sub-channel quantization.
* The PTQ mode in Qwix has a similar behavior and slightly better performance than the serving mode in AQT, due to the ``TransposedQArray`` design.
* AQT has quantized training support (quantized forwards and quantized backwards), while Qwix's QAT is based on fake quantization, which doesn't improve the training performance.
* AQT provides drop-in replacements for ``einsum`` and ``dot_general``, each of these having to be configured separately. Qwix provides addtional mechanisms to integrate with a whole model implicitly.
* Applying static-range quantization is easier in Qwix as it has more in-depth support with Flax.
