.. _post_training_quantization:

Post-Training Quantization (PTQ)
================================

.. note::
    This is for PTQ on XLA devices (CPU/GPU/TPU). ODML models deployed through
    the LiteRT converter should use :doc:`ODML modes <odml>`.

Post-training quantization optimizes serving performance on XLA devices. It's
achieved by quantizing weights ahead of time and computing with quantized types.
When static-range quantization is enabled, PTQ also pre-calculates the scales so
that the cost of activation quantization is minimal.

PTQ can be used alone, or used together with QAT to recover some quality.

PTQ with Qwix
-------------

PTQ in Qwix is implemented by ``PtqProvider`` and can be applied to model with
``quantize_model``.

.. tabs::

    .. tab:: Linen

        .. code-block:: python

            fp_model = SomeLinenModel()
            ptq_model = qwix.quantize_model(fp_model, qwix.PtqProvider(rules))

    .. tab:: NNX

        .. code-block:: python

            fp_model = SomeNnxModel()
            ptq_model = qwix.quantize_model(fp_model, qwix.PtqProvider(rules), model_input)

        Since NNX model allocates weights upon initialization, it's possible that the
        floating-point weights cannot fit in the serving topology at all. Using JIT can
        eliminate the intermediate ``fp_model``.

        .. code-block:: python

            def create_quantized_model():
              fp_model = SomeNnxModel()
              return qwix.quantize_model(fp_model, qwix.PtqProvider(rules), model_input)

            ptq_model = nnx.jit(create_quantized_model)()

        A more common practice is to use ``eval_shape`` instead of JIT above to obtain an
        abstract PTQ model, and use ``quantize_params`` below to obtain the quantized
        weights, as demonstrated below.


Weight quantization
-------------------

Besides quantizing the model, PTQ also requires weights to be quantized ahead of
time. This can be achieved by the ``quantize_params`` function.

.. tabs::

    .. tab:: Linen

        .. code-block:: python

            # Floating-point params, usually loaded from checkpoints.
            fp_params = ...

            # Initialize abstract quantized params, which serve as a template so that the
            # quantize_params function knows how to quantize each weight.
            abs_ptq_variables = jax.eval_shape(ptq_model.init, jax.random.key(0), model_input)

            ptq_params = qwix.quantize_params(fp_params, abs_ptq_variables['params'])

            # ptq_params contains the quantized weights and can be consumed by ptq_model.
            quantized_model_output = ptq_model.apply({'params': ptq_params}, model_input)

    .. tab:: NNX

        .. code-block:: python

            # Load or initialize unquantized params. This should be a "pure dict".
            fp_params = ...

            # Create an abstract quantized model, which serves as a template so that the
            # quantize_params function knows how to quantize each weight.
            abs_ptq_model = nnx.eval_shape(create_quantized_model)

            ptq_params = qwix.quantize_params(fp_params, abs_ptq_model)

            # Update the abstract model with the actual quantized params.
            nnx.update(abs_ptq_model, ptq_params)
            # Now abs_ptq_model contains the actual weights and can be called.
            abs_ptq_model(model_input)


The intermediate ``ptq_params`` can be saved to disk, creating a quantized
checkpoint. This practice is commonly known as **offline quantization**. Qwix
recommends **online quantization** whenever possible because

* Eliminating the offline quantization step improves the development velocity,
    and reduces the maintenance cost of multiple checkpoints.
* The structure of ``ptq_params`` is the implementation detail of Qwix, which is
    subject to change, creating incompatibility of quantized checkpoints.

When using online quantization, the ``fp_params`` may be too large to fit in the
HBM of the serving topology. To solve this, ``quantize_params`` also takes a
subtree of ``fp_params``. For example, we could load the checkpoints layer by
layer and quantize each layer immediately, which is known as **pipelined
checkpoint loading and quantization**.

Alternative way to quantize weights
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For smaller models where HBM limit is not a concern, weight quantization can be
achieved by feeding the unquantized weights to the PTQ models themselves. The
PTQ models will quantize them correctly and replace the original weights. This
can be convenient especially for NNX models.

.. tabs::

    .. tab:: Linen

        .. code-block:: python

            # Assume fp_variables contains the correct unquantized weights.
            _, ptq_variables = ptq_model.apply(fp_variables, model_input, mutable=True)
            # ptq_variables contains the quantized weights now.

        This could look tricky and non-obvious for most users. Thus it's recommended to
        always use ``quantized_params`` for Linen models.

    .. tab:: NNX

        .. code-block:: python

            # Assume model contains the correct unquantized weights. quantize_model will
            # also quantize the weights.
            ptq_model = qwix.quantize_model(model, qwix.PtqProvider(rules), model_input)
            # ptq_model contains the correct quantized weights now.

Static-range quantization
^^^^^^^^^^^^^^^^^^^^^^^^^^^

In :ref:`SRQ <srq>`, the PTQ model contains extra static scales that needs
to be calculated from the ``quant_stats`` collected during QAT. In this case,
additional arguments need to be provided to ``quantize_params``.

.. tabs::

    .. tab:: Linen

        .. code-block:: python

            model = SomeLinenModel(...)
            rules = [
                qconfig.QuantizationRule(
                    weight_qtype="int8",
                    act_qtype="int8",
                    act_static_scale=True,
                ),
            ]

            qat_model = qwix.quantize_model(model, qwix.QatProvider(rules))
            qat_variables = qat_model.init(jax.random.key(0), model_input)
            # qat_variables contains "params" and "quant_stats".

            ptq_model = qwix.quantize_model(model, qwix.PtqProvider(rules))
            abs_ptq_variables = jax.eval_shape(ptq_model.init, jax.random.key(0), model_input)

            ptq_params = qwix.quantize_params(
                qat_variables['params'],
                abs_ptq_variables['params'],
                qat_variables['quant_stats'],
            )

    .. tab:: NNX

        .. code-block:: python

            model = SomeNnxModel(...)
            rules = [
                qconfig.QuantizationRule(
                    weight_qtype="int8",
                    act_qtype="int8",
                    act_static_scale=True,
                ),
            ]

            qat_model = qwix.quantize_model(model, qwix.QatProvider(rules), model_input)
            # qat_model contains both params and quant_stats.

            # quantize_model converts the quant_stats if the PTQ model is converted from
            # a QAT model.
            ptq_model = qwix.quantize_model(qat_model, qwix.PtqProvider(rules), model_input)

            # It's also possible to use quantize_params for NNX models.
            ptq_params = qwix.quantize_params(
                nnx.to_pure_dict(nnx.state(qat_model, nnx.Param)),
                ptq_model,  # or abs_ptq_model
                nnx.to_pure_dict(nnx.state(qat_model, qwix.QuantStat)),
            )
