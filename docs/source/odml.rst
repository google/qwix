.. _odml_quantization:

ODML Quantization
=================

A distinct feature of Qwix is its ODML support. It's able to quantize every op,
perform QAT, and export full-integer LiteRT models.

XLA targets vs ODML targets
---------------------------

Quantization for XLA targets and ODML targets are different due to their
different hardware characteristics. XLA devices are more versatile and powerful,
as they are also designed for training. Quantizing the matmul and convolution
ops is usually sufficient for these platforms; quantizing the remaining
element-wise ops typically offers negligible benefits.

In contrast, ODML devices are inexpensive and have diverse runtimes specifically
for quantized inference. Quantized inputs with static ranges are usually
required by kernels in those runtimes, and they generate quantized outputs with
static ranges too. Some of those runtimes lack floating point MXUs,
necessitating static quantization for every operator. Fusion is also common in
ODML runtimes. For example, a matmul kernel can often fuse the subsequent
addition and ReLU.

The illustration shows how a dense layer is quantized differently on XLA targets
vs ODML targets.

.. container:: flex-container text-center

   .. graphviz::

      digraph {
        graph [label="Unquantized model" ordering="in" rankdir=LR]
        node [color="none" style="filled"]

        matmul [color=lightskyblue]
        add [color=lightskyblue]
        relu [color=lightskyblue]

        input -> matmul -> add -> relu -> output
        weight -> matmul
        bias -> add [ordering=in]
      }

   .. graphviz::

      digraph {
        graph [label="XLA targets" ordering="in" rankdir=LR]
        node [color="none" style="filled"]

        qw [label="quantized\nweight"]
        qx [label="quantize" color="burlywood1"]
        dq [label="dequantize" color="burlywood1"]
        int_op [label="int\nmatmul" color=lightskyblue]
        add [color=lightskyblue]
        relu [color=lightskyblue]

        input -> qx -> int_op
        qw -> int_op
        int_op -> dq -> add -> relu -> output
        bias -> add
      }

   .. graphviz::

      digraph {
        graph [label="ODML targets" ordering="in" rankdir=LR]
        node [color="none" style="filled"]

        qx [label="quantized\ninput"]
        qw [label="quantized\nweight" rank=0]
        bias [label="quantized\nbias"]
        output [label="quantized\noutput"]
        int_op [label="quantized\nmatmul+add+relu" color=lightskyblue]

        qx -> int_op -> output
        qw -> int_op
        bias -> int_op
      }


The other difference is how the models get deployed. For XLA targets, models are
either served directly in Python, or exported as saved models. Quantization is
completely done in the framework. For ODML targets, models need to undergo
LiteRT conversion, which allows transforming the graph for quantization. The
transformation during the conversion is more powerful as it has access to the
whole graph and can perform propagation and fusion easily. However, the
framework must provide enough annotations in the graph for the converter, where
a protocol is needed between the framework and the ODML converter.

ODML quantization with Qwix
---------------------------

ODML quantization in Qwix is implemented by ``OdmlQatProvider`` and
``OdmlConversionProvider``. **Asymmetric**
:ref:`static-range quantization <srq>` is enabled by default for ODML
targets.

.. code-block:: python

    rules = [
        qwix.QuantizationRule(
            weight_qtype='int8',
            act_qtype='int8',
        )
    ]

ODML QAT
^^^^^^^^

The ``OdmlQatProvider`` is very similar to ``QatProvder`` as it also inserts
``FakeQuant`` op in the graph. The differences are

* ``OdmlQatProvider`` supports many more ops and actually should support every
    op in the model.
* ``OdmlQatProvider`` is aware of the fusion pattern and will skip inserting
    ``FakeQuant`` between e.g. matmul and add.

To ensure all ops are quantized, the ``OdmlQatProvider`` has a strict mode that
will raise an error if an unsupported op is detected.

.. tabs::

    .. tab:: Linen

        .. code-block:: python

            fp_model = SomeLinenModel(...)
            provider = qwix.OdmlQatProvider(rules, strict=True)
            qat_model = qwix.quantize_model(fp_model, provider)
            # qat_model can be trained as usual.

    .. tab:: NNX

        .. code-block:: python

            fp_model = SomeNNXModel(...)
            provider = qwix.OdmlQatProvider(rules, strict=True)
            qat_model = qwix.quantize_model(fp_model, provider, model_input)
            # qat_model can be trained as usual.

ODML conversion
^^^^^^^^^^^^^^^

After QAT, the ODML conversion can be achieved by applying the
``OdmlConversionProvider`` to the model. The ``OdmlConversionProvider`` takes two
more arguments, the ``params`` and the ``quant_stats``, because it needs to
calculate static scales for weights and activations during conversion.

.. tabs::

    .. tab:: Linen

        .. code-block:: python

            qat_variables = ...  # from QAT.
            params = qat_variables['params']
            quant_stats = qat_variables['quant_stats']

            conversion_provider = qwix.OdmlConversionProvider(rules, params, quant_stats)
            conversion_model = qwix.quantize_model(fp_model, conversion_provider)

    .. tab:: NNX

        .. note::
            NNX support for ODML modes is experimental. The API is not finalized.

        .. code-block:: python

            qat_model = ...  # from QAT.
            params = nnx.to_pure_dict(nnx.state(qat_model, nnx.Param))
            quant_stats = nnx.to_pure_dict(nnx.state(qat_model, qwix.QuantStat)),

            conversion_provider = qwix.OdmlConversionProvider(rules, params, quant_stats)
            conversion_model = qwix.quantize_model(fp_model, conversion_provider, model_input)


The model can then be converted and exported using
`Google AI Edge <https://ai.google.dev/edge>`_.

.. tabs::

    .. tab:: Linen

        .. code-block:: python

            import ai_edge_jax

            litert_model = ai_edge_jax.convert(
                conversion_model.apply,
                {'params': params},
                (model_input,),
                _litert_converter_flags={'_experimental_strict_qdq': True},  # necessary for Qwix.
            )

            # Evaluate the LiteRT model on the host.
            litert_result = litert_model(model_input)
            # Export the LiteRT model.
            litert_model.export('/tmp/litert_model.tflite')

    .. tab:: NNX

        .. code-block:: python

            import ai_edge_jax

            graphdef, state = nnx.split(conversion_model)
            litert_model = ai_edge_jax.convert(
                lambda params, *args: nnx.merge(graphdef, params)(*args),
                state,
                (model_input,),
                _litert_converter_flags={'_experimental_strict_qdq': True},  # necessary for Qwix.
            )

            # Evaluate the LiteRT model on the host.
            litert_result = litert_model(model_input)
            # Export the LiteRT model.
            litert_model.export('/tmp/litert_model.tflite')
