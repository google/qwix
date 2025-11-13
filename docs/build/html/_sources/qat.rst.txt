.. _quantization_aware_training:

Quantization-Aware Training (QAT)
===================================

.. NOTE::
    Don't mistake quantization-aware training (QAT) with quantized training (QT).

The goal of quantization-aware training is to emulate the numerics during
serving, but still keep the model to be trainable. It's usually an optional step
performed after the original floating-point training.

In quantization-aware training, weights are still kept floating-point. Both
weights and activations are quantized dynamically inside the ops.

Qwix implements QAT using fake quantization, where quantized ops are emulated
using floating-point ops and ``FakeQuant`` ops on the inputs. In ``FakeQuant`` op,
the array is quantized and then dequantized immediately. The output is
equivalent to the actual quantized output.

.. raw:: html

    <section style="text-align: center">

.. container:: flex-container

    .. graphviz::
        :class: "inline-block"

        digraph {
         graph [label="PTQ mode"]
         node [color="none" style="filled"]

         qw [label="quantize" color="burlywood1"]
         qx [label="quantize" color="burlywood1"]
         dq [label="dequantize" color="burlywood1"]
         int_op [color=lightskyblue]
         input -> qx -> int_op
         weight -> qw -> int_op
         int_op -> dq -> output
        }

    .. raw:: html

        <div style="display: inline; vertical-align: middle; font-size: 200%;">=</div>

    .. graphviz::
        :class: "inline-block"

        digraph {
         label="QAT mode"
         node [color="none" style="filled"]

         subgraph cluster_w {
           style=dashed label="FakeQuant" labelloc=b labeljust=r
           qw [label="quantize" color="burlywood1"]
           dqw [label="dequantize" color="burlywood1"]
         }
         subgraph cluster_x {
           style=dashed label="FakeQuant" labelloc=b labeljust=l
           qx [label="quantize" color="burlywood1"]
           dqx [label="dequantize" color="burlywood1"]
         }
         fp_op [color=lightskyblue]
         input -> qx -> dqx -> fp_op
         weight -> qw -> dqw -> fp_op
         fp_op -> output
        }

.. raw:: html

    </section>

During the backward pass of QAT, the ``FakeQuant`` op is bypassed, allowing
gradients to pass through.

Alternatively, it's possible to implement QAT with quantized ops directly, in
which case the backward pass also needs to be manually implemented. This
approach is called **quantized training** and is not implemented in Qwix yet.

QAT with Qwix
-------------

QAT in Qwix is implemented by ``QatProvider`` and can be applied to model with
``quantize_model``.

.. tabs::

    .. tab:: Linen

        .. code-block:: py

            fp_model = SomeLinenModel(...)
            qat_model = qwix.quantize_model(fp_model, qwix.QatProvider(rules))

    .. tab:: NNX

        .. code-block:: py

            fp_model = SomeNnxModel(...)
            qat_model = qwix.quantize_model(fp_model, qwix.QatProvider(rules), model_input)

Since QAT model still consumes floating-point weights, there's no need to
convert model variables and the checkpoints can be used interchangeably. All the
existing training code should also just work.

Static-Range Quantization
-------------------------

:ref:`Static-range quantization <srq>` adds extra complexity during QAT
because extra calibration data need to be collected. Those data are called
quantization statistics and are stored in ``quant_stats`` collection in Linen
models, or as ``QuantStat`` variables in NNX models.

.. tabs::

    .. tab:: Linen

        .. code-block:: py

            rules = [
                qwix.QuantizationRule(
                    weight_qtype='int8',
                    act_qtype='int8',
                    act_static_scale=True,
                )
            ]
            qat_model = qwix.quantize_model(model, qwix.QatProvider(rules))
            qat_model.init(jax.random.key(0), model_input)['quant_stats']

        The output will look like

        .. code-block:: none

            {'Dense_0': {'dot_general0_lhs': {'count': Array(0, dtype=int32),
              'sum_of_absmax': Array([[0.]], dtype=float32)}},
             'Dense_1': {'dot_general0_lhs': {'count': Array(0, dtype=int32),
              'sum_of_absmax': Array([[0.]], dtype=float32)}}}

    .. tab:: NNX

        .. code-block:: py

            rules = [
                qwix.QuantizationRule(
                    weight_qtype='int8',
                    act_qtype='int8',
                    act_static_scale=True,
                )
            ]
            qat_model = qwix.quantize_model(model, qwix.QatProvider(rules), model_input)
            qat_model.linear1.dot_general0_lhs

        The output will look like

        .. code-block:: none

            QuantStat( # 2 (8 B)
              value={'count': Array(0, dtype=int32), 'sum_of_absmax': Array([[0.]], dtype=float32)}
            )

Standalone calibration process
------------------------------

If QAT is not used but SRQ is enabled, it's necessary to perform a standalone
calibration process to collect quantization statistics. This can happen when the
training dataset is not available or there aren't enough resources to do the
training.

The standalone calibration process can be achieved by only running the forward
pass of the QAT model, where quantization statistics get updated.

Recommended practices
---------------------

It's recommended to start QAT from a existing floating-point model with good
quality rather than from randomly initialized weights, because

* QAT may lead to instability (NaN) if started from random weights.
* QAT is slower than non-QAT training and takes more steps to converge.
* QAT does not cause significant changes to the distribution of the weights,
  and it is relatively easy to converge.
* It allows us to measure the quality impact of QAT.

QAT usually uses a smaller learning rate to ensure stability.
