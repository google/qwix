.. _quantization-basics:

Quantization Basics
===================

Quantization is the process to reduce the precision of the data types used in
machine learning models. It involves mapping continuous or high-precision values
(fp32 or bf16) to a smaller, discrete set of low-precision values (int8, fp8, or
lower). Quantization can be applied to model params (weights and biases) as well
as activations.

The primary goals of quantization are:

* **Reduce Model Size**: Lower-precision numbers require less memory or
  storage space. An int8 model is roughly 4x smaller than its fp32
  counterpart. This allows models to be fit in devices with limited memory.
* **Increase Inference Speed**: Computations with integers are often
  significantly faster on many hardware platforms (CPUs, GPUs, specialized
  accelerators like TPUs/NPUs) than floating-point computations.

Applying quantization is a trade-off since it has a negative impact on the model
quality. Fortunately, with various techniques such as QAT, we could usually
preserve the majority of the model capability.

Categories
----------

Quantization algorithms can be generally divided into the following 3 categories
from the least aggressive to the most. The more aggressive quantization brings
more performance gains and more quality loss.

.. _weight-only:

Weight-only quantization
^^^^^^^^^^^^^^^^^^^^^^^^

As the name suggests, this technique focuses only on quantizing the model's
weights to a lower precision format. The activations remain in their original
higher precision.

Since weights are static during inference, they can be quantized offline ahead
of time. If the hardware supports mixed precisions of the inputs, quantized
weights can be consumed directly. Otherwise, they are dequantized on the fly
during inference.

.. graphviz::

   digraph {
     graph [label="Weight-only quantization"]
     node [color="none" style="filled"]

     qw [label="quantize" color="burlywood1"]
     dq [label="dequantize" color="burlywood1"]
     fp_op [color=lightskyblue]

     subgraph cluster_offline {
       label="Offline" style=dashed labeljust=l
       weight -> qw
     }
     qw -> fp_op
     qw -> dq [label=scale style=dashed]
     input -> fp_op
     fp_op -> dq -> output
   }

The model's FLOP doesn't get fewer since the computation is still in
floating-point. However, performance could still get improved if the memory
bandwidth is the bottleneck. The reduced memory footprint also allows larger
batch size or fewer model shardings.

In Qwix, weight-only quantization can be enabled by setting ``act_qtype`` to
``None``.

.. code-block:: python

   # This enables int8 weight-only quantization.
   qwix.QuantizationRule(
       weight_qtype="int8",
       act_qtype=None,
   )

.. _drq:

Dynamic-Range Quantization (DRQ)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This technique quantizes the weights offline (similar to weight-only), and also
quantizes the activations dynamically during the inference process. The
quantized types of weights and activations don't need to be the same.

A activation is quantized using a dynamic scale calculated from the activation
itself, which is optimal. This means there's an additional cost of calibrating
the activation to get the scale. If the activation is sharded, this could
trigger a cross-device collective which can be expensive. Subchannel
quantization is usually necessary in this case to mitigate the collective.

.. graphviz::

   digraph {
     graph [label="Dynamic-Range Quantization"]
     node [color="none" style="filled"]

     qw [label="quantize" color="burlywood1"]
     qx [label="quantize" color="burlywood1"]
     dq [label="dequantize" color="burlywood1"]
     int_op [color=lightskyblue]

     subgraph cluster_offline {
       label="Offline" style=dashed labeljust=l
       weight -> qw
     }
     qw -> int_op
     input -> qx -> int_op
     qw -> dq [label=scale style=dashed]
     qx -> dq [label=scale style=dashed]
     int_op -> dq -> output
   }

In Qwix, DRQ can be enabled by setting ``act_qtype`` to a non-None value.

.. code-block:: python

   # This uses int4 for weights and int8 for activation.
   qwix.QuantizationRule(
       weight_qtype="int8",
       act_qtype="int8",
   )

.. _srq:

Static-Range Quantization (SRQ)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This technique is similar to dynamic-range quantization that both weights and
activations are quantized. The difference is activations are quantized using
static scales. This requires additional calibration data to be collected during
QAT or a standalone calibration process.

SRQ has the best performance and is usually required for ODML targets. It also
has the most significant impact on the model's quality.

.. graphviz::

   digraph {
     graph [label="Static-Range Quantization"]
     node [color="none" style="filled"]

     qw [label="quantize" color="burlywood1"]
     qx [label="quantize (static)" color="burlywood1"]
     dq [label="dequantize" color="burlywood1"]
     int_op [color=lightskyblue]

     subgraph cluster_offline {
       label="Offline" style=dashed labeljust=l
       static_scale [label="static scale"]
       weight -> qw
     }
     static_scale -> qx [style=dashed]
     input -> qx -> int_op
     qw -> int_op
     qw -> dq [label=scale style=dashed]
     static_scale -> dq [style=dashed]
     int_op -> dq -> output
   }

In Qwix, SRQ can be enabled by setting both the ``act_qtype`` and
``act_static_scale``.

.. code-block:: python

   # This enables int8 SRQ.
   qwix.QuantizationRule(
       weight_qtype="int8",
       act_qtype="int8",
       act_static_scale=True,
   )
