.. _get-started:

Get Started
=================================

This guide will demonstrate how to apply post-training quantization to a simple MLP model.

.. tabs::

   .. tab:: Linen

      .. code-block:: python

         import jax
         from flax import linen as nn

         class MLP(nn.Module):

           dhidden: int
           dout: int

           @nn.compact
           def __call__(self, x):
             x = nn.Dense(self.dhidden, use_bias=False)(x)
             x = nn.relu(x)
             x = nn.Dense(self.dout, use_bias=False)(x)
             return x

         model = MLP(64, 16)
         model_input = jax.random.uniform(jax.random.key(0), (8, 16))

   .. tab:: NNX

      .. code-block:: python

         import jax
         from flax import nnx

         class MLP(nnx.Module):

           def __init__(self, din, dhidden, dout, *, rngs: nnx.Rngs):
             self.linear1 = nnx.Linear(din, dhidden, use_bias=False, rngs=rngs)
             self.linear2 = nnx.Linear(dhidden, dout, use_bias=False, rngs=rngs)

           def __call__(self, x):
             x = self.linear1(x)
             x = nnx.relu(x)
             x = self.linear2(x)
             return x

         model = MLP(16, 64, 16, rngs=nnx.Rngs(0))
         model_input = jax.random.uniform(jax.random.key(0), (8, 16))

..

Since Qwix is able to quantize the whole model implicitly, there's no need to
modify the model code. The above model can also be substituted with any other
Linen/NNX models.

Quantization config
-----------------------

Qwix uses a regex-based configuration system to instruct how to quantize a Jax
model. Configurations are defined as a list of ``QuantizationRule``. Each rule
consists of a key that matches Flax modules, and a set of values that control
quantization behavior.

For example, to quantize the above model in int8 (w8a8), we need to define the
rules as below.

.. code-block:: python

   import qwix

   rules = [
       qwix.QuantizationRule(
           module_path='.*',  # this rule matches all modules.
           weight_qtype='int8',  # quantizes weights in int8.
           act_qtype='int8',  # quantizes activations in int8.
       )
   ]

Unlike some other libraries that provides limited number of **quantization
recipes**, Qwix doesn't have a list of presets. Instead, different quantization
schemas are achieved by combinations of quantization configs. For a full list of
available options, please check the
:py:class:`QuantizationRule <qwix.qconfig.QuantizationRule>`.

Apply quantization
-----------------------

With the above code, applying quantization is as simple as one line.

.. tabs::

   .. tab:: Linen

      .. code-block:: python

         ptq_model = qwix.quantize_model(model, qwix.PtqProvider(rules))

   .. tab:: NNX

      .. code-block:: python

         ptq_model = qwix.quantize_model(model, qwix.PtqProvider(rules), model_input)

..

We could inspect the params to verify that weights are now pre-quantized.

.. tabs::

   .. tab:: Linen

      .. code-block:: pycon

         >>> jax.eval_shape(ptq_model.init, jax.random.key(0), model_input)['params']
         {
           'Dense_0': {
             'kernel': WithAux(
               array=TransposedQArray(
                 qvalue=ShapeDtypeStruct(shape=(16, 64), dtype=int8),
                 scale=ShapeDtypeStruct(shape=(1, 64), dtype=float32),
                 ...
               ),
               ...
             )
           },
           'Dense_1': {
             'kernel': WithAux(
               array=TransposedQArray(
                 qvalue=ShapeDtypeStruct(shape=(64, 16), dtype=int8),
                 scale=ShapeDtypeStruct(shape=(1, 16), dtype=float32),
                 ...
               ),
               ...
             )
           }
         }

   .. tab:: NNX

      .. code-block:: pycon

         >>> jax.eval_shape(nnx.to_pure_dict, nnx.state(ptq_model))
         {
           'linear1': {
             'kernel': {
               'array': {
                 'qvalue': ShapeDtypeStruct(shape=(16, 64), dtype=int8),
                 'scale': ShapeDtypeStruct(shape=(1, 64), dtype=float32)
               }
             }
           },
           'linear2': {
             'kernel': {
               'array': {
                 'qvalue': ShapeDtypeStruct(shape=(64, 16), dtype=int8),
                 'scale': ShapeDtypeStruct(shape=(1, 16), dtype=float32)
               }
             }
           }
         }

..

Quantization providers
-----------------------

You may notice that we initialized a ``PtqProvider`` object above and applied it
to the model. ``PtqProvider`` implements ``QuantizationProvider`` interface, which
is a powerful abstraction that allows different quantization modes being
implemented and consumed in a consistent way.

Qwix ships with the following providers.

* :doc:`QAT provider <qat>`
* :doc:`PTQ provider <ptq>`
* :doc:`ODML provider <odml>`
* :doc:`LoRA/QLoRA provider <lora>`

It's also possible to implement your own provider by subclassing existing ones,
which is perfect for researchers to :doc:`explore novel quantization algorithms <extend>`.
