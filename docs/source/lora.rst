.. _applying_lora_qlora:

Applying LoRA/QLoRA
===================

Qwix also implements a ``LoraProvider`` that can apply LoRA and QLoRA to models
implicitly, based on the existing infrastructure on model surgery.

.. tabs::

    .. tab:: Linen

        .. code-block:: python

            rules = [
                qwix.LoraRule(
                    weight_qtype='nf4',
                    rank=16,
                    alpha=0.5,
                )
            ]
            lora_model = qwix.apply_lora_to_model(model, qwix.LoraProvider(lora_rules))

        .. code-block:: py
           :class: no-copybutton

            >>> jax.eval_shape(lora_model.init, jax.random.key(0), model_input)['params']
            {'Dense_0': {'kernel': QArrayWithAux(array=QArray(qvalue=ShapeDtypeStruct(shape=(16, 64), dtype=uint4), scale=ShapeDtypeStruct(shape=(1, 64), dtype=float32), zero_point=None, qtype='nf4'), ...),
             'kernel_lora_a': ShapeDtypeStruct(shape=(16, 16), dtype=float32),
             'kernel_lora_b': ShapeDtypeStruct(shape=(16, 64), dtype=float32)},
             'Dense_1': {'kernel': QArrayWithAux(array=QArray(qvalue=ShapeDtypeStruct(shape=(64, 16), dtype=uint4), scale=ShapeDtypeStruct(shape=(1, 16), dtype=float32), zero_point=None, qtype='nf4'), ...),
             'kernel_lora_a': ShapeDtypeStruct(shape=(64, 16), dtype=float32),
             'kernel_lora_b': ShapeDtypeStruct(shape=(16, 16), dtype=float32)}}

    .. tab:: NNX

        .. code-block:: python

            rules = [
                qwix.LoraRule(
                    weight_qtype='nf4',
                    rank=16,
                    alpha=0.5,
                )
            ]
            lora_model = qwix.apply_lora_to_model(model, qwix.LoraProvider(rules), model_input)

        .. code-block:: py
           :class: no-copybutton

            >>> jax.eval_shape(nnx.to_pure_dict, nnx.state(lora_model))
            {'linear1': {'kernel': {'array': {'qvalue': ShapeDtypeStruct(shape=(16, 64), dtype=uint4),
               'scale': ShapeDtypeStruct(shape=(1, 64), dtype=float32)}},
             'kernel_lora_a': ShapeDtypeStruct(shape=(16, 16), dtype=float32),
             'kernel_lora_b': ShapeDtypeStruct(shape=(16, 64), dtype=float32)},
             'linear2': {'kernel': {'array': {'qvalue': ShapeDtypeStruct(shape=(64, 16), dtype=uint4),
               'scale': ShapeDtypeStruct(shape=(1, 16), dtype=float32)}},
             'kernel_lora_a': ShapeDtypeStruct(shape=(64, 16), dtype=float32),
             'kernel_lora_b': ShapeDtypeStruct(shape=(16, 16), dtype=float32)}}
