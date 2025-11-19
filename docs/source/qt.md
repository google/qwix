# Training with Quantization (QAT/QT)

`Qwix` supports two primary methods for training models with quantization:
**Quantization-Aware Training (QAT)** and **Quantized Training (QT)**. While
related, they serve different purposes.

*   **Quantization-Aware Training (QAT)** aims to make the model aware of the
    precision loss that will occur during inference, which helps to recover the
    model quality degradation during PTQ. In Qwix, this is achieved by using
    low-precision integer operations in the forward pass, which introduces
    quantization noise, while using standard floating-point operations for the
    backward pass (via a Straight-Through Estimator). This allows the model's
    weights to adapt to the noise, improving final quantized accuracy, while
    maintaining stable training dynamics with full-precision gradients. Compared
    to fake quantization -- another common way of implementing QAT -- Qwix's QAT
    implementation produces the same numerics and is more performant.

*   **Quantized Training (QT)** goes a step further by performing computations
    using low-precision integer arithmetic in both the **forward and backward
    passes**. The operations themselves are quantized, providing a more
    performant usage of hardware behavior during the entire training process.

QAT only quantizes the forward pass, while QT quantizes both forward and
backward passes.

```{graphviz}
:align: center
digraph {
  graph [label="QAT mode"]
  node [color="none" style="filled"]

  qw [label="quantize" color="burlywood1"]
  qx [label="quantize" color="burlywood1"]
  dq [label="dequantize" color="burlywood1"]
  bwd [label="fp bwd\n(STE)" color="lightpink"]
  int_op [label="int_op" color=lightskyblue]

  input -> qx -> int_op
  weight -> qw -> int_op
  int_op -> dq -> output

  // Backward pass computes float gradients
  int_op -> bwd -> {fp_dlhs, fp_drhs}
}
```

```{graphviz}
:align: center
digraph {
  graph [label="QT mode"]
  node [color="none" style="filled"]

  qw [label="quantize" color="burlywood1"]
  qx [label="quantize" color="burlywood1"]
  dq [label="dequantize" color="burlywood1"]
  bwd [label="quantized\nbwd" color="lightpink"]
  int_op [label="int_op" color=lightskyblue]
  input -> qx -> int_op
  weight -> qw -> int_op
  int_op -> bwd -> {int_dlhs int_drhs}
  int_op -> dq -> output
}
```

## Training with `Qwix`

In `Qwix`, both QAT and QT are conveniently handled by the `qwix.QtProvider`.
The behavior is controlled by the `bwd_qtype` parameter in the `qwix.QtRule`.

*   **To enable QT**, set `QtRule.bwd_qtype` to a specific data type (e.g.,
    `'int8'`).
*   **To enable QAT**, set `QtRule.bwd_qtype` to `None` (which is the default)
    or use `QuantizationRule`, which doesn't expose `bwd_qtype`.

`````{tabs}
````{tab} Linen
```py
fp_model = SomeLinenModel(...)

# For Quantized Training (QT), set bwd_qtype.
qt_rules = [
    qwix.QtRule(
        weight_qtype='int8',
        act_qtype='int8',
        bwd_qtype='int8',
    )
]
qt_model = qwix.quantize_model(fp_model, qwix.QtProvider(qt_rules))

# For Quantization-Aware Training (QAT), leave bwd_qtype as None.
qat_rules = [
    qwix.QuantizationRule(
        weight_qtype='int8',
        act_qtype='int8',
    )
]
qat_model = qwix.quantize_model(fp_model, qwix.QtProvider(qat_rules))
```
````
````{tab} NNX
```py
fp_model = SomeNnxModel(...)

# For Quantized Training (QT), set bwd_qtype.
qt_rules = [
    qwix.QtRule(
        weight_qtype='int8',
        act_qtype='int8',
        bwd_qtype='int8',
    )
]
qt_model = qwix.quantize_model(fp_model, qwix.QtProvider(qt_rules), model_input)

# For Quantization-Aware Training (QAT), leave bwd_qtype as None.
qat_rules = [
    qwix.QuantizationRule(
        weight_qtype='int8',
        act_qtype='int8',
    )
]
qat_model = qwix.quantize_model(fp_model, qwix.QtProvider(qat_rules), model_input)
```
````
`````

Since QAT/QT models still consume floating-point weights, there's no need to
convert model variables and the checkpoints can be used interchangeably. All the
existing training code should also just work.

## Static-Range Quantization

[Static-range quantization](basics.md#srq) adds extra complexity during QT
because extra calibration data need to be collected. Those data are called
quantization statistics and are stored in `quant_stats` collection in Linen
models, or as `QuantStat` variables in NNX models.

`````{tabs}
````{tab} Linen
```py
rules = [
    qwix.QtRule(
        weight_qtype='int8',
        act_qtype='int8',
        act_static_scale=True,
    )
]
qt_model = qwix.quantize_model(model, qwix.QtProvider(rules))
qt_model.init(jax.random.key(0), model_input)['quant_stats']
```

The output will look like

```none
{'Dense_0': {'dot_general0_lhs': {'count': Array(0, dtype=int32),
   'sum_of_absmax': Array([[0.]], dtype=float32)}},
 'Dense_1': {'dot_general0_lhs': {'count': Array(0, dtype=int32),
   'sum_of_absmax': Array([[0.]], dtype=float32)}}}
```
````
````{tab} NNX
```py
rules = [
    qwix.QtRule(
        weight_qtype='int8',
        act_qtype='int8',
        act_static_scale=True,
    )
]
qt_model = qwix.quantize_model(model, qwix.QtProvider(rules), model_input)
qt_model.linear1.dot_general0_lhs
```

The output will look like

```none
QuantStat( # 2 (8 B)
  value={'count': Array(0, dtype=int32), 'sum_of_absmax': Array([[0.]], dtype=float32)}
)
```
````
`````

### Standalone calibration process

If QT is not used but SRQ is enabled, it's necessary to perform a standalone
calibration process to collect quantization statistics. This can happen when the
training dataset is not available or there aren't enough resources to do the
training.

The standalone calibration process can be achieved by only running the forward
pass of the QT model, where quantization statistics get updated.

## Recommended Practices

It's recommended to start QT from an existing, high-quality floating-point model
rather than from randomly initialized weights.

*   Stability: Low-precision training can be unstable (leading to NaN values) if
    started from random weights.
*   Efficiency: Fine-tuning is typically faster and requires fewer steps to
    converge than training from scratch.
*   Evaluation: Starting with a strong baseline allows you to accurately measure
    the quality impact of quantization.

When fine-tuning, it is common to use a smaller learning rate to maintain
stability.
