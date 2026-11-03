# ADMM-Q for Qwix

ADMM-Q as a calibration-based algorithm for [Qwix](https://github.com/google/qwix),
Google's JAX quantization library. It plugs in exactly like Qwix's GPTQ
(`qwix.contrib.gptq`): calibrate, transform the weights, serve with Qwix's
`PtqProvider`.

| File | Mirrors | Contents |
|---|---|---|
| `admmq_core.py` | `qwix/contrib/gptq_core.py` | `quantize_weight(W, H, how) -> (QArray, loss)`: the ADMM-Q solver on Qwix's grid |
| `admmq.py` | `qwix/contrib/gptq.py` | `AdmmqRule`, `AdmmqCalibrationProvider`, `quantize_params` |
| `admmq_test.py` | `qwix/tests/contrib/gptq_test.py` | core and end-to-end tests (Linen, NNX, einsum) |

`../solvers.py` (PyTorch) and `../solvers_jax.py` (standalone JAX) are unchanged.

## Usage

```python
import jax, jax.numpy as jnp, qwix
from qwix_admmq import admmq

rules = [admmq.AdmmqRule(module_path='.*', weight_qtype=jnp.int4)]

# 1. Calibration: collects X @ X.T per weight (same statistic as GPTQ).
cal_model = qwix.quantize_model(model, admmq.AdmmqCalibrationProvider(rules))
for batch in calibration_dataset:
    _, new_vars = cal_model.apply(variables, batch, mutable='quant_stats')
    variables.update(new_vars)

# 2. Weight transformation with ADMM-Q.
ptq_model = qwix.quantize_model(model, qwix.PtqProvider(rules))
abs_vars = jax.eval_shape(ptq_model.init, jax.random.key(0), model_input)
params = admmq.quantize_params(
    variables['params'], abs_vars['params'], variables['quant_stats'])

# 3. Serve with Qwix's PtqProvider, as for PTQ / GPTQ.
y = ptq_model.apply({'params': params}, model_input)
```

NNX models follow the NNX example in Qwix's `docs/source/gptq.md`, with
`admmq` in place of `gptq`.

## Behavior

- Same algorithm as `solvers_jax.ADMM_quantize`: clip-ratio scale search, ADMM
  with one float64 eigendecomposition per weight, a single guarded scale
  refresh when progress stalls, and PairSwap on ill-conditioned layers.
- Uses Qwix's grid and calibration, not SpinQuant's. For int4 the codes are
  in [-8, 7]. Symmetric scales are absmax / 7.5. Asymmetric (`minmax`) uses
  an integer zero point. Any `weight_calibration_method` and `tile_size` that
  Qwix supports also work here.
- Supports integer qtypes (`int4`, `int8`, `'int2'`…`'int7'`). Other qtypes
  (fp8, nf4, mx formats) fall back to Qwix's PTQ with a warning.
- float64 is enabled only inside `quantize_weight` (`jax.enable_x64`), not
  globally. Matmuls there also run at full float32 precision
  (`jax.default_matmul_precision('highest')`), because JAX otherwise allows
  TF32 on NVIDIA GPUs.

## Running the tests

```bash
pip install "flax>=0.12" absl-py   # plus jax
PYTHONPATH=/path/to/qwix:/path/to/admm-q-main python qwix_admmq/admmq_test.py
```

On Engaging login nodes, prefix the command with `taskset -c 0-3`. Otherwise
XLA tries to start one thread per core and hits the per-user thread limit.

## Note on Qwix's GPTQ

Qwix's `gptq_core.quantize_weight` (as of commit 67ff467) fails on asymmetric
rules. Line 158 evaluates `zero_point and zero_point.squeeze(1)`, which raises
"truth value of an array is ambiguous". The ADMM-Q tests therefore compare
against GPTQ on symmetric rules only.
