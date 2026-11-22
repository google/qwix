# MXFP quantization (mxfp8 / mxfp4 / nvfp4)

Qwix supports microscaled floating-point (MXFP) weight/activation qtypes.
Set `weight_qtype` / `act_qtype` in a `QtRule` or PTQ config to one of:
`mxfp8`, `mxfp4`, `nvfp4`.

## Block / tile sizes

- `mxfp8`, `mxfp4`: block size 32 (per-32-element scale).
- `nvfp4`: block size 16.

## Scale format

- `mxfp8` / `mxfp4`: `float8_e8m0fnu` block scales.
- `nvfp4`: `float8_e4m3fn` block scales.

## Hardware dispatch

`dot_general` on MXFP operands dispatches by platform:

| Platform | Path | Notes |
|----------|------|-------|
| GPU (Blackwell) | Fused `jax.nn.scaled_matmul` (cuDNN) | `mxfp_dot._gpu_mxfp_dot` |
| GPU (legacy) | `scaled_matmul` emulated/decomposed | |
| TPU / CPU | Native tiled fp8 `dot_general` | `mxfp_dot_general` returns `None` → fallback to `_fast_dot_general` |

When `mxfp_dot_general` returns `None` (non-GPU platforms), the caller
falls through to the native quantized `dot_general` path, which is correct
and MXU-accelerated on TPU.

## Performance note

The fp8 matmul itself is fast; the activation-quantization + cast plumbing
around it can dominate in full forward passes. Weight caching (PTQ: quantize
the weight once) amortizes weight-quant cost across reuses, but does not
amortize per-call activation quantization. Expect fp8 gains primarily in
large GEMMs with high weight reuse; a full small-model forward may not
benefit at current XLA lowering.
