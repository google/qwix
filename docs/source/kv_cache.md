# KV Cache Quantization

During long-context or large-batch LLM inference, the Key-Value (KV) cache often
consumes more High Bandwidth Memory (HBM) than the model weights themselves, and
autoregressive decoding becomes bottlenecked by reading cached keys and values
from HBM at every token step.

Quantizing the KV cache to `int8`, `fp8` (`jnp.float8_e4m3fn`), or `int4` cuts
cache memory by **2x-4x** and reduces HBM traffic during decoding—enabling
larger batch sizes, longer context windows, and faster token generation.

## How KV Cache Quantization Works in Qwix

Unlike static model weights (which live in `params` and are automatically
converted by `PtqProvider`), a KV cache is **dynamic runtime state** allocated,
sliced, and updated across decoding steps by your model or serving engine.

To work naturally with any cache manager or serving loop, Qwix provides three
composable building blocks:

1.  **`qwix.quantize`**: Quantizes newly projected `K` and `V` tensors before
    writing them into the cache.
2.  **`qwix.QArray`**: A native JAX PyTree container holding low-precision
    `qvalue` and floating-point `scale` arrays.
3.  **`qwix.einsum` / `qwix.dot_general`**: Drop-in replacements for
    `jnp.einsum` and `jax.lax.dot_general` that accept mixed
    `(jax.Array, QArray)` inputs and compute attention directly from low-bit
    caches.

## Quickstart: Two Integration Patterns

Choose the pattern that matches how your model or serving engine manages cache
state.

### Pattern 1: Store `QArray` Directly in the Cache

Because `qwix.QArray` is registered as a standard JAX PyTree (and Flax NNX data
type), you can pass it directly through `jax.jit`, `jax.lax.scan`,
`jax.lax.while_loop`, or store it in a cache dictionary.

```py
import jax
import jax.numpy as jnp
import qwix

# Batch-major cache shape: [batch, seq_len, num_kv_heads, head_dim]
B, S_max, K, D = 4, 2048, 8, 128

# 1. Allocate an int8 KV cache with per-token, per-head bf16 scales.
cache_k = qwix.QArray(
    qvalue=jnp.zeros((B, S_max, K, D), dtype=jnp.int8),
    scale=jnp.ones((B, S_max, K, 1), dtype=jnp.bfloat16),
)
cache_v = qwix.QArray(
    qvalue=jnp.zeros((B, S_max, K, D), dtype=jnp.int8),
    scale=jnp.ones((B, S_max, K, 1), dtype=jnp.bfloat16),
)


def decode_step(q_step, k_step, v_step, cache_k, cache_v, step_idx):
  # 2. Quantize the new K and V slices along head_dim (axis 3).
  #    channelwise_axes=(0, 1, 2) keeps an independent scale per
  #    (batch, token, kv_head).
  k_q = qwix.quantize(
      k_step,
      jnp.int8,
      channelwise_axes=(0, 1, 2),
      scale_dtype=jnp.bfloat16,
  )
  v_q = qwix.quantize(
      v_step,
      jnp.int8,
      channelwise_axes=(0, 1, 2),
      scale_dtype=jnp.bfloat16,
  )

  # 3. Write quantized values and scales into the cache at step_idx.
  cache_k = qwix.QArray(
      qvalue=jax.lax.dynamic_update_slice(
          cache_k.qvalue, k_q.qvalue, (0, step_idx, 0, 0)
      ),
      scale=jax.lax.dynamic_update_slice(
          cache_k.scale, k_q.scale, (0, step_idx, 0, 0)
      ),
  )
  cache_v = qwix.QArray(
      qvalue=jax.lax.dynamic_update_slice(
          cache_v.qvalue, v_q.qvalue, (0, step_idx, 0, 0)
      ),
      scale=jax.lax.dynamic_update_slice(
          cache_v.scale, v_q.scale, (0, step_idx, 0, 0)
      ),
  )

  # 4. Compute decode attention directly on QArray caches via qwix.einsum.
  #    q_step shape: [B, 1, K, G, D] (grouped-query attention)
  attn_logits = qwix.einsum('btkgd,bskd->bkgts', q_step, cache_k)
  attn_probs = jax.nn.softmax(attn_logits.astype(jnp.float32), axis=-1).astype(
      q_step.dtype
  )
  attn_out = qwix.einsum('bkgts,bskd->btkgd', attn_probs, cache_v)
  return attn_out, cache_k, cache_v
```

### Pattern 2: Store Separate Buffers and Wrap on Read

If your serving engine already manages raw `jax.Array` cache buffers by name
(for example, pre-allocating `nnx.Cache` variables or host-device KV pools),
store `qvalue` and `scale` as two ordinary arrays and wrap them into
`qwix.QArray(cached_val, cached_scale)` when reading the cache:

```py
# On cache write: unpack qvalue and scale into your existing buffer manager.
kv_q = qwix.quantize(
    k_step,
    jnp.float8_e4m3fn,
    channelwise_axes=(0, 1, 2),
    scale_dtype=jnp.bfloat16,
)
raw_k_qvalue, raw_k_scale = kv_q.qvalue, kv_q.scale

# On cache read: wrap into a QArray (zero-copy) and pass to qwix.einsum.
k_cached = qwix.QArray(qvalue=cached_k_buf, scale=cached_k_scale_buf)
attn_logits = qwix.einsum('btkgd,bskd->bkgts', q_step, k_cached)
```

## Choosing Precision and Granularity

*   **Precision (`qtype`)**:
    *   `jnp.int8`: Recommended default across all TPU generations and GPUs.
        Cuts KV cache memory by ~2x with high numerical fidelity—speeding up
        bandwidth-bound decoding (such as on TPU v5e and v6e) while matching or
        beating `bfloat16` step latency on higher-bandwidth `TPU7x`.
    *   `jnp.float8_e4m3fn`: Cuts KV cache memory by ~2x while providing wider
        floating-point dynamic range for outlier-heavy K/V distributions. Works
        best on hardware with native FP8 conversion support (such as `TPU7x` and
        NVIDIA Hopper/Blackwell GPUs; on TPU v5e and v6e, prefer `jnp.int8`).
    *   `jnp.int4`: Cuts KV cache memory by ~4x to maximize batch size or
        context length under tight HBM budgets, trading off higher quantization
        noise than 8-bit formats. Pair with per-token, per-head scaling.
*   **Scale granularity (`channelwise_axes`)**:
    *   **Per-token, per-head** (recommended): For shape `[B, S, K, D]`, pass
        `channelwise_axes=(0, 1, 2)` so each `(batch, token, head)` vector of
        length `D` gets its own scale (`scale` shape `[B, S, K, 1]`). With
        `D=128` and `bfloat16` scales, scale metadata adds ~1.6% memory overhead
        for 8-bit caches (~3.1% for `int4`). Stick with symmetric calibration
        (the default `absmax`) and avoid `tiled_axes` along `head_dim` so both
        attention matmuls stay on Qwix's zero-materialization paths.
    *   **Per-token across heads**: Pass `channelwise_axes=(0, 1)` (`scale`
        shape `[B, S, 1, 1]`) for an even smaller scale footprint.
*   **Scale dtype (`scale_dtype`)**:
    *   Use `scale_dtype=jnp.bfloat16` for `bfloat16` activations (or
        `jnp.float32` for `float32` activations) to preserve full exponent range
        and keep `P @ V` on the scale-folding fast path without promoting
        `bfloat16` outputs to `float32`.

## How Qwix Accelerates Decode

In autoregressive attention, the two KV matrix multiplications contract along
different axes of the cached tensor `[B, S, K, D]` (with scale `[B, S, K, 1]`):

1.  **Score computation (`Q @ K^T`, contracting `D`)**: Because `K.scale` has
    size `1` along `D`, `qwix.einsum` computes `Q @ K.qvalue` directly and
    rescales the resulting attention logits by `K.scale` (**output scaling**).
2.  **Value aggregation (`P @ V`, contracting `S`)**: Because `V.scale` varies
    along `S` while having size `1` along `D`, `qwix.einsum` folds `V.scale`
    into the attention weights `(P * V.scale) @ V.qvalue` (**scale folding**)
    before the matmul.

Both paths stream compact low-bit `qvalue` buffers directly from HBM without
materializing dequantized intermediate caches, allowing XLA to fuse decoding
end-to-end on TPU and GPU.

## Case Study: MaxText

To see how Pattern 2 fits into a full LLM framework, consider quantizing the
autoregressive KV cache in
[MaxText](https://github.com/AI-Hypercomputer/maxtext), Google's open-source
JAX/Flax NNX LLM codebase. Because MaxText manages pre-allocated `nnx.Cache`
buffers across prefill and decode, adding `int8`, `int4`, and `fp8` KV cache
quantization with Qwix takes three localized steps without changing cache
ownership or prefill kernels:

1.  **Full-precision prefill, quantized cache write**: During prefill,
    full-precision `K` and `V` are already in on-chip memory, so prefill
    attention runs in full precision across all attention kernels while
    `qwix.quantize` writes quantized value and scale tensors into MaxText's
    separate prefill `nnx.Cache` buffers (with autoregressive decode steps
    similarly updating separate `cached_ar_*` buffers).
2.  **Zero-copy `QArray` assembly on read**: During autoregressive decode,
    reading the cache wraps the stored value and scale buffers into
    `qwix.QArray(value, scale)` and uses `jax.tree.map` to transpose both fields
    together.
3.  **Unified `kv_einsum` dispatch**: A short helper checks
    `isinstance(rhs, qwix.QArray)`, routing quantized decode caches to
    `qwix.einsum` and full-precision tensors to `jnp.einsum` (see
    [Integration Tips](kv_cache.md#kv-cache-integration-tips) below for scoping
    model-wide quantization rules).

Benchmarked in MaxText on single-chip TPU v5e (`head_dim=128`, `kv_heads=8`,
context length `2048`, per-token scaling across heads), Qwix KV cache
quantization speeds up the autoregressive decode step by **1.5x-1.6x** with
`int8`/`fp8` and **2.0x-2.2x** with `int4` at batch sizes 32-64, while reducing
KV cache HBM footprint by 2x-4x.

(kv-cache-integration-tips)=

## Integration Tips

*   **Combining with `PtqProvider` or `QtProvider`**: Weight-only rules
    (`act_qtype=None`) compose directly with `qwix.einsum` even under a
    catch-all `module_path='.*'`. When enabling activation quantization
    (`act_qtype` set), scope `QuantizationRule.module_path` to your linear
    projections (for example, `r'.*(q_proj|k_proj|v_proj|o_proj|mlp).*'`) so
    activation rules apply to projection layers while `qwix.einsum` handles the
    attention dot products.
*   **Batch-major cache layout**: Store persistent KV caches in a batch-major
    axis order such as `[B, K, S, D]` or `[B, S, K, D]` (keeping `batch` outside
    `seq_len`) to avoid cross-lane relayout copies across `jax.jit` boundaries.
*   **Distributed SPMD sharding**: Because `kv.scale` has size `1` along
    quantized axes (such as `head_dim`), derive a companion `scale`
    `PartitionSpec` that sets any singleton axis to `None`, and apply
    `jax.device_put`, `jax.lax.with_sharding_constraint` (under
    `AxisType.Auto`), or `jax.sharding.reshard` (under `AxisType.Explicit`) to
    `kv.qvalue` and `kv.scale` separately.
