# 注意力后端特性支持 (Attention Backend Feature Support)

本文档由 `tools/pre_commit/generate_attention_backend_docs.py` 自动生成。
它展示了基于 `AttentionBackend.validate_configuration()` 校验的每个已注册注意力后端的特性支持情况。

**请勿手动编辑此文件。** 运行以下命令可重新生成此文件：

```bash
python tools/pre_commit/generate_attention_backend_docs.py
```

## 设置注意力后端 (Setting the Attention Backend)

### 命令行方式 (Command Line)

有两种方式可以通过命令行指定后端：

**选项 1：使用 `--attention-backend`（简便方式）**

```bash
vllm serve <model> --attention-backend FLASH_ATTN
```

**选项 2：使用 `--attention-config.backend` / `-ac.backend`（结构化配置）**

```bash
# 点号记法
vllm serve <model> --attention-config.backend FLASH_ATTN
vllm serve <model> -ac.backend FLASH_ATTN

# JSON 格式
vllm serve <model> --attention-config '{"backend": "FLASH_ATTN"}'
vllm serve <model> -ac '{"backend": "FLASH_ATTN"}'
```

> **注意：** `--attention-backend` 与 `--attention-config.backend` 是互斥的。请使用其中一种，不要同时使用。

### Python API

将 `AttentionConfig` 与 `LLM` 类配合使用：

```python
from vllm import LLM
from vllm.config import AttentionConfig
from vllm.v1.attention.backends.registry import AttentionBackendEnum

# 方法 1：使用带有枚举类型的 AttentionConfig
llm = LLM(
    model="Qwen/Qwen3-0.6B",
    attention_config=AttentionConfig(backend=AttentionBackendEnum.FLASH_ATTN),
)

# 方法 2：使用带有字符串的 attention_backend 参数
llm = LLM(
    model="Qwen/Qwen3-0.6B",
    attention_backend="FLASH_ATTN",
)
```

## 后端选择行为 (Backend Selection Behavior)

### 手动选择 (Manual Selection)

当您显式通过 `--attention-backend` 或 `AttentionConfig` 设置后端时：

1. 系统会根据您的配置（模型数据类型 dtype、头大小 head size、计算能力 compute capability 等）对后端进行**校验（Validated）**。
2. 如果该后端**不支持**您的配置，系统将抛出错误并列出具体原因。
3. 如果校验通过，则使用该后端。

选择不兼容后端时的错误示例：

```text
ValueError: Selected backend FLASHMLA is not valid for this configuration.
Reason: ['compute capability not supported']
```

### 自动选择 (Automatic Selection)

当未指定任何后端时（默认情况）：

1. vLLM 按照**优先级顺序**遍历所有后端（见下表）。
2. 根据您的配置对每个后端进行校验。
3. 选择**第一个兼容的后端**。
4. 如果没有兼容的后端，系统将抛出错误，并列出所有后端及其不兼容的具体原因。

## 后端优先级 - CUDA (Backend Priority)

当没有显式选择后端时，vLLM 从这些按优先级排序的列表中选择第一个兼容的后端。

优先级 **1 = 最高**（最先尝试）。

### 标准注意力机制 (Standard Attention: MHA, MQA, GQA)

**Blackwell (SM 10.x):**

| 优先级 | 后端 (Backend) |
| -------- | ------- |
| 1 | `FLASHINFER` |
| 2 | `FLASH_ATTN` |
| 3 | `TRITON_ATTN` |
| 4 | `FLEX_ATTENTION` |
| 5 | `TURBOQUANT` |

**Ampere/Hopper (SM 8.x-9.x):**

| 优先级 | 后端 (Backend) |
| -------- | ------- |
| 1 | `FLASH_ATTN` |
| 2 | `FLASHINFER` |
| 3 | `TRITON_ATTN` |
| 4 | `FLEX_ATTENTION` |
| 5 | `TURBOQUANT` |

### MLA 注意力机制 (DeepSeek-style)

**Blackwell (SM 10.x):**

| 优先级 | 后端 (Backend) |
| -------- | ------- |
| 1 | `FLASHINFER_MLA` |
| 2 | `TOKENSPEED_MLA` |
| 3 | `CUTLASS_MLA` |
| 4 | `FLASH_ATTN_MLA` |
| 5 | `FLASHMLA` |
| 6 | `TRITON_MLA` |
| 7 | `FLASHINFER_MLA_SPARSE`**\*** |
| 8 | `FLASHMLA_SPARSE` |

> **\*** 对于稀疏 MLA，FP8 KV 缓存总是首选 `FLASHINFER_MLA_SPARSE`。对于 BF16 KV 缓存，在 Query Head 数量较少（<= 16）时首选 `FLASHINFER_MLA_SPARSE`，否则首选 `FLASHMLA_SPARSE`。
>
> **注意：** ROCm 和 CPU 平台拥有它们自己的选择逻辑。详情请参阅平台特定的文档。

## 图例说明 (Legend)

| 列名 | 描述 |
| ------ | ----------- |
| **Dtypes** | 支持的模型数据类型（fp16、bf16、fp32） |
| **KV Dtypes** | 支持的 KV 缓存数据类型（`auto`、`fp8`、`fp8_e4m3`、`fp8_e5m2` 等） |
| **Block Sizes** | 支持的 KV 缓存块大小（%N 表示 N 的倍数） |
| **Head Sizes** | 支持的注意力头大小 |
| **Sink** | 注意力 Sink 支持（用于 StreamingLLM） |
| **Non-Causal** | 针对 Decoder 模型的非因果（双向）注意力支持 |
| **Sparse** | 稀疏注意力支持（仅限 MLA） |
| **MM Prefix** | 多模态前缀全文注意力支持 |
| **DCP** | 解码上下文并行支持（`--decode-context-parallel-size`） |
| **Attention Types** | 支持的注意力模式（Decoder、Encoder、Enc-Dec） |
| **Compute Cap.** | 要求的 CUDA 计算能力（非 CUDA 后端为 N/A） |

**符号：** ✅ = 支持，❌ = 不支持

## 标准注意力后端 (Standard Attention Backends)

| 后端 | 版本 | Dtypes | KV Dtypes | Block Sizes | Head Sizes | Sink | Non-Causal | MM Prefix | DCP | Attention Types | Compute Cap. |
| ------- | ------- | ------ | --------- | ----------- | ---------- | ---- | ---------- | --------- | --- | --------------- | ------------ |
| `CPU_ATTN` | | fp16, bf16, fp32 | `auto`, `fp8`, `fp8_e4m3`, `fp8_e5m2` | %16 | 32, 64, 80, 96, 112, 128, 160, 192, 224, 256, 512 | ❌ | ✅ | ❌ | ❌ | All | N/A |
| `FLASHINFER` | Native† | fp16, bf16 | `auto`, `float16`, `bfloat16`, `fp8`, `fp8_e4m3`, `fp8_e5m2` | 16, 32, 64, 128, 256, 512, 1024 | 64, 128, 256, 512 | ❌ | ✅ | ❌ | ✅ | Decoder | 8.x-9.x |
| `FLASHINFER` | XQA† | fp16, bf16 | `auto`, `float16`, `bfloat16`, `fp8`, `fp8_e4m3`, `fp8_e5m2` | 16, 32, 64, 128, 256, 512, 1024 | 64, 128, 256, 512 | ❌ | ❌ | ❌ | ✅ | Decoder | 9.0 |
| `FLASHINFER` | trtllm-gen† | fp16, bf16 | `auto`, `float16`, `bfloat16`, `fp8`, `fp8_e4m3`, `fp8_e5m2`, `nvfp4` | 16, 32, 64, 128, 256, 512, 1024 | 64, 128, 256, 512 | ✅ | ✅ | ❌ | ✅ | Decoder | 10.x |
| `FLASH_ATTN` | FA2* | fp16, bf16 | `auto`, `float16`, `bfloat16` | %16 | Any | ❌ | ✅ | ❌ | ✅ | All | ≥8.0 |
| `FLASH_ATTN` | FA3* | fp16, bf16 | `auto`, `float16`, `bfloat16`, `fp8`, `fp8_e4m3`, `fp8_e5m2` | %16 | Any | ✅ | ✅ | ❌ | ✅ | All | 9.x |
| `FLASH_ATTN` | FA4* | fp16, bf16 | `auto`, `float16`, `bfloat16` | %16 | Any | ✅ | ✅ | ❌ | ✅ | All | ≥10.0 |
| `FLASH_ATTN_DIFFKV` | | fp16, bf16 | `auto` | Any | Any | ❌ | ❌ | ❌ | ✅ | Decoder | Any |
| `FLEX_ATTENTION` | | fp16, bf16, fp32 | `auto`, `float16`, `bfloat16` | %16 | Any | ❌ | ✅ | ✅ | ❌ | Decoder, Encoder Only | Any |
| `HPC_ATTN` | | fp16, bf16 | `auto`, `fp8_e4m3` | 64 | 128 | ❌ | ❌ | ❌ | ❌ | Decoder | ≥9.0 |
| `ROCM_AITER_FA` | | fp16, bf16 | `auto`, `float16`, `bfloat16`, `fp8`, `fp8_e4m3`, `fp8_e5m2` | 16, 32 | 64, 128, 256 | ✅ | ✅ | ❌ | ❌ | Decoder | N/A |
| `ROCM_AITER_UNIFIED_ATTN` | | bf16 | `auto`, `bfloat16`, `fp8`, `fp8_e4m3` | %16 | Any | ✅ | ❌ | ✅ | ❌ | All | N/A |
| `ROCM_ATTN` | | fp16, bf16, fp32 | `auto`, `float16`, `bfloat16`, `fp8`, `fp8_e4m3`, `fp8_e5m2` | %16 | 32, 64, 80, 96, 128, 160, 192, 224, 256 | ❌ | ✅ | ✅ | ❌ | Decoder, Encoder, Encoder Only | N/A |
| `TRITON_ATTN` | | fp16, bf16, fp32 | `auto`, `float16`, `bfloat16`, `fp8`, `fp8_e4m3`, `fp8_e5m2`, `int4_per_token_head`, `int8_per_token_head`, `fp8_per_token_head` | %16 | Any | ✅ | ✅ | ✅ | ❌ | All | Any |
| `TRITON_ATTN_DIFFKV` | | fp16, bf16 | `auto`, `bfloat16` | Any | Any | ❌ | ❌ | ❌ | ❌ | Decoder | Any |
| `TURBOQUANT` | | fp16, bf16 | `turboquant_k8v4`, `turboquant_4bit_nc`, `turboquant_k3v4_nc`, `turboquant_3bit_nc` | 16, 32, 64, 128 | Any | ❌ | ❌ | ❌ | ❌ | Decoder | Any |

> **†** FlashInfer Native 是常规的 FlashInfer 路径。XQA 是通过 FlashInfer 的 TRTLLM 解码 API 暴露的 SM90 解码路径。trtllm-gen 用于 SM100 并支持 Sinks。可以通过 `--attention-config.use_trtllm_attention=0` 禁用 XQA/trtllm-gen。
>
> **\*** 可通过 `--attention-config.flash_attn_version=2`、`3` 或 `4` 指定 FlashAttention 的版本。默认在 SM100+ (Blackwell) 上为 FA4，在 SM90 (Hopper) 上为 FA3，其余情况为 FA2。

## MiniMax M3 稀疏注意力后端 (MiniMax M3 Sparse Attention Backends)

用于 MiniMax M3 稀疏（“Lightning Indexer”）层的块稀疏 GQA 后端。它直接由模型进行硬编码连接，不属于上述自动优先级列表的一部分。Lightning Indexer 对 KV 缓存块进行评分，选择 top-k 块（加上固定的初始/本地块），注意力机制仅针对这些块进行计算；索引键保存在一个单独的侧边缓存中。

| 后端 | Dtypes | KV Dtypes | Block Sizes | Head Sizes | Sink | Non-Causal | MM Prefix | DCP | Attention Types | Compute Cap. |
| ------- | ------ | --------- | ----------- | ---------- | ---- | ---------- | --------- | --- | --------------- | ------------ |
| `MINIMAX_M3_SPARSE` | bf16, fp16 | `bfloat16`, `fp8`, `fp8_e4m3`, `fp8_e5m2` | 128 | 128 | ❌ | ❌ | ❌ | ❌ | Decoder | Any |

## MLA (Multi-head Latent Attention) 后端

MLA 在 Prefill 预填充和 Decode 解码阶段使用不同的后端。

### 预填充后端 (Prefill Backends)

要显式选择一个预填充后端，请使用 `-ac.mla_prefill_backend=<BACKEND>`（例如 `FLASH_ATTN`、`FLASHINFER`）。否则，预填充后端将在运行时根据硬件和配置自动选择。

| 后端 | 描述 | Dtypes | Compute Cap. | 备注 |
| ------- | ----------- | ------ | ------------ | ----- |
| `FLASH_ATTN`‡ | FlashAttention varlen (FA2/FA3/FA4) | fp16, bf16 | Any | 在 SM100+ 上为 FA4，在 SM90 上为 FA3，其余情况为 FA2 |
| `TRTLLM_RAGGED` | TensorRT-LLM 参差不齐注意力 | fp16, bf16 | 10.x | 仅限 (qk_nope_head_dim=128, qk_rope_head_dim=64, v_head_dim=128) 或 (qk_nope_head_dim=192, qk_rope_head_dim=64, v_head_dim=256) |
| `FLASHINFER` | FlashInfer CUTLASS 后端 | fp16, bf16 | 10.x | 仅限 (qk_nope_head_dim=128, qk_rope_head_dim=64, v_head_dim=128) |
| `TOKENSPEED_MLA` | | fp16, bf16 | 10.x | 仅限 (qk_nope_head_dim=128, qk_rope_head_dim=64, v_head_dim=128) |

> **‡** 自动选择会最先尝试 FlashAttention。在 Blackwell (SM100) 上，回退降级顺序依次为 TRT-LLM Ragged、FlashInfer，然后是 TokenSpeed MLA。在其他 GPU 上，仅考虑 FlashAttention。

### 解码后端 (Decode Backends)

MLA 解码后端使用标准的 `-ac.backend=<BACKEND>` 参数进行选择（例如 `FLASHMLA`、`TRITON_MLA`）。

| 后端 | Dtypes | KV Dtypes | Block Sizes | Head Sizes | Sink | Non-Causal | Sparse | MM Prefix | DCP | Attention Types | Compute Cap. |
| ------- | ------ | --------- | ----------- | ---------- | ---- | ---------- | ------ | --------- | --- | --------------- | ------------ |
| `CUTLASS_MLA` | fp16, bf16 | `auto`, `float16`, `bfloat16`, `fp8`, `fp8_e4m3` | 128 | Any | ❌ | ❌ | ❌ | ❌ | ✅ | Decoder | 10.x |
| `FLASHINFER_MLA` | fp16, bf16 | `auto`, `float16`, `bfloat16`, `fp8`, `fp8_e4m3` | 32, 64 | Any | ❌ | ❌ | ❌ | ❌ | ✅ | Decoder | 10.x |
| `FLASHINFER_MLA_SPARSE` | fp16, bf16 | `auto`, `float16`, `bfloat16`, `fp8`, `fp8_e4m3` | 32, 64 | Any | ❌ | ❌ | ❌ | ❌ | ✅ | Decoder | 10.x |
| `FLASHINFER_MLA_SPARSE_SM120` | bf16 | `auto`, `fp8`, `fp8_e4m3`, `fp8_ds_mla` | 64, 256 | Any | ❌ | ❌ | ❌ | ❌ | ❌ | Decoder | 12.x |
| `FLASHMLA` | fp16, bf16 | `auto`, `float16`, `bfloat16`, `fp8`, `fp8_e4m3` | 64 | Any | ❌ | ❌ | ❌ | ❌ | ✅ | Decoder | 9.x-10.x |
| `FLASHMLA_SPARSE` | bf16 | `auto`, `bfloat16`, `fp8_ds_mla` | 64 | 576 | ❌ | ❌ | ✅ | ❌ | ❌ | Decoder | 9.x-10.x |
| `FLASH_ATTN_MLA` | fp16, bf16 | `auto`, `float16`, `bfloat16` | %16 | Any | ❌ | ❌ | ❌ | ❌ | ✅ | Decoder | 9.x |
| `FLASH_ATTN_MLA_SPARSE` | fp16, bf16 | `auto`, `float16`, `bfloat16` | 64 | Any | ❌ | ❌ | ✅ | ❌ | ❌ | Decoder | 9.x |
| `ROCM_AITER_MLA` | fp16, bf16 | `auto`, `float16`, `bfloat16`, `fp8`, `fp8_e4m3`, `fp8_e5m2` | %1 | Any | ❌ | ❌ | ❌ | ❌ | ❌ | Decoder | N/A |
| `ROCM_AITER_MLA_SPARSE` | fp16, bf16 | `auto`, `float16`, `bfloat16`, `fp8`, `fp8_e4m3` | 1, 64 | Any | ❌ | ❌ | ✅ | ❌ | ❌ | Decoder | N/A |
| `ROCM_AITER_TRITON_MLA` | fp16, bf16 | `auto` | Any | Any | ❌ | ❌ | ❌ | ❌ | ❌ | Decoder | N/A |
| `TOKENSPEED_MLA` | fp16, bf16 | `fp8`, `fp8_e4m3` | 32, 64 | Any | ❌ | ❌ | ❌ | ❌ | ❌ | Decoder | 10.x |
| `TRITON_MLA` | fp16, bf16 | `auto`, `float16`, `bfloat16`, `fp8`, `fp8_e4m3` | %16 | Any | ❌ | ❌ | ❌ | ❌ | ✅ | Decoder | Any |
| `XPU_MLA_SPARSE` | fp16, bf16 | `auto`, `float16`, `bfloat16` | Any | 576 | ❌ | ❌ | ✅ | ❌ | ❌ | Decoder | Any |

### DeepSeek V4 解码后端 (DeepSeek V4 Decode Backends)

DeepSeek V4 稀疏 MLA 使用其专属的解码后端，通过 `--attention-backend=<BACKEND>` 进行选择（例如 `FLASHMLA_SPARSE_DSV4`、`FLASHINFER_MLA_SPARSE_DSV4`）。它们共享 V4 稀疏索引流水线（Compressor + SWA + Indexer，256 Token 的块，头大小 512）；在 NVIDIA GPU 上，SM12x 默认是 `FLASHINFER_MLA_SPARSE_DSV4`，其他受支持的 CUDA 架构上默认是 `FLASHMLA_SPARSE_DSV4`。

| 后端 | Dtypes | KV Dtypes | Block Sizes | Head Sizes | Sink | Non-Causal | Sparse | MM Prefix | DCP | Attention Types | Compute Cap. |
| ------- | ------ | --------- | ----------- | ---------- | ---- | ---------- | ------ | --------- | --- | --------------- | ------------ |
| `FLASHINFER_MLA_SPARSE_DSV4` | bf16 | `auto`, `bfloat16`, `fp8`, `fp8_e4m3`, `fp8_ds_mla` | 256 | 512 | ✅ | ❌ | ✅ | ❌ | ❌ | Decoder | 10.x, 12.x |
| `FLASHMLA_SPARSE_DSV4` | bf16 | `auto`, `fp8_ds_mla`, `fp8` | 256 | 512 | ✅ | ❌ | ✅ | ❌ | ❌ | Decoder | 9.x-10.x |
| `ROCM_FLASHMLA_SPARSE_DSV4` | fp16, bf16 | `auto` | Any | Any | ❌ | ❌ | ❌ | ❌ | ❌ | Decoder | N/A |
