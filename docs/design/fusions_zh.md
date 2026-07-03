# 算子融合的 torch.compile Passes (Fusion torch.compile passes)

vLLM 在编译时（通过自定义的 [`torch.compile`](torch_compile_zh.md) Inductor Passes）应用了一系列算子核/操作融合（Kernel/Operator fusions），以将优化与模型定义分离开来，并避免破坏模型代码中的层抽象。这些融合由 [`PassConfig`][vllm.config.compilation.PassConfig] 中的字段控制，并在适当的 [优化级别 (Optimization levels)](optimization_levels_zh.md) 下自动启用。

## 快速参考 (Quick Reference)

下表将每种融合映射到其对应的控制 Flag/配置旋钮、融合的操作、默认启用的优化级别以及参考加速比。

- **Fullgraph（全图）** 列指出该融合是否需要整个模型图可见（通过 Inductor Partition 或设置 `splitting_ops=[]`）。
- 最后一列指出该融合是针对所有的 `num_tokens` 还是仅在较低或较高的 Token 范围下激活。

!!! info "信息"
    加速比很大程度上取决于具体的模型、Batch Size 和硬件设备。如果手动调优性能，请务必在启用和禁用该融合的情况下分别测试您的具体用例，以验证实际影响。

| 算子融合 (Fusion) | `PassConfig` 标志 | 融合的操作 | 默认启用级别 | 端到端加速比 | 全图 (Fullgraph) | `num_tokens` 范围 |
| ------------------------------------------------------------------------------ | ---------------------------- | ---------------------------------------------- | ------------------------------ | ------------------ | --------- | ------------ |
| [AllReduce + RMSNorm](#allreduce--rmsnorm-fuse_allreduce_rms) | `fuse_allreduce_rms` | All-reduce → RMSNorm (+residual_add) (→ quant) | O2 (Hopper/Blackwell 且 TP > 1) | 5-20% | ❌ | 较低 |
| [Attention + Quant](#attention--quantization-fuse_attn_quant) | `fuse_attn_quant` | Attention output → FP8/NVFP4 quant | 默认关闭 | 3-7% | ✅ | 始终启用 |
| [MLA Attention + Quant](#attention--quantization-fuse_attn_quant) | `fuse_attn_quant` | MLA Attention output → FP8/NVFP4 quant | 默认关闭 | 待定 (TBD) | ✅ | 始终启用 |
| [RoPE + KV-Cache Update](#rope--kv-cache-update-fuse_rope_kvcache) | `fuse_rope_kvcache` | Rotary embedding → KV cache write | 仅限 O2 (ROCm/AITER) | 2-4% | ❌ | 较低 |
| [QK Norm + RoPE](#qk-norm--rope-enable_qk_norm_rope_fusion) | `enable_qk_norm_rope_fusion` | Q/K RMSNorm → rotary embedding | 默认关闭 | 2-3% | ❌ | 较低 |
| [Sequence Parallelism](#sequence-parallelism-enable_sp) | `enable_sp` | AllReduce → ReduceScatter + AllGather | 默认关闭 | AsyncTP 的先决条件 | ✅ | 较高 |
| [AsyncTP GEMM + collective](#asynctp-gemm--collective-overlap-fuse_gemm_comms) | `fuse_gemm_comms` | GEMM → reduce-scatter / all-gather → GEMM | 默认关闭 | 7-10% | ✅ | 较高 |
| [RMSNorm + Quant](#rmsnorm--quantization-fuse_norm_quant) | `fuse_norm_quant` | RMSNorm (+residual add) → FP8/FP4 quant | O1 (有条件启用) | 1-4% | ❌ | 始终启用 |
| [SiLU+Mul + Quant](#silumul--quantization-fuse_act_quant) | `fuse_act_quant` | SiLU+Mul activation → FP8/FP4 quant | O1 (有条件启用) | 1-4% | ❌ | 始终启用 |
| [RMSNorm + Padding](#rmsnorm--padding-fuse_act_padding) | `fuse_act_padding` | Residual add + RMSNorm → padding | 仅限 O1 (ROCm/AITER) | 待定 (TBD) | ❌ | 始终启用 |
| [MLA Dual RMSNorm](#mla-dual-rmsnorm-fuse_mla_dual_rms_norm) | `fuse_mla_dual_rms_norm` | 配对的 Q + KV RMSNorm (+ FP8 quant) → 1 个 Kernel | 仅限 O1 (ROCm/AITER) | 1-2% | ❌ | 始终启用 |

## 支持情况矩阵 (Support Matrix)

下表列出了每种融合在各个平台上支持的量化方案。

- **—** 表示该融合在相应平台上不可用。最新和开发中的工作可在追踪 Issue 中查看：[#36066](https://github.com/vllm-project/vllm/issues/36066)。

| 算子融合 (Fusion) | SM100 (Blackwell) | SM90 (Hopper) | SM89 (Ada) | SM80 (Ampere) | ROCm |
| ---------------------------- | ---------------------------------------- | ---------------------------------------- | ---------------------------------------- | ------------- | ---------------------------------------- |
| `fuse_allreduce_rms` | FP16/BF16, FP8 static, NVFP4 | FP16/BF16, FP8 static | — | — | — |
| `fuse_attn_quant`\* | FP8 static\*, NVFP4\* | FP8 static\*, | FP8 static\* | — | FP8 static\* |
| `fuse_attn_quant` (MLA)\* | FP8 static\*, FP8 per-group\*, NVFP4\* | FP8 static\*, FP8 per-group\* | FP8 static\*, FP8 per-group\* | — | FP8 static\* (未测试) |
| `fuse_rope_kvcache` | — | — | — | — | FP16/BF16 |
| `enable_qk_norm_rope_fusion` | FP16/BF16 | FP16/BF16 | FP16/BF16† | FP16/BF16† | — |
| `enable_sp` | FP16/BF16, FP8 static† | FP16/BF16, FP8 static | FP16/BF16† | FP16/BF16† | — |
| `fuse_gemm_comms` | FP16/BF16, FP8 static† | FP16/BF16, FP8 static | FP16/BF16† | FP16/BF16† | — |
| `fuse_norm_quant` | FP8 static, FP8 per-token, FP8 per-group | FP8 static, FP8 per-token, FP8 per-group | FP8 static, FP8 per-token, FP8 per-group | — | FP8 static, FP8 per-token, FP8 per-group |
| `fuse_act_quant` | FP8 static, NVFP4 | FP8 static, FP8 per-group (128/64) | FP8 static, FP8 per-group (128/64) | — | FP8 per-group |
| `fuse_act_padding` | — | — | — | — | FP16/BF16 |
| `fuse_mla_dual_rms_norm` | — | — | — | — | BF16 |

\* `fuse_attn_quant` 的支持取决于所使用的注意力后端；并非所有后端都支持融合量化输出。有关每个后端的详细信息，请参阅 [`fuse_attn_quant` 章节](#attention--quantization-fuse_attn_quant)。

† `enable_sp` 和 `fuse_gemm_comms` 目前仅对 SM90 进行自动配置；其他架构的支持需要显式设置 `PassConfig.sp_min_token_num`。SM100 的支持还需要设置环境变量 `VLLM_DISABLED_KERNELS=FlashInferFP8ScaledMMLinearKernel`。

## 启用 / 禁用算子融合

融合配置通过嵌套在 `CompilationConfig` 内部的 `PassConfig` 暴露：

```python
from vllm import LLM
from vllm.config import CompilationConfig, PassConfig

llm = LLM(
    model="...",
    optimization_level=2, # 默认优化级别
    compilation_config=CompilationConfig(
        pass_config=PassConfig(
            fuse_norm_quant=True,
            fuse_act_quant=True,
            fuse_allreduce_rms=False,  # 禁用特定融合
        )
    ),
)
```

也可以通过任何 `vllm ...` 命令的命令行标志来启用融合：

```bash
# 启用 O2 默认配置，但关闭 allreduce 融合
vllm serve meta-llama/Llama-3.1-8B-Instruct -O2 -cc.pass_config.fuse_allreduce_rms=False

# 上述命令等价于更冗长的：
vllm serve meta-llama/Llama-3.1-8B-Instruct -O2 --compilation-config '{"pass_config": {"fuse_allreduce_rms": false}}'

# 在其他命令中具有相同语法，例如 vllm bench：
vllm bench latency --model=meta-llama/Llama-3.1-8B-Instruct -O2 -cc.pass_config.fuse_allreduce_rms=False
```

由用户显式设置的字段优先级始终高于优化级别的默认设置。

## 算子融合详情 (Fusion Details)

### AllReduce + RMSNorm (`fuse_allreduce_rms`)

!!! warning "警告"
    TP+DP 和 TP+PP 的组合目前存在异常（参见 [#34458](https://github.com/vllm-project/vllm/issues/34458) 和 [#35426](https://github.com/vllm-project/vllm/issues/35426)）。目前仅在安装了 FlashInfer 的 NVIDIA Hopper (SM90) 和 Blackwell (SM100) 上受支持。

**融合内容**：将张量并行 All-reduce 通信操作与随后的残差相加（residual add）、RMSNorm，以及可选的量化步骤融合到单个 FlashInfer / TensorRT-LLM 通信算子核中。此融合仅在 `num_tokens` 较小时有收益，因此仅在较低的编译范围内执行。

涵盖模式：

- `AllReduce → RMSNorm(+residual_add)`：在安装了 FlashInfer 的 CUDA sm90+ 上
- `AllReduce → RMSNorm(+residual_add) → FP8 static quant`：在安装了 FlashInfer 的 CUDA sm90+ 上
- `AllReduce → RMSNorm(+residual_add) → NVFP4 dynamic quant`：在安装了 FlashInfer 的 CUDA sm100+ 上

使用该融合算子核的最大张量大小取决于硬件设备（例如在 SM90/SM100 上，TP=2 时为 64 MB），并可通过 `PassConfig.fi_allreduce_fusion_max_size_mb` 进行配置。

**代码位置**：

- Pass 文件：[`vllm/compilation/passes/fusion/allreduce_rms_fusion.py`](https://github.com/vllm-project/vllm/blob/main/vllm/compilation/passes/fusion/allreduce_rms_fusion.py)
- FlashInfer all-reduce 实现：[`vllm/distributed/device_communicators/flashinfer_all_reduce.py`](https://github.com/vllm-project/vllm/blob/main/vllm/distributed/device_communicators/flashinfer_all_reduce.py)
- 基准测试：[`benchmarks/kernels/benchmark_fused_collective.py`](https://github.com/vllm-project/vllm/blob/main/benchmarks/kernels/benchmark_fused_collective.py)

### Attention + Quantization (`fuse_attn_quant`)

!!! info "信息"
    `fuse_attn_quant` 目前在任何默认优化级别下均未启用，必须显式设置。它要求整个模型图可见（Inductor Partition 或 `splitting_ops=[]`）。

**融合内容**：在注意力计算之后直接融合注意力输出量化，从而消除了注意力输出的全精度内存往返读写（Memory round-trip）。此融合同时支持标准 `Attention` 和 `MLAAttention`（被 DeepSeek-V2/V3/R1 模型所使用）。

涵盖模式：

`Attention → FP8 static quant`：
- `TRITON_ATTN`：CUDA, ROCm
- `FLASHINFER`：安装了 FlashInfer 的 CUDA sm100+
- `ROCM_ATTN`：ROCm
- `ROCM_AITER_UNIFIED_ATTN`：启用了 AITER 的 ROCm

`Attention → NVFP4 dynamic quant`：
- `FLASHINFER`：安装了 FlashInfer 的 CUDA sm100+

`MLAAttention → FP8 static, FP8 per-group, NVFP4 dynamic quant`：
- MLA 融合在图级别对 `unified_mla_attention_with_output` 操作起作用，并适用于所有 MLA 解码和预填充后端组合。与标准 `Attention` 后端（其算子核直接写入 FP8 输出）不同，目前没有任何 MLA 预填充或解码后端支持直接输出 FP8/FP4。该融合将写入中间缓冲区并在单独的步骤中进行量化，因此尚未能消除内存往返读写。

!!! info "信息"
    目前预计 MLA 注意力融合不会带来显著的加速收益。一旦 MLA 预填充/解码算子核支持直接输出 FP8/FP4，这一情况将会得到改善。

其他注意力后端目前尚不支持融合输出量化。

**代码位置**：

- Pass 文件 (Attention)：[`vllm/compilation/passes/fusion/attn_quant_fusion.py`](https://github.com/vllm-project/vllm/blob/main/vllm/compilation/passes/fusion/attn_quant_fusion.py)
- Pass 文件 (MLAAttention)：[`vllm/compilation/passes/fusion/mla_attn_quant_fusion.py`](https://github.com/vllm-project/vllm/blob/main/vllm/compilation/passes/fusion/mla_attn_quant_fusion.py)
- 注意力后端目录：[`vllm/v1/attention/backends/`](https://github.com/vllm-project/vllm/blob/main/vllm/v1/attention/backends/)

### RoPE + KV-Cache Update (`fuse_rope_kvcache`)

!!! info "信息"
    仅限 ROCm/AITER。在 NVIDIA CUDA 或 CPU 上不可用。由于 AITER 融合算子核的性能问题，该融合默认仅在 `num_tokens ≤ 256` 时启用。此阈值可通过 `PassConfig.rope_kvcache_fusion_max_token_num` 进行配置。

**融合内容**：将旋转位置编码（Rotary Positional Embedding）算子核与 KV 缓存 Scatter/Write 融合为单个算子核，避免了对 Key 和 Value 张量的单独读写。

要求：启用了 AITER 的 AMD ROCm、处于活动状态的 `rotary_embedding` 自定义算子（自动），以及在图中可见的 `kv_cache` 更新操作（通过使用 Inductor 图划分或将其从 `splitting_ops` 中移除）。如果满足这些条件，该融合在优化级别 O1 及以上会自动启用。

**代码位置**：

- Pass 文件：[`vllm/compilation/passes/fusion/rope_kvcache_fusion.py`](https://github.com/vllm-project/vllm/blob/main/vllm/compilation/passes/fusion/rope_kvcache_fusion.py)

### Sequence Parallelism (`enable_sp`)

**融合内容**：使用 ReduceScatter + 本地 RMSNorm + AllGather 替换 AllReduce 通信，从而在 TP Rank 之间拆分序列维度。这重构了图，以便后续的 AsyncTP Pass 能够将 ReduceScatter / AllGather 与周边的 GEMMs 进行融合。

序列并行（Sequence Parallelism）本身并不能直接提高性能；它是 AsyncTP Pass (`fuse_gemm_comms`) 的先决条件。SP 仅在高于基于设备能力和模型 `hidden_size` 自动配置的最小 Token 阈值时应用。目前仅在 `hidden_size >= 8192` 模型的 H100/SM90 上激活。此阈值可通过 `PassConfig.sp_min_token_num` 进行配置。

通用转换过程：

```text
输入 (Input) → AllReduce → RMSNorm → 输出 (Output)
转换为：
输入 (Input) → ReduceScatter → 本地 RMSNorm → AllGather → 输出 (Output)
```

涵盖模式：

- 首个 Block：`AllReduce → RMSNorm` 转换为 `ReduceScatter → RMSNorm → AllGather`
- 中间 Blocks：`AllReduce → fused_add_RMSNorm` 转换为 `ReduceScatter → fused_add_RMSNorm → AllGather`
- 两者均带有可选的 `→ FP8 static quant` 后缀

要求：`use_inductor_graph_partition=True` **或者** 静态大小能够被 `tensor_parallel_size` 整除的分段式编译。

支持硬件：仅在 NVIDIA CUDA 上测试，ROCm 上可能可行。FP8 AllGather 需要 SM90+。

**代码位置**：

- Pass 文件：[`vllm/compilation/passes/fusion/sequence_parallelism.py`](https://github.com/vllm-project/vllm/blob/main/vllm/compilation/passes/fusion/sequence_parallelism.py)

### AsyncTP GEMM + Collective Overlap (`fuse_gemm_comms`)

!!! info "信息"
    需要 `enable_sp=True`（自动启用）。如果未应用序列并行，则此 Pass 不执行任何操作。

**融合内容**：在序列并行对图进行转换后，使用 `torch.ops.symm_mem` 对称内存原语将 GEMM 算子核与周边的 ReduceScatter（输出投影）和 AllGather（输入投影）进行融合，从而使通信和计算重叠。由于这种重叠仅在 `num_tokens` 较大时有收益，因此融合（和之前的 SP）仅在高于 `PassConfig.sp_min_token_num` 的高段编译范围内执行。

涵盖模式：

- `GEMM → reduce-scatter` 转换为 `fused_matmul_reduce_scatter`
- `all-gather → GEMM` 转换为 `all_gather_matmul`
- 这两种模式的 FP8 缩放变体

支持硬件：支持对称内存（`torch.distributed._symmetric_memory`）的 NVIDIA CUDA 平台。

在 B200 上，由于不支持模式匹配的 FP8 FlashInfer 缩放 MM，必须将其禁用（参见 [#27893](https://github.com/vllm-project/vllm/issues/27893)）：

```shell
VLLM_DISABLED_KERNELS=FlashInferFP8ScaledMMLinearKernel ...
```

**代码位置**：

- Pass 文件：[`vllm/compilation/passes/fusion/collective_fusion.py`](https://github.com/vllm-project/vllm/blob/main/vllm/compilation/passes/fusion/collective_fusion.py)
- 序列并行 Pass：[`vllm/compilation/passes/fusion/sequence_parallelism.py`](https://github.com/vllm-project/vllm/blob/main/vllm/compilation/passes/fusion/sequence_parallelism.py)

### QK Norm + RoPE (`enable_qk_norm_rope_fusion`)

!!! info "信息"
    仅适用于在旋转位置编码（RoPE）之前对 Q 和 K 应用逐头（per-head）RMSNorm 的模型（例如 Qwen）。由于 H100 上的性能问题，默认在任何优化级别下均未启用：[#34391](https://github.com/vllm-project/vllm/issues/34391)。

**融合内容**：将拆分 QKV → 重塑（Reshape） → Q/K RMSNorm → 重塑（Reshape） → 旋转位置编码（RoPE）这一序列操作融合为单个 `fused_qk_norm_rope` CUDA 算子核。

```text
# 未融合前：
q, k, v = split(qkv)
q_norm = rms_norm(q.view(heads))
k_norm = rms_norm(k.view(kv_heads))
q_rope, k_rope = rotary_embedding(q_norm, k_norm, ...)

# 融合后：
fused_qk_norm_rope(qkv, ...)
```

支持硬件：仅限 CUDA (SM80+)，仅在 SM90 和 SM100 上测试过。

**代码位置**：

- Pass 文件：[`vllm/compilation/passes/fusion/qk_norm_rope_fusion.py`](https://github.com/vllm-project/vllm/blob/main/vllm/compilation/passes/fusion/qk_norm_rope_fusion.py)
- CUDA 算子核：[`csrc/ops.h`](https://github.com/vllm-project/vllm/blob/main/csrc/ops.h) 中的 `fused_qk_norm_rope`

### RMSNorm + Quantization (`fuse_norm_quant`)

!!! warning "警告"
    在 NVIDIA 显卡上，Inductor 生成的融合算子核实际上比我们的自定义 CUDA 算子核更快。因此，此融合仅在 `rms_norm` 或 `quant_fp8` 使用自定义算子核时才启用。

**融合内容**：将自定义的 `rms_norm` / `fused_add_rms_norm` 操作与随后的量化结合为单个融合算子核，从而消除了全精度激活张量的中间读写。融合了两种变体：

- *常规 RMSNorm + 量化*：`rms_norm(x) → quant_fp8(y)`
- *残差相加 RMSNorm + 量化*：`fused_add_rms_norm(x, residual) → quant_fp8(y)` — 还会就地更新残差。

请注意，AITER 融合目前在 `vllm.compilation.passes.fusion.rocm_aiter_fusion` 的独立 Pass 中进行。

支持的量化方案/硬件组合：

- FP8 static per-tensor: CUDA & HIP kernel
- FP8 dynamic per-token: CUDA & HIP kernel, AITER
- FP8 dynamic per-token-group (128/64): CUDA & HIP kernel, AITER

**代码位置**：

- Pass 文件：[`vllm/compilation/passes/fusion/rms_quant_fusion.py`](https://github.com/vllm-project/vllm/blob/main/vllm/compilation/passes/fusion/rms_quant_fusion.py)
- ROCm AITER Pass：[`vllm/compilation/passes/fusion/rocm_aiter_fusion.py`](https://github.com/vllm-project/vllm/blob/main/vllm/compilation/passes/fusion/rocm_aiter_fusion.py)
- CUDA/HIP 算子核：[`csrc/layernorm_quant_kernels.cu`](https://github.com/vllm-project/vllm/blob/main/csrc/layernorm_quant_kernels.cu)

### SiLU+Mul + Quantization (`fuse_act_quant`)

!!! warning "警告"
    与 `fuse_norm_quant` 类似：在 NVIDIA 上，Inductor 生成的融合算子核比我们的自定义操作更快。此融合仅在 `silu_and_mul` 或 `quant_fp8` 使用自定义算子核时启用，或用于 NVFP4 量化模型（因为 FP4 量化始终是自定义操作）。

**融合内容**：将 `silu_and_mul` 门控升级投影（Gate-up projection）激活操作与随后的量化融合到单个算子核中，避免了全精度激活后张量的实例化。

请注意，AITER 融合在 `vllm.compilation.passes.fusion.rocm_aiter_fusion` 的独立 Pass 中。

支持的量化方案/硬件组合：

- FP8 static per-tensor: CUDA & HIP kernel
- FP8 dynamic per-group (128/64): CUDA kernel (SM89+，在 SM100+ 上使用 DeepGemm 时不激活)
- NVFP4 dynamic: 仅限带 FlashInfer 的 CUDA SM100+
- FP8 per-token-group (128): 仅限 ROCm AITER

**代码位置**：

- Pass 文件：[`vllm/compilation/passes/fusion/act_quant_fusion.py`](https://github.com/vllm-project/vllm/blob/main/vllm/compilation/passes/fusion/act_quant_fusion.py)
- ROCm AITER Pass：[`vllm/compilation/passes/fusion/rocm_aiter_fusion.py`](https://github.com/vllm-project/vllm/blob/main/vllm/compilation/passes/fusion/rocm_aiter_fusion.py)
- CUDA/HIP 算子核目录：[`csrc/quantization/`](https://github.com/vllm-project/vllm/blob/main/csrc/quantization/)
- 融合的 SiLU+Mul+BlockQuant 算子核：[`csrc/quantization/fused_kernels/fused_silu_mul_block_quant.cu`](https://github.com/vllm-project/vllm/blob/main/csrc/quantization/fused_kernels/fused_silu_mul_block_quant.cu)

### RMSNorm + Padding (`fuse_act_padding`)

!!! info "信息"
    仅限 ROCm/AITER。针对 GPT-OSS 模型。

**融合内容**：将残差相加 + RMSNorm 与随后的 Padding 操作融合，该 Padding 操作将隐藏维度填充（Pad）到下游 AITER Triton GEMM 算子核所要求的倍数。

要求：启用了 AITER RMSNorm 的 AMD ROCm。当隐藏大小为 2880 且*没有*启用 AITER Triton GEMMs 时，在优化级别 O1 及以上默认启用。

**代码位置**：

- Pass 文件：[`vllm/compilation/passes/fusion/rocm_aiter_fusion.py`](https://github.com/vllm-project/vllm/blob/main/vllm/compilation/passes/fusion/rocm_aiter_fusion.py) 中的 `RocmAiterTritonAddRMSNormPadFusionPass`

### MLA Dual RMSNorm (`fuse_mla_dual_rms_norm`)

!!! info "信息"
    仅限 ROCm/AITER。针对 DeepSeek-V3 / Kimi-K2 MLA 注意力。

!!! note "注意"
    当使用原生的 `rms_norm` 实现时（目前在 CUDA 和 ROCm 上为默认），Inductor 的内置融合已经能够自动合并这些 Norms。这个显式 Pass 针对的是 AITER 的自定义 `rms_norm` 处于活动状态的情况，因为 Inductor 无法自行对其进行融合。

**融合内容**：将 MLA 注意力中配对的 `q_a_layernorm` 和 `kv_a_layernorm` RMSNorm 操作融合到通过 AITER 调用单个 `fused_qk_rmsnorm` HIP 算子核中，从而将每个 MLA 层的算子核启动开销从 2 次减少到 1 次。

```text
# 未融合前：
q_c, kv_lora = split(projected, [q_dim, kv_dim])
kv_c, k_pe   = split(kv_lora,  [kv_c_dim, k_pe_dim])
q_c  = rms_norm(q_c,  q_weight,  eps)
kv_c = rms_norm(kv_c, kv_weight, eps)

# 融合后：
q_c, kv_lora = split(projected, [q_dim, kv_dim])
kv_c, k_pe   = split(kv_lora,  [kv_c_dim, k_pe_dim])
q_normed, kv_normed = fused_mla_dual_rms_norm(
    q_c, q_weight, kv_c, kv_weight, eps1, eps2)
```

要求：启用了 AITER 的 AMD ROCm。当 AITER 可用时，在优化级别 O1 及以上默认启用。

**FP8 注意力变体（逐 Token 量化）**。在使用逐 Token FP8 的 `q_b_proj` 时，只有 *q* 潜变量（latent）被 FP8 量化，而 *kv* 保持为 BF16。
`RocmAiterRMSNormQuantFusionPass` 首先将 q 侧折叠进 `rocm_aiter_rmsnorm_fused_dynamic_quant`，将 kv 留给常规的 `rms_norm` —— 这打破了上述对称模式。然后，相同的 Pass 会对这种不对称对进行匹配，并将其降级为 `fused_mla_dual_rms_norm_per_token_quant`。

```text
# 未融合前 (q 侧 norm+quant 融合; kv 侧仍为常规 rms_norm):
q_c, kv_lora = split(projected, [q_dim, kv_dim])
kv_c, k_pe   = split(kv_lora,  [kv_c_dim, k_pe_dim])
q_fp8, q_scale = rocm_aiter_rmsnorm_fused_dynamic_quant(q_c, q_weight, eps, fp8)
kv_normed      = rms_norm(kv_c, kv_weight, eps)          # bf16

# 融合后：
q_c, kv_lora = split(projected, [q_dim, kv_dim])
kv_c, k_pe   = split(kv_lora,  [kv_c_dim, k_pe_dim])
q_fp8, q_scale, kv_normed = fused_mla_dual_rms_norm_per_token_quant(
    q_c, q_weight, kv_c, kv_weight, eps1, eps2)
```

**代码位置**：

- Pass 文件：[`vllm/compilation/passes/fusion/rocm_aiter_fusion.py`](https://github.com/vllm-project/vllm/blob/main/vllm/compilation/passes/fusion/rocm_aiter_fusion.py)（`MLADualRMSNormFusionPass`，`MLADualRMSPerTokenQuantPattern`）
- 自定义算子文件：[`vllm/_aiter_ops.py`](https://github.com/vllm-project/vllm/blob/main/vllm/_aiter_ops.py)（`fused_mla_dual_rms_norm`，`fused_mla_dual_rms_norm_per_token_quant`）
- AITER 算子核项目：`fused_qk_rmsnorm`，`fused_qk_rmsnorm_per_token_quant`
