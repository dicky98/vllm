# 优化级别 (Optimization Levels)

## 概述 (Overview)

vLLM 提供了 4 个优化级别（`-O0`、`-O1`、`-O2`、`-O3`），允许用户在启动时间与性能之间进行权衡：

- `-O0`：无优化。启动速度最快，但性能最低。
- `-O1`：快速优化。进行简单编译和快速融合，并启用 `PIECEWISE`（分段式）CUDA 图。
- `-O2`：默认优化。增加编译范围、更多的融合操作，以及启用 `FULL_AND_PIECEWISE`（完整与分段结合式）CUDA 图。
- `-O3`：激进优化。目前与 `-O2` 相同，但未来可能会包含耗时较长或实验性的优化。

所有优化级别的默认行为都可以通过手动设置底层的 Flags 来达到。
用户手动设置的 Flags 优先级高于优化级别的默认设置。

## 各级别总结与使用示例

```bash
# CLI 命令用法
vllm serve RedHatAI/Llama-3.2-1B-FP8 -O1

# Python API 用法
from vllm.entrypoints.llm import LLM

llm = LLM(
    model="RedHatAI/Llama-3.2-1B-FP8",
    optimization_level=2 # 等价于 -O2
)
```

### `-O0`：无优化

以最快速度启动 —— 不进行自动调优，不进行编译，也不使用 CUDA 图。
此级别非常适合开发和调试的初始阶段。

设置参数：

- `-cc.cudagraph_mode=NONE`
- `-cc.mode=NONE`（同时也导致 `-cc.custom_ops=["none"]`）
- `-cc.pass_config.fuse_...=False`（禁用所有算子融合）
- `--kernel-config.enable_flashinfer_autotune=False`

### `-O1`：快速优化

优先考虑快速启动，但依然启用编译和 CUDA 图等基本优化。
对于大多数开发场景来说，这是一个很好的折中方案：您既能获得较快的启动速度，又能确保您的代码不会破坏 CUDA 图或导致编译报错。

设置参数：

- `-cc.cudagraph_mode=PIECEWISE`
- `-cc.mode=VLLM_COMPILE`
- `--kernel-config.enable_flashinfer_autotune=True`

算子融合 (Fusions)：

- `-cc.pass_config.fuse_norm_quant=True`*
- `-cc.pass_config.fuse_act_quant=True`*
- `-cc.pass_config.fuse_act_padding=True`†
- `-cc.pass_config.fuse_mla_dual_rms_norm=True`†

\* 这些融合仅在相应算子使用自定义 Kernel 时启用，否则 Inductor 融合的效果会更好。</br>
† 这些融合仅适用于 ROCm，且需要 AITER。

### `-O2`：全面优化（默认值）

优先考虑性能，代价是增加启动时间。
对于生产环境的工作负载，建议使用该级别，因此它是默认级别。
在此级别下，由于编译范围变大，算子融合的耗时*可能*会变长。

设置参数（在 `-O1` 之上）：

- `-cc.cudagraph_mode=FULL_AND_PIECEWISE`
- `-cc.pass_config.fuse_allreduce_rms=True`
- `-cc.pass_config.fuse_rope_kvcache=True`†

† 这些融合仅适用于 ROCm，且需要 AITER。

### `-O3`：激进优化

该级别目前与 `-O2` 相同，但未来可能会包含更为耗时或实验性的优化。

## 问题排查 (Troubleshooting)

### 常见问题

1. **启动时间过长**：使用 `-O0` 或 `-O1` 获得更快的启动速度。
2. **编译报错**：使用 `debug_dump_path` 获取更多调试信息。
3. **性能问题**：确保在生产环境中使用了 `-O2`。
