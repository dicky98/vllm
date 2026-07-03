# 双批次重叠 (Dual Batch Overlap, DBO)

## 动因 (Motivation)

vLLM 中 DBO（Dual Batch Overlap，双批次重叠）系统的核心设计初衷是，将 MoE（混合专家）层中的稀疏 All-to-All 通信（Sparse all-to-all communication）与周边的计算重叠起来。该系统目前仅针对 DP+EP（数据并行 + 专家并行）部署。

## 简介 (Introduction)

双批次重叠系统的工作原理是在模型运行器（Model Runner）中对批次（Batch）进行拆分，创建两个工作线程，然后在这两个工作线程上分别运行模型。当启用 DBO 时，`FusedMoEModularKernel` 内的让步点（Yield points）允许这两个 CPU 工作线程（也称为 UBatch 线程）以“乒乓（ping-pong）”交替方式运行，从而当其中一个在进行计算时，另一个在等待通信。在整个代码库中，`ubatch` 可用作 `microbatch`（微批次）的缩写；这是 ASCII 友好的 μ-batch 缩写形式。

DBO 系统包括对 `GpuModelRunner` 和 `ModularKernel` 的修改，并定义了两个实用类：`UBatchWrapper` 和 `UBatchContext`。`UBatchWrapper` 管理线程生命周期和模型的 CUDA 图执行。`UBatchContext` 包装了 `ForwardContext`，以协调两个 UBatch 线程之间的同步。

下面是 vLLM 目前实现的数据重叠调度表（Overlap Schedule）：

```python
# 调度符号图例：
#    S = 共享专家 (Shared expert)
#    A0 = MLA qkv proj (投影)
#    A1 = 核心注意力计算 + out proj (输出投影) + MoE gate (门控)
#    D = 分发通信 (Dispatch)
#    C = 结合通信 (Combine)

# 计算 (Comp): |-A0₀-A1₀-||-MLP₁-||-S₁-MLP₀-||-S₀-A0₁-A1₁-|
# 通信 (Comm): |----D₁---||--D₀--||----C₁---||-----C₀-----|
# 顺序: D₁ 发送, A0₀, A1₀, D₁ 接收, D₀ 发送, MLP₁, D₀ 接收,
#       C₁ 发送, S₁, MLP₀, C₁ 接收, C₀ 发送, S₀, A0₁, A1₁, C₀ 接收.
# MLP_SHARED_OVERLAP = "mlp_shared_overlap"
```

## 运行 DBO (Running with DBO)

要启用 DBO 系统，请在 `vllm serve` 命令中传入 `--enable-dbo` 参数。这必须与 `--data-parallel-size N`（其中 N 大于 1）和 `--enable-expert-parallel` 结合使用。此外，还有两个配置旋钮：

* `--dbo-decode-token-threshold`：在仅 Decode（解码）的批次中启用 DBO 所需的最小 Token 数量。
* `--dbo-prefill-token-threshold`：在包含至少一个 Prefill（预填充）的批次中启用 DBO 所需的最小 Token 数量。

目前，DBO 仅在搭配 DeepEP 时受支持。如果您的工作负载主要为 Decode 请求，必须安装 DeepEP 并将 `--all2all-backend` 参数设置为 `deepep_low_latency`；如果主要为 Prefill 请求，则应设置为 `deepep_high_throughput`。

以下是启动具有专家并行和 DBO 启用的双 DP Rank 服务器的命令示例：
例如：`vllm serve deepseek-ai/DeepSeek-V2-Lite --trust-remote-code --data-parallel-size 2 --enable-expert-parallel --enable-dbo --all2all-backend deepep_low_latency`

*注意：`CUDA_VISIBLE_DEVICES` 中必须至少有两个可见的 GPU。*

## DBO 组件 (DBO Components)

* `GPUModelRunner`
* `UBatchWrapper`
* `UBatchContext`

### GPU 模型运行器 (GPU Model Runner)

批次由 `GPUModelRunner` 类分割为微批次。这通过两个步骤完成。首先，在所有 DP Rank 之间进行协调，以确定是否应用微批次化（Microbatching）。微批次化在所有 DP Rank 上必须保持一致。如果微批次化在任何一个 DP Rank 上不可行，则在所有 Rank 上都将其禁用。如果所有 DP Rank 都准备使用微批次化，则总 Token 数量会被填充（Pad）到所有 Rank 中的最大 Token 数。如果在应用填充后，任何 Rank 的第二个微批次变为空，则微批次化将中止，所有 Rank 都不会使用微批次。一旦所有 Rank 都启动了微批次化，就会执行第二步：`CommonAttentionMetadata` 会被 `GPUModelRunner` 对半切分，使每个微批次拥有一个 Attention 元数据。

### UBatchWrapper

`UBatchWrapper` 类是一个模型包装器，负责 DBO 的所有线程、`UBatchContext` 和 CUDA 图的管理。它的设计对于 GPU 模型运行器来说相对透明。

其具体实现会将模型运行两次，每个微批次运行一次。每次模型调用都发生在一个 UBatch 线程内。这些线程被并行启动并通过 `UBatchContext` 进行同步。每个线程会得到一个切分后的 Attention 元数据版本，用于运行其对应批次的那一半。

DBO 的 CUDA 图完全由 `UBatchWrapper` 管理。因此，DBO 仅支持在完整 CUDA 图（Full CUDA graphs）模式下运行。然而，一旦 DBO CUDA 图被捕获，就可以重新回放而不需要任何多线程或 CPU 同步。

#### 接口 (Interfaces)

`__init__` 方法接收模型、`VllmConfig`、`CUDAGraphMode` 和设备（device）。

`forward` 方法仅接收模型参数。它根据 `forward_context` 中是否存在 `ubatch_slices` 对象来决定是否使用 DBO 运行。否则，模型将在没有 DBO 的情况下运行。

### UBatchContext

`UBatchContext` 类是一个 `ForwardContext` 包装类，供 `UBatchWrapper` 类用于同步两个 UBatch 线程。它只能通过 `make_ubatch_contexts` 实例化。

当其中一个 UBatch 线程到达 `dbo_yield` 调用时，它会暂停并启动另一个线程，另一个线程将一直运行直到到达相同的 `dbo_yield` 调用。这种“乒乓”动态过程在每次 `dbo_yield` 调用时交替切换线程，直至模型执行完成。

当前的实现中，所有的 `dbo_yield` 和 `dbo_maybe_run_recv_hook` 调用都放在 `FusedMoEModularKernel.forward` 方法中。

#### 接口 (Interfaces)

`make_ubatch_contexts` 函数初始化两个 `UBatchContext`，分别对应两个 UBatch 线程。它接收两个 CUDA 流、已存在的 `ForwardContext` 以及一个 CPU 线程屏障（Barrier）。该函数是实例化 `UBatchContext` 的唯一途径。它将处理所有的事件初始化。

`dbo_register_recv_hook` 方法注册一个回调函数，可在另一个 UBatch 线程的 `UBatchContext` 中由 `FusedMoEPrepareAndFinalizeModular` 类返回。该回调将在另一个线程调用 `dbo_maybe_run_recv_hook` 时运行。这通常用于等待 All-to-All 算子核（Kernel）的完成。

`dbo_maybe_run_recv_hook` 方法运行由 `dbo_register_recv_hook` 函数设置的回调（如果该回调存在）。

`dbo_yield` 方法使当前线程进入睡眠，并唤醒另一个 UBatch 线程。
