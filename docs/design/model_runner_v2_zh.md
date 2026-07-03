# Model Runner V2 设计文档 (Model Runner V2 Design Document)

## 简介 (Introduction)

自从 vLLM V1 首次实现以来，我们发现了一些根本性的设计失误并积累了大量的技术债务。许多新特性的加入在最初的设计中并未被考虑到。同时，我们对采样技术（例如 Gumbel-max 采样）、工具链（例如 Triton）以及 CUDA 特性（例如 UVA）也有了更深入的认识。基于这些知识，我们从根本原则出发，重新实现了 Model Runner V2 (MRV2)，使其更加清晰、高效且模块化。

事后看来，V1 的许多设计选择并非最优。尽管 MRV2 目前尚未实现全部特性，也未经过严苛测试，且仍有一些待确定的设计决策，但我们相信它相比 V1 是一次实质性的改进。

本文档介绍了 MRV2 的设计。

## 1. 常驻批次 (Persistent Batch)

V1 中一个重大的复杂性源头是其常驻批次（Persistent batch）的实现。

### 背景

V1 引入常驻批次是为了在输入准备期间将 CPU 开销降到最低。当为一个步骤调度请求时，Model Runner 必须构建连续的输入张量（例如块表 block tables 和针对每个请求的温度值），以馈送给模型。在 Python 中，每一步都从头开始构建这些张量通常非常缓慢，特别是像块表这样的大张量。

常驻批次的优化利用了相邻步骤中的请求批次绝大多数是完全相同的事实。每一步中只有少数几个请求（如果有的话）会加入或结束。通过维护常驻的状态张量并应用增量差异（incremental diffs），而不是从头重构输入，可以显著降低 CPU 开销。

### V1 方法存在的问题

虽然效率很高，但 V1 的常驻批次设计由于将常驻状态与输入张量耦合在一起，引入了不必要的复杂性。V1 直接将常驻状态张量用作模型和采样器的输入，这强加了严格的布局和排序要求。当有请求加入或结束时，这通常需要复杂的跨张量重新排序，而不能简单地进行行插入/删除。

V1 还必须维护 `CachedRequestState`（请求状态的冗余备份副本），因为在请求仍然处于活动状态时，常驻张量中的行可能会被覆盖。

其结果是导致了复杂的记账式管理，这在异步调度下变得更加困难。

![V1 中的常驻批次](../assets/design/model_runner_v2/persistent_batch_v1.png)

### MRV2 的解决方案

MRV2 将常驻状态张量与每步的输入张量解耦。给定该步骤的请求顺序（通常由注意力后端决定），MRV2 从常驻状态中收集（Gather）输入张量。

1. 预分配一个具有固定大小 `max_num_reqs` 行的张量（在大多数平台上默认是 1024）。
2. 在每个请求的活动生命周期内（直至结束或被抢占），为其分配一行固定的位置。
3. 将抢占（Preemption）视为完成。在恢复时，将请求数据作为全新的状态重新加入。

这消除了对 `CachedRequestState` 的需要并简化了记账逻辑。大型状态张量主要存储在 GPU 内存中，因此 Gather 操作在 GPU 上并行运行，开销极低。

![MRV2 中的常驻批次](../assets/design/model_runner_v2/persistent_batch_mrv2.png)

## 2. 异步优先 (Async-First)

vLLM 现在严重依赖异步调度。调度器和 Worker 在 GPU 执行第 `N` 步时准备第 `N+1` 步的输入，使 CPU 和 GPU 的工作相互重叠，从而最大化利用率。

V1 最初设计时并未考虑异步调度，其支持需要后期改造的行为和各类 Hack。而 MRV2 则假定核心模型执行循环是一个没有任何 CPU 同步点的 CUDA 流。CPU 的入口点只是将工作排入流的队列中。

![异步执行时间线](../assets/design/model_runner_v2/async_sched.png)

## 3. 移除异步屏障 (Removing Async Barrier)

异步执行的一个关键要求是 CPU 操作必须保持非阻塞。必须避免显式同步（例如 `torch.accelerator.synchronize`）和隐式同步（例如未固定内存的 `.to("cuda")`）。

然而，当 CPU 和 GPU 同时操作同一块内存时，异步执行可能会引入竞争条件（Race conditions）。

不安全的示例：

```python
class ModelRunner:
    def __init__(self, ...):
        # 固定内存缓冲区
        self.states = torch.zeros(
            max_num_reqs, dtype=torch.int32, device="cpu", pin_memory=True
        )

    def execute_step(self, ...):
        self.states[req_idx] = new_req.data
        states = self.states.to("cuda", non_blocking=True)
```

当 GPU 仍在通过异步拷贝从 `self.states` 中读取数据时，CPU 可能会修改它。

V1 通过在临界区周围维护一个异步屏障（Async barrier）来解决此问题。这避免了竞争，但也有其缺点：

1. 很容易漏掉需要保护的缓冲区（容易产生 bug）。
2. 组织不够灵活（所有的 CPU 工作必须留在屏障内）。
3. 由于存在同步，重叠（Overlap）可能变少。

![共享 CPU 缓冲区的竞争条件](../assets/design/model_runner_v2/async_race_condition.png)

### MRV2 的解决方案：消除竞争

MRV2 将常驻的 CPU 状态与拷贝的张量分离开来：

```python
class ModelRunner:
    def __init__(self, ...):
        # 不固定内存
        self.states = torch.zeros(
            max_num_reqs, dtype=torch.int32, device="cpu", pin_memory=False
        )

    def execute_step(self, ...):
        self.states[req_idx] = new_req.data
        tmp_states = self.states.pin_memory()
        states = tmp_states.to("cuda", non_blocking=True)
```

现在，当 GPU 从 `tmp_states` 读取数据时，CPU 写入 `self.states`，从而在没有显式同步的情况下消除了竞争。

![临时固定内存拷贝消除竞争](../assets/design/model_runner_v2/async_no_race_condition.png)

## 4. StagedWriteTensor

对于像块表这样的大张量，MRV2 通过使用 `StagedWriteTensor` 避免了每一步都进行完整的 CPU 到 GPU 拷贝：

1. 将基础张量保留在 GPU 上。
2. 在 CPU 上暂存差异（Stage diffs）。
3. 将差异打包到连续的缓冲区中。
4. 将打包好的差异拷贝到 GPU。
5. 启动一个 GPU 算子核（Kernel）来应用差异。

使用示例：

```python
# 在 GPU 上初始化状态
state = StagedWriteTensor(size=(1024, 1000), dtype=torch.int32, device="cuda")

# 在第 2 行、起始索引为 3 的位置写入 [3, 1, 2]
state.stage_write(row=2, start=3, value=[3, 1, 2])

# 在第 0 行、起始索引为 1 的位置写入 [-1, -2, -5]
state.stage_write(row=0, start=1, value=[-1, -2, -5])

# 应用暂存的写入内容
state.apply_write()
```

这支持不规则（Ragged）更新，且无需 CPU-GPU 同步，只需极少的 Kernel 启动。这对于块表以及 CPU/GPU 混合写入的状态（如 `num_computed_tokens`）特别有用。

## 5. GPU 原生的输入元数据准备和输出处理

MRV2 使用 Triton 算子核来准备诸如 `input_ids`、`positions`、`query_start_loc` 和 `seq_lens` 等输入。

收益：

1. 更好的异步行为：GPU 可以推导值（例如在投机解码中），而 CPU 此时可能还不知道这些值。
2. 更低的 CPU 开销：在 GPU 上准备输入开销极低，并且避免了 Python 的性能瓶颈。

### 通用虚拟寻址 (Universal Virtual Addressing, UVA)

MRV2 在某些路径中使用 UVA，以允许 GPU 算子核直接访问驻留在 CPU 上的大型张量（例如 `prefill_token_ids`），而无需将这些张量复制到 GPU 内存中。

## 6. Triton 原生采样器 (Triton-Native Sampler)

MRV2 基本上用 Triton 重新实现了采样逻辑，以获得更好的数值/内存控制以及优化空间。

### Gumbel 采样算子核

MRV2 引入了一个 Triton Gumbel 采样算子核，它避免了显式地将 Softmax 实例化为张量，并利用来自种子（Seed）输入的无状态算子核内随机数生成器（RNG）。

### 高效的 Top-K Logprobs

V1 在计算 Top-K 之前会将全词表的 Logprobs 实例化。MRV2 则先从 Logits 中识别出 Top-K 的 Token，然后仅针对选定的 Token 计算 Logprobs。这显著降低了 GPU 内存占用的峰值。

### 内存高效的 Prompt Logprobs

MRV2 支持更细粒度的分块（Chunking），包括在单个 Prompt 内部进行分块，以避免长 Prompt 上的内存激增。

### 更好地兼容投机解码

MRV2 没有将每个请求的采样状态展开以匹配每个 Logit 的形状，而是在算子核内部使用间接寻址（`idx_mapping`）将每个 Logits 向量映射到正确的请求状态。这简化了对复杂采样参数和 Logits 处理器（Logits Processors）的支持。

## 7. 模块化 (Modularity)

MRV2 强调模块化。相比于 V1 庞大而错综复杂的 `gpu_model_runner.py`，MRV2 将特性逻辑拆分到专用的文件中（例如 `mrope_utils.py`、`penalties.py` 等）。

它还将模型输入整合到了 `InputBatch` 类中，减少了与 Model Runner 属性的直接耦合。

## 8. 不滥用 `dummy_run`

在 V1 中，`dummy_run` 承担了太多的职责：

- 初始内存分析（Profiling）和 `torch.compile`
- CUDA 图捕获 (CUDA graph capture)
- 热身 (Warmups)
- 用于 EP+DP 的空 DP 前向传播

MRV2 简化了这一点：

1. `execute_model` 支持空运行（Dummy runs）而不影响状态。
2. `dummy_run` 将分析、热身和空 DP 前向传播委托给 `execute_model`。
3. CUDA 图捕获使用单独的专用路径。

这降低了复杂度，并消除了由于 `execute_model` 和 `dummy_run` 行为差异引起的 Bug。

## 9. 显式 CUDA 图管理

V1 的 CUDA 图处理是隐式的，且难以推导。MRV2 使用 `CUDAGraphManager`，通过标准的 PyTorch API 显式地捕获和启动完整的 CUDA 图。

这使得图的生命周期和执行模式决策更加清晰且易于扩展。例如：MRV2 可以将多个草稿模型（Draft model）的前向传播捕获到一个 CUDA 图中。

## 开发哲学 (Development Philosophy)

MRV2 的修改应当符合更高的代码质量标准。随着与 V1 特性差距的填补，特性应当在 MRV2 的设计背景下从第一性原理出发进行重新审视，而不是简单地快速移植 V1 的行为。

一个关键要求是保持模块化和清晰的抽象边界，即使这需要前期进行更多的设计迭代。
