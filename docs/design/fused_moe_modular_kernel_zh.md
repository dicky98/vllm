# Fused MoE 模块化算子核 (Fused MoE Modular Kernel)

## 简介 (Introduction)

`FusedMoEModularKernel` 具体实现于 [这里](../../vllm/model_executor/layers/fused_moe/modular_kernel.py)。

基于输入激活值（Input activations）的格式，`FusedMoE` 的实现大体上可分为两类：

* 连续的 / 标准的 / 非批处理的 (Contiguous / Standard / Non-Batched)，以及
* 批处理的 (Batched)

!!! note "注意"
    在本文档中，“连续的（Contiguous）”、“标准的（Standard）”和“非批处理的（Non-Batched）”这三个术语是可以互换使用的。

输入激活值的格式完全取决于所使用的 All2All 分发（All2All Dispatch）。

* 在**连续的**变体中，All2All Dispatch 返回的激活值是一个形状为 `(M, K)` 的连续张量，以及形状为 `(M, num_topk)` 的 TopK ID 和 TopK 权重。具体的示例可参考 `DeepEPHTPrepareAndFinalize`。
* 在**批处理的**变体中，All2All Dispatch 返回的激活值是一个形状为 `(num_experts, max_tokens, K)` 的张量。在这里，流向同一个专家的激活值/Token 会被批处理在一起。请注意，并非该张量中的所有条目都是有效的。激活值张量通常伴随着一个大小为 `num_experts` 的 `expert_num_tokens` 张量，其中 `expert_num_tokens[i]` 表示流向第 `i` 个专家的有效 Token 的数量。具体的示例可参考 `DeepEPLLPrepareAndFinalize`。

无论是连续还是批处理变体，`FusedMoE` 操作通常由多个操作组成，如下面的图表所示：

![非批处理 FusedMoE](../assets/design/fused_moe_modular_kernel/fused_moe_non_batched.png)

![批处理 FusedMoE](../assets/design/fused_moe_modular_kernel/fused_moe_batched.png)

!!! note "注意"
    在操作方面，批处理和非批处理情况之间的主要区别在于 Permute / Unpermute（置换/逆置换）操作。所有其他操作都是相同的。

## 动因 (Motivation)

从图表中可以看出，这其中包含了非常多的操作，且每个操作都可以有多种不同的实现。这些操作拼凑在一起以构成一个有效的 `FusedMoE` 实现的方式数量迅速增加，变得难以处理。模块化算子核（Modular Kernel）框架通过将这些操作分组到不同的逻辑组件中来解决这一问题。这种宽泛的分类使得各种组合变得可控，并防止了代码重复。这也将 All2All Dispatch & Combine 的实现与 `FusedMoE` 的具体实现解耦，使它们能够独立开发和测试。此外，模块化算子核框架为不同的组件引入了抽象类（Abstract classes），从而为未来的实现提供了一个定义良好的骨架。

本文档的其余部分将重点讨论连续/非批处理的情况。外推到批处理情况应当是非常直观的。

## 模块化算子核组件 (ModularKernel Components)

`FusedMoEModularKernel` 将 `FusedMoE` 操作分为 3 个部分：

1. `TopKWeightAndReduce`
2. `FusedMoEPrepareAndFinalizeModular`
3. `FusedMoEExpertsModular`

### TopKWeightAndReduce

TopK 权重应用（TopK Weight Application）和归约（Reduction）组件发生在 Unpermute 操作之后、All2All Combine 之前。请注意，`FusedMoEExpertsModular` 负责 Unpermute，而 `FusedMoEPrepareAndFinalizeModular` 负责 All2All Combine。在 `FusedMoEExpertsModular` 中进行 TopK 权重应用和归约是有价值的。但是有些实现选择在 `FusedMoEPrepareAndFinalizeModular` 中进行。为了支持这种灵活性，我们提供了一个 `TopKWeightAndReduce` 抽象类。

请在 [这里](../../vllm/model_executor/layers/fused_moe/topk_weight_and_reduce.py) 查看 `TopKWeightAndReduce` 的实现。

`FusedMoEPrepareAndFinalizeModular::finalize()` 方法接受一个 `TopKWeightAndReduce` 参数，该参数在方法内部被调用。
`FusedMoEModularKernel` 作为 `FusedMoEExpertsModular` 和 `FusedMoEPrepareAndFinalize` 实现之间的桥梁，以决定 TopK 权重应用和归约发生在何处。

* 如果 `FusedMoEExpertsModular` 实现自身处理权重应用和归约，则 `FusedMoEExpertsModular::finalize_weight_and_reduce_impl` 方法返回 `TopKWeightAndReduceNoOp`。
* 如果 `FusedMoEExpertsModular` 实现需要 `FusedMoEPrepareAndFinalizeModular::finalize()` 来处理权重应用和归约，则 `FusedMoEExpertsModular::finalize_weight_and_reduce_impl` 方法返回 `TopKWeightAndReduceContiguous` / `TopKWeightAndReduceNaiveBatched` / `TopKWeightAndReduceDelegate`。

### FusedMoEPrepareAndFinalizeModular

`FusedMoEPrepareAndFinalizeModular` 抽象类暴露了 `prepare`、`prepare_no_receive` 和 `finalize` 函数。
`prepare` 函数负责输入激活值的量化（Quantization）和 All2All Dispatch。如果实现了 `prepare_no_receive`，它类似于 `prepare`，但它不会等待接收来自其他 Worker 的结果。相反，它返回一个“接收者（receiver）”回调，必须调用该回调以等待 Worker 的最终结果。并非所有 `FusedMoEPrepareAndFinalizeModular` 类都必须支持此方法，但如果它可用，则可用于将工作与初始的 All-to-All 通信交错进行（例如，将共享专家与 Fused 专家交错）。`finalize` 函数负责调用 All2All Combine。此外，`finalize` 函数可能会或可能不会执行 TopK 权重应用和归约（请参考 TopKWeightAndReduce 部分）。

![FusedMoEPrepareAndFinalizeModular 模块](../assets/design/fused_moe_modular_kernel/prepare_and_finalize_blocks.png)

### FusedMoEExpertsModular

`FusedMoEExpertsModular` 类是 MoE 核心计算发生的地方。`FusedMoEExpertsModular` 抽象类暴露了几个重要的函数：

* `apply()`
* `workspace_shapes()`
* `finalize_weight_and_reduce_impl()`

#### apply()

在 `apply` 方法中，各实现会执行：

* Permute（置换）
* 与权重 W1 的矩阵乘法（Matmul）
* 激活函数 (Act) + 元素相乘 (Mul)
* 量化 (Quantization)
* 与权重 W2 的矩阵乘法 (Matmul)
* Unpermute（逆置换）
* （可选）TopK 权重应用 + 归约

#### workspace_shapes()

核心的 `FusedMoE` 实现会执行一系列操作。为其中每一个操作分别创建输出内存将是低效的。为此，各个实现需要在 `workspace_shapes()` 方法的输出中声明 2 个工作空间形状（Workspace shapes）、工作空间的数据类型以及 `FusedMoE` 的输出形状。此信息用于在 `FusedMoEModularKernel::forward()` 中分配工作空间张量和输出张量，并传递给 `FusedMoEExpertsModular::apply()` 方法。这些工作空间随后可用作 `FusedMoE` 实现中的中间缓冲区。

#### finalize_weight_and_reduce_impl()

有时在 `FusedMoEExpertsModular::apply()` 内部进行 TopK 权重应用和归约会很高效。可以在 [这里](https://github.com/vllm-project/vllm/pull/20228) 查看一个示例。我们设计了 `TopKWeightAndReduce` 抽象类以支持此类实现。请参考 TopKWeightAndReduce 部分。
`FusedMoEExpertsModular::finalize_weight_and_reduce_impl()` 返回实现希望 `FusedMoEPrepareAndFinalizeModular::finalize()` 使用的 `TopKWeightAndReduce` 对象。

![FusedMoEExpertsModular 模块](../assets/design/fused_moe_modular_kernel/fused_experts_blocks.png)

### FusedMoEModularKernel

`FusedMoEModularKernel` 由 `FusedMoEPrepareAndFinalizeModular` 和 `FusedMoEExpertsModular` 对象组成。
`FusedMoEModularKernel` 的伪代码/轮廓：

```py
class FusedMoEModularKernel:
    def __init__(self,
                 prepare_finalize: FusedMoEPrepareAndFinalizeModular,
                 fused_experts: FusedMoEExpertsModular):

        self.prepare_finalize = prepare_finalize
        self.fused_experts = fused_experts

    def forward(self, DP_A):

        Aq, A_scale, _, _, _ = self.prepare_finalize.prepare(DP_A, ...)

        workspace13_shape, workspace2_shape, _, _ = self.fused_experts.workspace_shapes(...)

        # 分配工作空间 (workspaces)
        workspace_13 = torch.empty(workspace13_shape, ...)
        workspace_2 = torch.empty(workspace2_shape, ...)

        # 执行 fused_experts
        fe_out = self.fused_experts.apply(Aq, A_scale, workspace13, workspace2, ...)

        # 如果 fused_experts 实现自己执行了 TopK 权重应用和归约，
        # war_impl 将是一个 TopKWeightAndReduceNoOp 类型的对象。
        war_impl = self.fused_experts.finalize_weight_and_reduce_impl()

        output = self.prepare_finalize.finalize(fe_out, war_impl,...)

        return output
```

## 实操指南 (How-To)

### 如何添加一个 FusedMoEPrepareAndFinalizeModular 类型

通常，一个 `FusedMoEPrepareAndFinalizeModular` 类型会由一个 All2All Dispatch & Combine 实现 / Kernel 所支持。例如：

* `DeepEPHTPrepareAndFinalize` 类型由 DeepEP 高吞吐量（High-Throughput）All2All Kernel 支持。
* `DeepEPLLPrepareAndFinalize` 类型由 DeepEP 低延迟（Low-Latency）All2All Kernel 支持。

#### 步骤 1：添加一个 All2All 管理器

All2All 管理器的目的是设置 All2All Kernel 的实现。`FusedMoEPrepareAndFinalizeModular` 的实现通常会从 All2All 管理器获取一个 Kernel 实现的“句柄（handle）”，以调用 Dispatch 和 Combine 函数。请在 [这里](../../vllm/distributed/device_communicators/all2all.py) 查看 All2All 管理器的具体实现。

#### 步骤 2：添加一个 FusedMoEPrepareAndFinalizeModular 类型

本节描述了 `FusedMoEPrepareAndFinalizeModular` 抽象类暴露的各个函数的意义。

- `FusedMoEPrepareAndFinalizeModular::prepare()`：该方法实现了量化和 All2All Dispatch。通常它会调用相关 All2All 管理器中的 Dispatch 函数。
- `FusedMoEPrepareAndFinalizeModular::has_prepare_no_receive()`：指示该子类是否实现了 `prepare_no_receive`。默认为 `False`。
- `FusedMoEPrepareAndFinalizeModular::prepare_no_receive()`：该方法实现量化和 All2All Dispatch。它不会等待分发操作的结果，而是返回一个可以被调用以等待最终结果的 thunk 函数。通常会调用相关 All2All 管理器中的 Dispatch 函数。
- `FusedMoEPrepareAndFinalizeModular::finalize()`：执行 All2All Combine，并可能在其中处理 TopK 权重应用和归约。通常会调用相关 All2All 管理器中的 Combine 函数。
- `FusedMoEPrepareAndFinalizeModular::activation_format()`：如果 prepare 方法的输出（即 All2All dispatch 结果）是批处理格式，则返回 `FusedMoEActivationFormat.BatchedExperts`；否则返回 `FusedMoEActivationFormat.Standard`。
- `FusedMoEPrepareAndFinalizeModular::topk_indices_dtype()`：TopK ID 的数据类型。一些 All2All Kernel 对 TopK ID 的数据类型有严格要求。此要求会传递给 `FusedMoe::select_experts` 函数以使之得到满足。如果没有严格要求，则返回 `None`。
- `FusedMoEPrepareAndFinalizeModular::max_num_tokens_per_rank()`：一次提交给 All2All Dispatch 的最大 Token 数量。
- `FusedMoEPrepareAndFinalizeModular::num_dispatchers()`：分发单元的总数。该值决定了 Dispatch 输出的大小。Dispatch 输出的形状为 `(num_local_experts, max_num_tokens, K)`。这里 `max_num_tokens = num_dispatchers() * max_num_tokens_per_rank()`。

我们建议选择一个与您的 All2All 实现最接近的现有 `FusedMoEPrepareAndFinalizeModular` 实现作为参考。

### 如何添加一个 FusedMoEExpertsModular 类型

`FusedMoEExpertsModular` 执行 FusedMoE 的核心计算操作。抽象类暴露的各个函数及其意义如下：

- `FusedMoEExpertsModular::activation_formats()`：返回所支持的输入和输出激活值格式（即连续格式 / 批处理格式）。
- `FusedMoEExpertsModular::supports_expert_map()`：如果该实现支持专家映射（Expert map），则返回 `True`。
- `FusedMoEExpertsModular::workspace_shapes()` / `FusedMoEExpertsModular::finalize_weight_and_reduce_impl` / `FusedMoEExpertsModular::apply`：参考上面的 `FusedMoEExpertsModular` 章节。

### FusedMoEModularKernel 初始化

`FusedMoEMethodBase` 类有 3 个方法共同负责创建 `FusedMoEModularKernel` 对象。它们是：

* `maybe_make_prepare_finalize`
* `select_gemm_impl`
* `init_prepare_finalize`

#### maybe_make_prepare_finalize

在适当的情况下（例如当启用了 EP + DP 时），`maybe_make_prepare_finalize` 方法负责构建 `FusedMoEPrepareAndFinalizeModular` 的一个实例。基类方法目前为 EP+DP 情况构建所有的 `FusedMoEPrepareAndFinalizeModular` 对象。派生类可以重写此方法以针对不同场景构建 prepare/finalize 对象。例如，`ModelOptNvFp4FusedMoE` 可以为 EP+TP 场景构建一个 `FlashInferCutlassMoEPrepareAndFinalize`。
请参考以下类中的实现：

* `ModelOptNvFp4FusedMoE`

#### select_gemm_impl

`select_gemm_impl` 方法在基类中未定义。派生类有责任实现一个方法来构建一个有效/合适的 `FusedMoEExpertsModular` 对象。
请参考以下派生类中的实现：

* `UnquantizedFusedMoEMethod`
* `CompressedTensorsW8A8Fp8MoEMethod`
* `CompressedTensorsW8A8Fp8MoECutlassMethod`
* `Fp8MoEMethod`
* `ModelOptNvFp4FusedMoE`

#### init_prepare_finalize

根据输入和环境变量设置，`init_prepare_finalize` 方法创建适当的 `FusedMoEPrepareAndFinalizeModular` 对象。随后，该方法向 `select_gemm_impl` 查询适当的 `FusedMoEExpertsModular` 对象，并构建 `FusedMoEModularKernel` 对象。

请查看 [init_prepare_finalize](https://github.com/vllm-project/vllm/blob/1cbf951ba272c230823b947631065b826409fa62/vllm/model_executor/layers/fused_moe/layer.py#L188)。
**重要提示**：`FusedMoEMethodBase` 派生类在其 `apply` 方法中使用 `FusedMoEMethodBase::fused_experts` 对象。当设置允许构建有效的 `FusedMoEModularKernel` 对象时，我们会用它重写 `FusedMoEMethodBase::fused_experts`。这实质上使得派生类无需关心使用了何种 `FusedMoE` 实现。

### 如何进行单元测试 (How To Unit Test)

我们在 [test_modular_kernel_combinations.py](../../tests/kernels/moe/test_modular_kernel_combinations.py) 中编写了 `FusedMoEModularKernel` 的单元测试。

该单元测试遍历了 `FusedMoEPrepareAndFinalizeModular` 和 `FusedMoEPremuteExpertsUnpermute` 类型的全部组合，并在它们兼容时运行一些正确性测试。
如果您要添加某些 `FusedMoEPrepareAndFinalizeModular` / `FusedMoEExpertsModular` 实现，请执行以下操作：

1. 分别将实现类型添加到 [mk_objects.py](../../tests/kernels/moe/modular_kernel_tools/mk_objects.py) 中的 `MK_ALL_PREPARE_FINALIZE_TYPES` 和 `MK_FUSED_EXPERT_TYPES`。
2. 更新 [/tests/kernels/moe/modular_kernel_tools/common.py](../../tests/kernels/moe/modular_kernel_tools/common.py) 中的以下方法：
   `Config::is_batched_prepare_finalize()`、`Config::is_batched_fused_experts()`、`Config::is_standard_fused_experts()`、`Config::is_fe_16bit_supported()`、`Config::is_fe_fp8_supported()`、`Config::is_fe_block_fp8_supported()`。

执行这些操作将把新的实现添加到测试套件中。

### 如何检查 `FusedMoEPrepareAndFinalizeModular` 和 `FusedMoEExpertsModular` 的兼容性

单元测试文件 [test_modular_kernel_combinations.py](../../tests/kernels/moe/test_modular_kernel_combinations.py) 也可以作为独立脚本执行。
示例：`python3 -m tests.kernels.moe.test_modular_kernel_combinations --pf-type DeepEPLLPrepareAndFinalize --experts-type BatchedTritonExperts`
作为一种用途，该脚本可用于测试 `FusedMoEPrepareAndFinalizeModular` 与 `FusedMoEExpertsModular` 的兼容性。当传入不兼容的类型时，该脚本将报错。

### 如何进行性能分析 (How To Profile)

请查看 [profile_modular_kernel.py](../../tests/kernels/moe/modular_kernel_tools/profile_modular_kernel.py)。
该脚本可用于为任何兼容的 `FusedMoEPrepareAndFinalizeModular` 和 `FusedMoEExpertsModular` 类型生成单次 `FusedMoEModularKernel::forward()` 调用的 Torch 性能追踪（Trace）。
示例：`python3 -m tests.kernels.moe.modular_kernel_tools.profile_modular_kernel --pf-type DeepEPLLPrepareAndFinalize --experts-type BatchedTritonExperts`

## FusedMoEPrepareAndFinalizeModular 的实现

请参阅 [Fused MoE 算子核特性](./moe_kernel_features_zh.md#fused-moe-all2all) 以获取所有可用的模块化 prepare 和 finalize 子类的列表。

## FusedMoEExpertsModular 的实现

请参阅 [Fused MoE 算子核特性](./moe_kernel_features_zh.md#fused-experts) 以获取所有可用的模块化专家的列表。
