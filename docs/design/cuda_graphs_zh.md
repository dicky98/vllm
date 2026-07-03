# CUDA 图 (CUDA Graphs)

本文档介绍了 vLLM v1 中除先前 [torch.compile 集成](torch_compile.md) 之外的全新 CUDA 图（CUDA Graphs）模式。简而言之，我们：

1. 增加了灵活的 `cudagraph_mode` 配置。
2. 使得完整的 CUDA 图（Full CUDA Graphs）支持与编译正交（即互不干扰）。
3. 引入了 CUDA 图分发器（CUDA Graphs dispatcher）作为中央控制器，自动针对每个批次选择所需的运行时模式和 CUDA 图。

在本文档中，我们将讨论：

* [动因](#motivation)
* [CUDA 图模式](#cudagraphmodes)
* [详细设计](#detailed-design)
* [不同 CUDA 图模式的使用示例](#usage-guide)
* [视觉编码器 (ViT) CUDA 图](cuda_graphs_multimodal.md)

!!! note "注意"
    在本文档中，我们将纯 Decode 阶段（`max_query_len=1`）或投机解码阶段（`max_query_len = 1 + num_spec_tokens`）称为**均匀 Decode（uniform decode）**批次，反之则称为**非均匀（non-uniform）**批次（即 Prefill 阶段或 Prefill-Decode 混合阶段的批次）。

!!! note "注意"
    以下内容大多基于 <https://github.com/vllm-project/vllm/pull/20059> 的最新一次 Commit。

## 动因 (Motivation)

最初的“分段式编译（piecewise compilation）”设计是为了允许分段式 CUDA 图捕获（piecewise cudagraph capture），从而排除 CUDA 图不支持的操作（主要是 Attention 计算）。这在保持与所有注意力后端兼容的同时，从 CUDA 图中获得了一定的加速。后来，我们通过不进行分段式编译增加了对“完整 CUDA 图（full cudagraphs）”的支持，以便在注意力机制本身支持 CUDA 图的情况下进一步降低延迟。

然而，编译与 CUDA 图捕获之间的这种紧密耦合导致了“要么全有、要么全无”的体验，缺乏灵活性。许多注意力后端也还没准备好进行统一的“完整” CUDA 图捕获（例如，目前仅 FlashAttention 3 支持），或者仅在纯 Decode 批次中支持 CUDA 图（例如 Flashinfer、FlashMLA 和 Mamba 等）。这导致了令人困惑的性能与兼容性折中、不一致的 CUDA 图支持，以及日益复杂的代码结构。

这促使我们寻求一种更细粒度的 CUDA 图解决方案，其具备以下特性：

* 显式感知 Prefill/Mixed 阶段或（均匀）Decode 阶段的批次，并分别对它们进行 CUDA 图捕获。
* 将 CUDA 图捕获逻辑与编译逻辑分离开来（在可行范围内），以实现特性的正交性。这意味着：
    * 使用同一个编译后的图，同时支持捕获分段式（Piecewise）和完整（Full）的 CUDA 图。
    * 支持在不进行编译的情况下进行完整的 CUDA 图捕获。
* 在运行时根据批次的具体组成，在完整和分段式 CUDA 图之间进行动态分发切换。
* 集中控制 CUDA 图的行为，以降低代码复杂度，并提供更多的扩展性。

这些特性对于所有类型的启动/性能折中与特性支持，在 CUDA 图捕获和编译方面提供了最大的灵活性。

## CUDA 图模式 (CudagraphModes)

[CUDAGraphMode][vllm.config.compilation.CUDAGraphMode] 是您在 `CompilationConfig.cudagraph_mode` 中调节的单一旋钮：

* `NONE` — 关闭 CUDA 图。适合调试。
* `PIECEWISE` — 单模式策略（以前的默认行为）。它是最灵活的：注意力或其他与 CUDA 图不兼容的操作保持 Eager 模式执行，其余所有部分都放入 CUDA 图中。这需要分段式编译。
* `FULL` — 单模式策略，仅对非均匀批次捕获完整的 CUDA 图，然后均匀 Decode 批次复用相同 `batch_size` 的非均匀批次的 CUDA 图（因为它们是兼容的）；适合小模型或 Prompt 较短的工作负载。
* `FULL_DECODE_ONLY` — 仅对均匀 Decode 阶段捕获完整的 CUDA 图，对 Prefill/Mixed 等不启用 CUDA 图；适用于 Prefill 不那么重要的 P/D（预填充/解码分离）设置下的 Decode 实例，这样我们可以节省 `PIECEWISE` CUDA 图所需的显存。
* `FULL_AND_PIECEWISE` — （默认模式）对均匀 Decode 阶段采用完整的 CUDA 图，对其他阶段采用分段式 CUDA 图；这通常是性能最高的设置（特别是对于小模型或 MoE 的低延迟场景），但它也需要最大的显存开销，且捕获（Capture）耗时最长。

默认值：如果您在 v1 上启用了分段式编译，我们默认使用 `FULL_AND_PIECEWISE` 以获得更好的性能（对于池化模型，仍为 `PIECEWISE`）。否则（例如分段式编译不可用时），我们默认使用 `NONE`。

虽然 `NONE`、`PIECEWISE` 和 `FULL` 是单模式配置，分别等同于以前的 Eager 执行、分段式 CUDA 图和完整 CUDA 图的实现，但 `FULL_DECODE_ONLY` 和 `FULL_AND_PIECEWISE` 是新追加的双模式配置，它们需要通过分发机制（Dispatching）根据运行时的批次动态地在具体的运行时模式之间进行切换。

!!! note "注意"
    在这里，单模式 `NONE`、`PIECEWISE` 和 `FULL` 被视为 CUDA 图分发的“运行时模式”。如果使用双模式，分发器将始终根据批次组成动态分发到其成员模式之一（如果在没有合适的 CUDA 图可用时，也可能会分发到 `NONE`）。

虽然级联注意力（Cascade attention）与 CUDA 图不兼容，但它现在已与所有可能的 CUDA 图模式配置兼容。如果某个批次使用了级联注意力，若有 `PIECEWISE` 模式可用，它总是会被分发到 `PIECEWISE` 模式（否则为 `NONE`）。

!!! note "注意"
    并非所有的 CUDA 图模式都与每个注意力后端兼容。我们会自动将模式“降级（downgrade）”到最近的受支持模式。例如，如果某个后端仅支持纯 Decode/均匀批次的 CUDA 图，且启用了分段式编译，我们会将 `FULL` 转换为 `FULL_AND_PIECEWISE`；否则将其转换为 `FULL_DECODE_ONLY`。

## 详细设计 (Detailed Design)

### 概述

新的 CUDA 图逻辑构建在分段式编译的基础之上，并支持双 CUDA 图运行时模式切换。该系统包含以下核心组件：

* [CUDAGraphWrapper][vllm.compilation.cuda_graph.CUDAGraphWrapper]：包装器，负责在被包装的可调用对象上进行 CUDA 图的捕获与回放（Replay）。
* [CudagraphDispatcher][vllm.v1.cudagraph_dispatcher.CudagraphDispatcher]：中央控制器，包含关于 CUDA 图的单一事实来源，并处理它们之间的分发。
* [CUDAGraphMode][vllm.config.compilation.CUDAGraphMode]：枚举类型，描述支持的模式和运行时模式（如上所述）。
* [BatchDescriptor][vllm.forward_context.BatchDescriptor]：运行时批次的唯一表示，用于进行分发。

请参见下图，以快速对比以前与当前搭配 Inductor 编译的 CUDA 图设计模式。我们可以看到，以前的 CUDA 图逻辑和编译逻辑紧密耦合在 vLLM 的 `PiecewiseBackend` 中，且 CUDA 图是被 `batch_size` 隐式、被动地分发。现在，CUDA 图逻辑被分离到了 `CUDAGraphWrapper` 类中，共同负责完整和分段式 CUDA 图的功能，而分发则是通过 `CudagraphDispatcher` 显式地基于**运行时模式**加上 `BatchDescriptor` 作为**分发键（Dispatch key）**来完成。

**重构前：**

![重构前设计](../assets/design/cuda_graphs/previous_design.png)

**重构后：**

![重构后设计](../assets/design/cuda_graphs/current_design.png)

### `BatchDescriptor`

`BatchDescriptor` 是 `ForwardContext` 内部的一个组件，与 CUDA 图运行时模式一起，作为运行时分发键的核心结构。其原型为：

```python
class BatchDescriptor(NamedTuple):
    num_tokens: int
    num_reqs: int
    uniform: bool = False
    has_lora: bool = False
```

其中 `num_tokens` 可以是填充后的 Token 长度，`uniform` 表示所有请求的查询长度是否相同。许多注意力后端仅在批次为均匀批次时才支持完整 CUDA 图；纯 Decode 批次是均匀的，但其查询长度可能不为 1（即 `num_tokens == num_reqs` ），这发生在投机解码的验证阶段，此时 Decode 批次的查询长度将是 `1 + num_speculative_tokens`。

此结构的目标是用最少的可能项来唯一标识一个（已填充的）批次，以对应一个 CUDA 图。

!!! note "注意"
    未来，`BatchDescriptor` 的原型可能会扩展以支持更一般的情况，例如加入更多项（如 `uniform_query_len` 以支持多个不同的均匀 Decode 长度设置，参见 <https://github.com/vllm-project/vllm/pull/23679>），或为支持输入不一定感知 Token 长度的模型（例如某些多模态输入）而进行的修改。

### `CudagraphDispatcher`

`CudagraphDispatcher` 负责维护两套有效的分发键，一套用于 `FULL` 运行时模式，另一套用于 `PIECEWISE` 运行时模式，并在执行模型前向传播之前分发正确的运行时模式和分发键。它接收初始键（即针对已填充输入的粗略 `batch_descriptor`），返回选定的运行时模式和最终的 `batch_descriptor`，然后通过前向上下文（Forward contexts）将此决策告知 `CUDAGraphWrapper` 实例。请注意，`CudagraphDispatcher` 是可用 CUDA 图键的唯一事实来源，而 `CUDAGraphWrapper` 实例可以盲目地信任前向上下文来分发到对应的 CUDA 图。这使我们可以简化包装器代码并将逻辑集中在分发器中。

分发键通过分发器的 `initialize_cudagraph_keys` 方法进行初始化，该方法在所有可能的注意力后端初始化完毕后由 `GPUModelRunner` 调用。未来我们可以在这里实现更多花样，例如“准备”各种 CUDA 图的组合。目前，我们仅根据编译配置中 `cudagraph_mode` 的 `decode_mode` / `mixed_mode` 与 `cudagraph_capture_sizes` 的有效组合来追加可用键。

分发代码看起来像这样：

```python
batch_descriptor=BatchDescriptor(num_tokens=num_input_tokens, uniform_decode=...)
runtime_mode, batch_descriptor = cudagraphdispatcher.dispatch(batch_descriptor)
# 执行
with set_forward_context(
    ..., 
    cudagraph_runtime_mode=runtime_mode, 
    batch_descriptor=batch_descriptor,
):
     output = self.model(...)
```

在 `dispatch()` 方法内部，分发器将搜索适当的 CUDA 图运行时模式和现有的分发键并返回。我们基本上按照优先级 `FULL` > `PIECEWISE` > `None` 来搜索现有的键。如果分发键不存在，则默认返回 `NONE` 模式以进行 Eager 模式执行。具体实现可以在[这里](https://github.com/vllm-project/vllm/blob/main/vllm/v1/cudagraph_dispatcher.py#L91)找到。

以下是模型执行器中运行时工作流的简化演示：
![执行器运行时](../assets/design/cuda_graphs/executor_runtime.png)

### `CUDAGraphWrapper`

`CUDAGraphWrapper` 实例包装了一个可运行对象，并简单地模仿该可运行对象，同时追加了 CUDA 图功能。每个包装器实例都绑定到一个特定的 `runtime_mode`（受限于 `PIECEWISE` 和 `FULL` 模式），并负责捕获/回放以及透传直接调用（Pass through）该可运行对象。在运行时，每个包装器将：

1. 从全局前向上下文中检查 `runtime_mode` 和 `batch_descriptor`（分发键）。
2. 如果 `runtime_mode` 为 `NONE`，或者 `runtime_mode` 与该包装器的模式不匹配，则直接调用可运行对象。
3. 否则（即 `runtime_mode` 与包装器的模式匹配），包装器将执行 CUDA 图捕获（如果键不存在，则创建一个新条目并将其缓存）或回放（如果键在缓存中存在）。

上述步骤基于一个假设：CUDA 图包装器将直接信任前向上下文中的内容（由分发器控制）。这使我们能够简化和集中逻辑，降低复杂度，同时也降低了包装器与分发器之间状态不匹配的风险。这也使得同一包装器类可以同时复用于 `FULL` 和 `PIECEWISE` 运行时模式。具体实现参见[这里](https://github.com/vllm-project/vllm/blob/f751e50b7a2aae3110d83ed0d88202fc91b3e78a/vllm/compilation/cuda_graph.py#L106)。

#### 嵌套包装器设计 (Nested Wrapper design)

使完整 CUDA 图和分段式 CUDA 图能够共存且兼容的核心机制是**嵌套式 CUDA 图包装器设计**，该设计建立在仅包含单个分段式 FX 图的分段式编译之上。我们在整个模型的外部包装一个 `FULL` 模式的包装器，以实现完整 CUDA 图的功能；同时，每个分段式后端在编译内部都会通过一个 `PIECEWISE` 模式的包装器进行包装。

下面的流程图清晰地描述了它的工作原理：
![包装器流程](../assets/design/cuda_graphs/wrapper_flow.png)

因此，对于 `FULL` 运行时模式，由于分段式包装器未被激活，捕获/回放完整的 CUDA 图是安全的。对于 `PIECEWISE` 模式情况也类似，因为 `FULL` 模式包装器和 `PIECEWISE` 模式包装器之间没有冲突。而对于 `NONE` 运行时模式，`FULL` 和 `PIECEWISE` 包装器均不会被激活，因此我们只需回退到 Eager 模式执行。

### 完整 CUDA 图捕获与热身

当运行器首次调用模型前向传播（使用 `_dummy_run`）且处于非 `NONE` 运行时模式时，会发生 CUDA 图的捕获。为了捕获完整的 CUDA 图，我们通过正确设置注意力元数据（Attn metadata）来显式捕获不同的情况（即 Prefill/Mixed 批次或均匀 Decode 批次），以确保底层的注意力后端启动所需的算子核子程序。为了区分 Prefill/Mixed 批次还是均匀 Decode 批次，最关键的属性是注意力元数据中的 `max_query_len`（对大多数注意力后端都适用）。对于均匀 Decode，我们将其设置为所需的 `uniform_query_len`；否则，对于非均匀 Decode 批次，我们直接将其设置为 `num_tokens`。

CUDA 图包装器不再管理热身（Warm-up）逻辑。热身过程现在由 GPU 模型运行器直接控制，其中分配 `NONE` 运行时模式以进行 Eager 执行以用于热身。在为完整 CUDA 图进行热身时，在热身 `dummy_run` 调用期间显式运行注意力（Attention）也至关重要。

## 注意力后端的 CUDA 图兼容性

为了表明注意力后端的 CUDA 图兼容性，我们引入了一个新的枚举类型 [AttentionCGSupport][vllm.v1.attention.backend.AttentionCGSupport]，它用于追踪注意力后端支持 CUDA 图的能力。其值按能力大小排序，即 `ALWAYS` > `UNIFORM_BATCH` > `UNIFORM_SINGLE_TOKEN_DECODE` > `NEVER`。

```python
class AttentionCGSupport(enum.Enum):
    """ 注意力后端支持 CUDA 图的常量
    这里我们不考虑级联注意力（Cascade attention），因为目前
    它绝不支持 CUDA 图。"""

    ALWAYS = 3
    """始终支持 CUDA 图；支持 Mixed-Prefill-Decode"""
    UNIFORM_BATCH = 2
    """支持对查询长度都相同的批次使用 CUDA 图，这可用于投机解码，
       即 "decodes" 长度为 1 + num_speculative_tokens"""
    UNIFORM_SINGLE_TOKEN_DECODE = 1
    """仅支持对包含 query_len==1 decode 的批次使用 CUDA 图"""
    NEVER = 0
    """不支持 CUDA 图"""
```

假设我们拥有混合注意力后端（例如在 Mamba Mixer 模型中）。在这种情况下，我们寻求所有后端能力的最小值来确定模型的最终能力，并且我们可能会通过将模式降级到最适配的模式来解决不兼容的 CUDA 图模式。例如，如果最小能力是 `UNIFORM_BATCH`，则将 `FULL` 模式降级为 `FULL_AND_PIECEWISE` 模式；如果对于 -O3 编译模式最小能力是 `NEVER`，则降级为 `PIECEWISE` 模式。关于完整的降级回退策略（Fallback policy），请参阅 [此代码][vllm.v1.worker.gpu_model_runner.GPUModelRunner._check_and_update_cudagraph_mode]。

下表列出了在撰写本文档时支持完整 CUDA 图的后端：

| 注意力后端 (Attention Backend) | CG 支持度 (cudagraph_support) | 备注 |
| :---------------- | :---------------- | :------- |
| FlashAttention v2 | `UNIFORM_BATCH` | 实际上是 `ALWAYS`，但出于性能原因，通过规避方案回退到 `FULL_AND_PIECEWISE` |
| FlashAttention v3 | `ALWAYS` | 具有针对两类批次的统一程序，因此 `FULL` 模式很好 |
| Triton Attention | `ALWAYS` | 更倾向于使用 `FULL_AND_PIECEWISE`，因为对于 Prefill/Mixed 和纯 Decode 批次它有不同的 Kernel |
| AITER FlashAttention | `UNIFORM_BATCH` | |
| FlashInfer | `UNIFORM_SINGLE_TOKEN_DECODE` | 在 Blackwell 上使用 TRTLLM 注意力时会被设置为 `UNIFORM_BATCH` |
| FlashMLA | `UNIFORM_BATCH` | |
| FlashInferMLA | `UNIFORM_BATCH` | |
| FlashInferMLASparse | `UNIFORM_BATCH` | |
| AITER MLA | `UNIFORM_SINGLE_TOKEN_DECODE` | |
| CUTLASS MLA | `UNIFORM_SINGLE_TOKEN_DECODE` | |
| Mamba attention | `UNIFORM_SINGLE_TOKEN_DECODE` | |

未列出的后端均被声明为 `NEVER`。

## 使用指南 (Usage guide)

现在，命令行接口（CLI）直接为 `compilation_config` 使用大写的 CUDA 图模式字符串：`--compilation-config '{"cudagraph_mode": "..."}'`，其中 `...` 应该是 `NONE`、`PIECEWISE`、`FULL`、`FULL_DECODE_ONLY` 和 `FULL_AND_PIECEWISE` 之一。需要注意的是，所有与 `PIECEWISE` 相关的模式都需要分段式编译，所有与 `FULL` 相关的模式都需要注意力后端的 CUDA 图支持。例如：

```bash
vllm serve --model meta-llama/Llama-3.1-8B-Instruct --compilation-config '{"cudagraph_mode": "FULL_AND_PIECEWISE"}'
```

### Python 示例

```python
import os
os.environ.setdefault("VLLM_LOGGING_LEVEL", "DEBUG")

import vllm
from vllm.config import CUDAGraphMode

compilation_config = {"mode": 3, "cudagraph_mode": "FULL_AND_PIECEWISE"}
model = vllm.LLM(
    model="meta-llama/Llama-3.1-8B-Instruct",
    dtype="auto",
    compilation_config=compilation_config,
)
sampling_params = vllm.SamplingParams(
    temperature=0,  # 贪婪解码
    max_tokens=1024,
)
outputs = model.generate(
    ["My name is John and"],
    sampling_params=sampling_params,
)
```

### 分段式编译与全图自定义 Passes (注意力融合、序列并行)

不幸的是，一些自定义编译 Passes（如 `AttnQuantFusionPass` 和 `SequenceParallelismPass`）必须看到完整的图才能生效，因此它们与分段式编译不兼容。作为一个短期解决方案，当启用注意力融合时，我们通过设置 `splitting_ops=[]` 自动禁用分段式编译。我们使用 CUDA 图模式 `FULL` 或 `FULL_DECODE_ONLY`（取决于后端支持情况）。然而，这导致了另一种优化不兼容性与令人困惑的性能权衡。

长期来看，我们增加了在 Inductor 中而不是紧随 Dynamo 之后对图进行划分（Partition）的能力。这可以通过设置 `CompilationConfig.use_inductor_graph_partition=True` 启用，但该功能目前处于实验阶段，且仅在 `torch>=2.9` 时可用。由于它必须编译完整的图且无法复用分段编译伪影，这也增加了编译时间。一旦 vLLM 支持 2.9，我们计划将其作为默认方法，因为这也将加快分段式 CUDA 图的捕获。
