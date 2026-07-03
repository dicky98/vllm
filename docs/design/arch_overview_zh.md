# 架构概述 (Architecture Overview)

本文档提供 vLLM 架构的概述。

[TOC]

## 入口点 (Entrypoints)

vLLM 提供了许多用于与系统进行交互的入口点。下图展示了它们之间的关系。

![Entrypoints Diagram](../assets/design/arch_overview/entrypoints.excalidraw.png)

### LLM 类 (LLM Class)

`LLM` 类提供了用于执行离线推理（Offline Inference）的主要 Python 接口。所谓离线推理，是指不需要使用单独的模型推理服务器而直接与模型进行交互。

以下是 `LLM` 类的一个基本使用示例：

??? code

    ```python
    from vllm import LLM, SamplingParams

    # 定义输入提示词列表
    prompts = [
        "Hello, my name is",
        "The capital of France is",
        "The largest ocean is",
    ]

    # 定义采样参数
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

    # 用 OPT-125M 模型初始化 LLM 引擎
    llm = LLM(model="facebook/opt-125m")

    # 为输入提示词生成输出
    outputs = llm.generate(prompts, sampling_params)

    # 打印生成的输出结果
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
    ```

更多 API 细节可以在 API 文档的 [离线推理](../api/README.md#offline-inference) 章节中找到。

`LLM` 类的实现代码位于 [vllm/entrypoints/llm.py](../../vllm/entrypoints/llm.py)。

### 在线服务 (Online Serving)

vLLM 的第二个主要接口是通过其在线服务器进行服务。
该服务器可以使用 `vllm serve` 命令启动：

```bash
vllm serve <model>
```

`vllm` CLI 的代码实现位于 [vllm/entrypoints/cli/main.py](../../vllm/entrypoints/cli/main.py)。

有时您可能会看到直接使用 API 服务器入口点，而不是通过 `vllm` CLI 命令。例如：

```bash
python -m vllm.entrypoints.openai.api_server --model <model>
```

!!! warning "警告"

    `python -m vllm.entrypoints.openai.api_server` 方式已被废弃（deprecated），并在未来的版本中可能会变得不再受支持。

该部分代码实现位于 [vllm/entrypoints/openai/api_server.py](../../vllm/entrypoints/openai/api_server.py)。

关于 API 服务器的更多细节，可以在 [在线服务](../serving/online_serving/README.md) 文档中找到。

## V1 进程架构 (V1 Process Architecture)

vLLM V1 使用多进程架构来分离关注点并最大化吞吐量。了解这一架构对于在部署时合理规划 CPU 资源非常重要。其关键进程如下：

### API 服务器进程 (API Server Process)

API 服务器进程处理 HTTP 请求（例如兼容 OpenAI 的 API），执行输入预处理（分词 Tokenization、多模态数据加载），并将结果流式传回给客户端。它通过 ZMQ 套接字（ZMQ sockets）与引擎核心进程通信。

默认情况下，有 **1 个 API 服务器进程**。但当使用数据并行（Data Parallelism）时，API 服务器的数量会自动扩展以匹配数据并行大小。这也可以通过 `--api-server-count` 标志进行手动配置。每个 API 服务器都以多对多的拓扑结构通过 ZMQ 连接到**所有**引擎核心，从而使任何 API 服务器都能够将请求路由到任何引擎核心。每个 API 服务器进程使用多个 CPU 线程进行媒体加载（由 `VLLM_MEDIA_LOADING_THREAD_COUNT` 控制，默认为 8）。

相关代码参见 [vllm/entrypoints/openai/api_server.py](../../vllm/entrypoints/openai/api_server.py) 和 [vllm/v1/utils.py](../../vllm/v1/utils.py)。

### 引擎核心进程 (Engine Core Process)

引擎核心进程运行调度器（Scheduler）、管理 KV 缓存，并协调 GPU 工作进程（GPU Workers）之间的模型执行。它运行一个忙等待循环（busy loop），不断调度请求并将工作分派给 GPU 工作进程。

**每个数据并行 Rank 拥有 1 个引擎核心进程**。例如，使用 `--data-parallel-size 4` 时，会有 4 个引擎核心进程。

相关代码参见 [vllm/v1/engine/core.py](../../vllm/v1/engine/core.py) 和 [vllm/v1/engine/utils.py](../../vllm/v1/engine/utils.py)。

### GPU 工作进程 (GPU Worker Processes)

每个 GPU 都由一个专用的工作进程管理。工作进程加载模型权重、执行前向传播并管理 GPU 内存。工作进程与拥有它们的引擎核心进程进行通信。

**每个 GPU 拥有 1 个工作进程**。GPU 工作进程的总数等于每个引擎核心对应的 `tensor_parallel_size x pipeline_parallel_size`。

相关代码参见 [vllm/v1/executor/multiproc_executor.py](../../vllm/v1/executor/multiproc_executor.py) 和 [vllm/v1/worker/gpu_worker.py](../../vllm/v1/worker/gpu_worker.py)。

### DP 协调器进程 (DP Coordinator Process，有条件的)

当使用数据并行（`--data-parallel-size > 1`）时，会额外启动一个协调器进程，用于管理各 DP Rank 之间的负载均衡，并协调 MoE 模型同步的前向传播。

只有在启用数据并行时，才会有 **1 个 DP 协调器进程**。

相关代码参见 [vllm/v1/engine/coordinator.py](../../vllm/v1/engine/coordinator.py)。

### 进程数量汇总 (Process Count Summary)

对于拥有 `N` 个 GPU、`TP` 张量并行大小、`DP` 数据并行大小以及 `A` 个 API 服务器数量的部署：

| 进程类型 | 数量 | 备注 |
| - | - | - |
| API 服务器 | `A`（默认等于 `DP`） | 处理 HTTP 请求和输入预处理 |
| 引擎核心 | `DP`（默认为 1） | 调度器和 KV 缓存管理 |
| GPU 工作进程 | `N`（`= DP x PP x TP`） | 每个 GPU 一个，执行模型前向传播 |
| DP 协调器 | 如果 `DP > 1` 则为 1，否则为 0 | 跨 DP Rank 的负载均衡 |
| **总计** | **`A + DP + N`（如果 DP > 1 则再 + 1）** | |

例如，一个典型的单节点 4 GPU 部署（`vllm serve -tp=4`）拥有：

- 1 个 API 服务器 + 1 个 引擎核心 + 4 个 GPU 工作进程 = **6 个进程**

<figure markdown="1">
![V1 Process Architecture - TP=4](../assets/design/arch_overview/v1_process_architecture_tp4.png)
</figure>

一个 8 GPU 数据并行部署（`vllm serve -tp=2 -dp=4`）拥有：

- 4 个 API 服务器 + 4 个 引擎核心 + 8 个 GPU 工作进程 + 1 个 DP 协调器 = **17 个进程**

<figure markdown="1">
![V1 Process Architecture - TP=2, DP=4](../assets/design/arch_overview/v1_process_architecture_tp2_dp4.png)
</figure>

有关 CPU 资源规划的具体建议，请参阅 [GPU 部署的 CPU 资源](../configuration/optimization.md#cpu-resources-for-gpu-deployments)。

## LLM 引擎 (LLM Engine)

`LLMEngine` 和 `AsyncLLMEngine` 类是 vLLM 系统运行的核心，处理模型推理和异步请求的处理过程。

![LLMEngine Diagram](../assets/design/arch_overview/llm_engine.excalidraw.png)

### LLMEngine

`LLMEngine` 类是 vLLM 引擎的核心组件。它负责接收来自客户端的请求，并从模型中生成输出。`LLMEngine` 包含了输入预处理、模型执行（可能分布式跨越多个主机和/或 GPU）、调度和输出处理。

- **输入预处理**：使用指定的的分词器（Tokenizer）对输入文本进行分词处理。
- **调度**：选择在每个步骤中处理哪些请求。
- **模型执行**：管理语言模型的执行，包括跨多 GPU 的分布式执行。
- **输出处理**：处理模型生成的输出，将语言模型的 Token ID 解码为人类可读的文本。

`LLMEngine` 的实现代码位于 [vllm/engine/llm_engine.py](../../vllm/engine/llm_engine.py)。

### AsyncLLMEngine

`AsyncLLMEngine` 类是 `LLMEngine` 类的一个异步包装器。它使用 `asyncio` 来创建背景循环，以持续处理传入的请求。`AsyncLLMEngine` 是为在线服务设计的，能够处理多个并发请求并将输出流式传输给客户端。

兼容 OpenAI 的 API 服务器使用了 `AsyncLLMEngine`。在 [examples/applications/api_server/server.py](../../examples/applications/api_server/server.py) 中还有一个简单的 Demo API 服务器作为使用示例。

`AsyncLLMEngine` 的实现代码位于 [vllm/engine/async_llm_engine.py](../../vllm/engine/async_llm_engine.py)。

## 工作进程 (Worker)

工作进程是运行模型推理的独立进程。vLLM 遵循使用一个进程控制一个加速器设备（如 GPU）的通用做法。例如，如果我们使用大小为 2 的张量并行和大小为 2 的流水线并行，我们总共将拥有 4 个工作进程。工作进程通过它们的 `rank` 和 `local_rank` 进行标识。`rank` 用于全局调度，而 `local_rank` 主要用于分配加速器设备和访问本地资源（如文件系统和共享内存）。

## 模型运行器 (Model Runner)

每个工作进程都有一个模型运行器对象，负责加载和运行模型。许多模型执行逻辑都存在于此，例如准备输入张量和捕获 CUDAGraph。

## 模型 (Model)

每个模型运行器对象都有一个模型对象，即实际的 `torch.nn.Module` 实例。请参阅 [huggingface_integration](huggingface_integration.md) 了解各种配置如何影响最终得到的类。

## 类层级结构 (Class Hierarchy)

下图展示了 vLLM 的类层级结构：

![Class Hierarchy](../assets/design/hierarchy.png)

在此类层级结构背后，有几个重要的设计选择：

1. **可扩展性 (Extensibility)**：层级结构中的所有类都接受一个包含所有必要配置信息的配置对象。[VllmConfig](https://github.com/vllm-project/vllm/blob/d1c6799b8870e513bf4f2305cbf6cda9fc3d773b/vllm/config.py#L2036) 类是传递的主要配置对象。类层级结构非常深，每个类都需要读取它感兴趣的配置。通过将所有配置封装在一个对象中，我们可以轻松地传递配置对象并访问我们需要的配置。假设我们想要添加一个新功能（鉴于大语言模型推理领域发展如此之快，这通常是很常见的情况），且该功能只涉及模型运行器。我们将必须在 `VllmConfig` 类中添加一个新的配置选项。由于我们将整个配置对象传递了下去，因此只需在 `VllmConfig` 中添加配置项，模型运行器就能直接访问它。我们不需要更改引擎、工作进程或模型类的构造函数来传递该新配置选项。

2. **一致性 (Uniformity)**：模型运行器需要一个统一的接口来创建和初始化模型。vLLM 支持 50 多种流行的开源模型。每种模型都有其自己的初始化逻辑。如果构造函数签名随模型不同而变化，则模型运行器在没有复杂且易出错的检查逻辑的情况下，无法相应地调用构造函数。通过让模型类的构造函数保持统一，模型运行器可以轻松地创建和初始化模型，而无需了解具体的模型类型。这对于组合模型也很有用。视觉语言模型（VLM）通常由一个视觉模型和一个语言模型组成。通过使构造函数统一，我们可以轻松创建视觉模型和语言模型，并将它们组合成视觉语言模型。

!!! note "注意"
    为了支持这一设计变化，所有 vLLM 模型的构造函数签名已更新为：

    ```python
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
    ```

    为避免意外传递不正确的参数，构造函数现在是仅限关键字（keyword-only）的。这确保了如果传递了旧配置，构造函数将抛出错误。vLLM 开发人员已经完成了对 vLLM 内部所有模型的修改。对于树外（Out-of-tree）注册的模型，开发人员需要更新他们的模型，例如添加垫片代码（Shim code）以将旧的构造函数签名适配为新签名：

    ??? code

        ```python
        class MyOldModel(nn.Module):
            def __init__(
                self,
                config,
                cache_config: Optional[CacheConfig] = None,
                quant_config: Optional[QuantizationConfig] = None,
                lora_config: Optional[LoRAConfig] = None,
                prefix: str = "",
            ) -> None:
                ...

        from vllm.config import VllmConfig
        class MyNewModel(MyOldModel):
            def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
                config = vllm_config.model_config.hf_config
                cache_config = vllm_config.cache_config
                quant_config = vllm_config.quant_config
                lora_config = vllm_config.lora_config
                super().__init__(config, cache_config, quant_config, lora_config, prefix)

        from packaging import version
        if version.parse(__version__) >= version.parse("0.6.4"):
            MyModel = MyNewModel
        else:
            MyModel = MyOldModel
        ```

    这样，模型就能同时在旧版本和新版本的 vLLM 下正常工作。

3. **初始化时的分片与量化 (Sharding and Quantization at Initialization)**：某些特性需要更改模型权重。例如，张量并行需要对模型权重进行分片，而量化需要对模型权重进行量化。这有两种可能的实现方式。第一种方式是在模型初始化**后**更改模型权重；第二种方式是在模型初始化**期间**更改模型权重。vLLM 选择了后者。第一种方法无法扩展到超大模型。假设我们要在 16 张 H100 80GB GPU 上运行一个 405B 模型（具有大约 810GB 的权重）。理想情况下，每个 GPU 只需加载 50GB 权重。如果我们在模型初始化后更改权重，我们就需要将完整的 810GB 权重加载到每个 GPU 上然后再进行分片，这会导致巨大的内存开销。相反，如果我们在模型初始化期间对权重进行分片，每个层将仅创建其需要的那部分权重分片，从而带来小得多的内存开销。同样的想法也适用于量化。需要注意的是，我们还在模型的构造函数中添加了一个额外的参数 `prefix`，以便模型可以根据前缀不同进行不同的自我初始化。这对于非均匀量化非常有用，因为模型的不同部分可能会采用不同的量化方式。`prefix` 对于顶层模型通常是空字符串，而对于子模型则是类似 `"vision"` 或 `"language"` 的字符串。一般来说，它与 checkpoint 文件中模块 state dict 的名称相对应。

这种设计的一个缺点是很难为 vLLM 中的单个组件编写单元测试，因为每个组件都需要由一个完整的配置对象进行初始化。我们通过提供一个默认的初始化函数来解决这一问题，该函数会创建一个所有字段均设置为 `None` 的默认配置对象。如果我们想要测试的组件仅仅关心配置对象中的少数几个字段，我们可以创建一个默认配置对象并设置我们关心的字段。通过这种方式，我们可以在隔离状态下测试该组件。另外，vLLM 中的大多数测试都是测试整个系统的端到端测试，因此这不是一个严重的问题。

总之，完整的配置对象 `VllmConfig` 可以被视为在所有 vLLM 类之间共享的引擎级全局状态。
