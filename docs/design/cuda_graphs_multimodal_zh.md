# 视觉编码器 (ViT) CUDA 图 (Vision Encoder (ViT) CUDA Graphs)

vLLM 中的 [CUDA 图](cuda_graphs_zh.md) 基础设施主要针对**解码器（Language Model, LM）**的前向传播。此外，vLLM 还支持独立于解码器，将**编码器（Vision Transformer, ViT）**的前向传播捕获为 CUDA 图。该功能基于 <https://github.com/vllm-project/vllm/pull/35963>。

对于双塔式（Two-tower）视觉编码器（例如带有动态瓦片划分 dynamic tiling 的 DeepSeek-OCR 的 SAM + CLIP），**双路径图（Dual-path graph）**模式可以捕获两套独立的 CUDA 图 —— 一套用于全局图像路径，另一套用于局部图像块路径 —— 从而支持每条路径的独立预算（Budget）选择与部分 Eager 模式回退。这基于 <https://github.com/vllm-project/vllm/pull/43586>。

!!! note "注意"
    编码器 CUDA 图与解码器 CUDA 图是正交的 —— 两者可以同时启用。如 [CUDA 图设计文档](cuda_graphs_zh.md) 中所述，编码器图捕获视觉编码器的执行（例如 Qwen3-VL 中的 ViT），而解码器图捕获语言模型的执行。

## 动因 (Motivation)

视觉编码器推理会在主机端（Host side）引入 CUDA Kernel 启动开销。当 Batch Size 较小或图像尺寸较小时，这种开销尤为显著。

编码器 CUDA 图通过在模型初始化期间，以多个 Token 预算级别（Token budget levels）预先捕获完整的编码器前向传播，并在运行时重新播放（Replay）相应的图，来消除这种开销。

对于像 DeepSeek-OCR 这样的双塔式视觉编码器（SAM + CLIP 结合动态瓦片划分），其全局图像路径和局部图像块路径拥有独立的 Token Profile（每个全局图像 272 个 Token，而每个局部图像块 100 个 Token）。如果将两条路径捕获为一个单一的庞大图，会显著降低打包（Packing）效率。双路径图模式将每条路径捕获为一组独立的预算，允许管理器独立地打包和回放每条路径。

## 设计 (Design)

编码器 CUDA 图系统采用**基于预算的捕获/回放 (Budget-based capture/replay)** 策略，由 [EncoderCudaGraphManager][vllm.v1.worker.encoder_cudagraph.EncoderCudaGraphManager] 进行管理。该系统包含以下核心组件：

* [EncoderCudaGraphManager][vllm.v1.worker.encoder_cudagraph.EncoderCudaGraphManager]：协调编码器 CUDA 图的捕获、回放、贪婪打包以及数据并行执行。
* [SupportsEncoderCudaGraph][vllm.model_executor.models.interfaces.SupportsEncoderCudaGraph]：一个运行时可检查的协议（Protocol），模型实现该协议以选择启用编码器 CUDA 图。
* [EncoderItemSpec][vllm.v1.worker.encoder_cudagraph_defs.EncoderItemSpec]：描述单个编码器输入项（图像或视频）及其输入尺寸和输出 Token 数量。
* [BudgetGraphMetadata][vllm.v1.worker.encoder_cudagraph.BudgetGraphMetadata]：持有单个 Token 预算级别的已捕获 CUDA 图及其关联的 I/O 缓冲区。

### 基于预算的图捕获

我们会在不同的 **Token 预算** 级别（例如 `[2048, 4096, 8192, 13824]`）下预先捕获多个 CUDA 图。每个预算定义了固定的 Token 容量，且所有预算共享相同的最大 Batch Size（图像数量）。每个级别的 `BudgetGraphMetadata` 存储了图以及预分配的输入、元数据和输出缓冲区：

```python
@dataclass
class BudgetGraphMetadata:
    token_budget: int
    max_batch_size: int
    max_frames_per_batch: int
    graph: torch.cuda.CUDAGraph
    input_buffers: dict[str, torch.Tensor]  # 例如 pixel_values, embeddings, seq metadata
    output_buffer: torch.Tensor      # 编码器隐藏状态
```

预算是通过 `get_encoder_cudagraph_budget_range()` 从模型提供的范围内以 2 的幂次自动生成的，且最大预算总是被包含进去（即使它不处于 2 的幂次边界上）。用户也可以在 `CompilationConfig` 中通过 `encoder_cudagraph_token_budgets` 显式指定预算。

当 `EncoderCudaGraphConfig.enable_dual_path_graph` 为 `True` 时，管理器生成两个独立的预算列表 —— `global_token_budgets`（`global_token_per_image` 的倍数）和 `local_token_budgets`（`local_token_per_patch` 的倍数） —— 并分别将捕获的图存储在 `budget_graphs["global"]` 和 `budget_graphs["local"]` 下。

### 运行时的贪婪装箱 (Greedy bin-packing at runtime)

当一批图像到达时，管理器按输出 Token 数量对图像进行排序（从小到大），并贪婪地将尽可能多的图像打包到每个子批次中，同时保持在**最大**的 Token 预算和最大 Batch Size 限制之内。一旦一个子批次被确定（下一个图像会溢出任意一个约束），管理器会找到符合该子批次总 Token 数的**最小**预算，并回放相应的 CUDA 图。这一过程一直重复，直至整批图像处理完毕。超出所有预算的图像将回退到 Eager 模式执行。

对于双路径模型，管理器会路由到 `_execute_local_dual_path()`，它在打包期间同时约束全局和局部 Token 预算（参见 [双路径图捕获](#_2)）。

对于每次图回放：

1. 调用 `prepare_encoder_cudagraph_replay_buffers()`，从实际批次输入中计算出缓冲区的值（包括 `pixel_values` 和预先计算的元数据）。
2. 将预分配的 `input_buffers` 清零，然后将回放的值切片复制到其中。
3. 回放 CUDA 图。
4. 克隆 `output_buffer` 中的输出（因为该缓冲区会在不同的回放中被重复使用，克隆是必要的）。

### 双路径图捕获 (Dual-Path graph capture)

对于双塔式视觉编码器（例如 DeepSeek-OCR），`EncoderCudaGraphConfig` 将 `enable_dual_path_graph` 设置为 `True` 并提供 `global_token_per_image` / `local_token_per_patch`。管理器捕获两套独立的 CUDA 图 —— 一套用于**全局**图像路径，一套用于**局部**图像块路径 —— 分别保存在 `budget_graphs["global"]` 和 `budget_graphs["local"]` 下。

**预算生成**。会生成两个独立的预算列表：

* `global_token_budgets` — `global_token_per_image` 的 2 的幂次元数倍数（例如，对于 DeepSeek-OCR，为 `[272, 544, 1088, 2176, 4352, 8704, 13824]`）。
* `local_token_budgets` — `local_token_per_patch` 的 2 的幂次元数倍数（例如，对于 DeepSeek-OCR，为 `[0, 100, 200, 400, 800, 1600, 3200, 6400, 12800]`）。总是会包含 `0` 预算，以处理没有局部图像块的图像（即小于或等于 640×640 且只产生全局特征的图像）。

两个列表的上限均为同一个 `max_budget`。

**双路径贪婪打包**。每个 `EncoderItemSpec` 同时提供 `global_output_tokens`（每张图像为常数）和 `local_output_tokens`（与图像块的数量成比例）。双路径打包算法同时约束这两个预算：

* 按总输出 Token 数量（全局 + 局部）对图像进行排序，从小到大。
* 贪婪地打包图像：只有当累计的全局 Token 数 ≤ `max_global_budget` **且**累计的局部 Token 数 ≤ `max_local_budget` 且图像数量 ≤ `max_batch_size` 时，图像才会被加入当前的子批次中。
* 一旦任意一个约束会发生溢出，则确定当前子批次，并**独立地**为每条路径寻找最小的符合预算。
* 重复此过程，直到所有图像被打包。

**部分图回退 (Partial graph fallback)**。打包后，每个子批次会进入以下四种执行场景之一：

| 全局预算 (Global budget) | 局部预算 (Local budget) | 执行方式 |
| :---: | :---: | --- |
| 匹配成功 | 匹配成功 | 两条路径均使用 CUDA 图回放 |
| 匹配成功 | `None` | 全局路径使用图回放 + 局部路径被跳过（无图像块） |
| `None` | 匹配成功 | 全局路径回退到 Eager 执行 + 局部路径使用图回放 |
| `None` | `None` | 两条路径均回退到 Eager 模式执行 |

请注意，`0` 预算的图实际上在局部路径中绝不会被回放 —— 它表示局部图像块的处理应当被完全跳过。

**逐路径的缓冲区键**。全局和局部路径使用不同的缓冲区键。对于 DeepSeek-OCR，全局路径使用 `pixel_values`（完整图像，形状为 `[B, 3, 1280, 1280]`），而局部路径使用 `images_crop`（图像块，形状为 `[P, 3, 1024, 1024]`）。管理器遍历每个捕获图自身的 `input_buffers.keys()`，而不是共享的 `buffer_keys` 列表，因此两条路径可以使用不同的缓冲区。

**后处理**。`postprocess_encoder_output` 方法接收一个 `local_output` 参数（一个张量或 `None`），包含局部路径编码器的输出。模型负责将全局和局部特征组装为最终的单图像嵌入。对于 DeepSeek-OCR，这意味着将全局输出重塑为 `[B, 272, n_embed]`，局部输出重塑为 `[P, 100, n_embed]`，用换行 Token 组装图像块网格，并为每张图像拼接 `[patches_grid, global, view_separator]`。

!!! note "注意"
    双路径设计支持部分 CUDA 图覆盖 —— 一条路径可以命中，而另一条路径回退到 Eager。这避免了对无瓦片图像的零填充图像块缓冲区进行无谓的计算，并避免了由每张图像不同的 `crop_shape` 导致的图失效。

### 数据并行支持 (Data-parallel support)

当 `mm_encoder_tp_mode="data"` 时，管理器通过 `get_load_balance_assignment` 采用负载均衡分配，将图像分发到各个 TP Rank 上，在各个 Rank 上进行本地执行，然后通过 `tensor_model_parallel_all_gather` 按原始顺序收集结果。

### 视频推理支持 (Video inference support)

继实现图像推理的 ViT 完整 CUDA 图支持（<https://github.com/vllm-project/vllm/pull/35963>）之后，<https://github.com/vllm-project/vllm/pull/38061> 扩展了编码器 CUDA 图框架以支持 Qwen3-VL 的视频推理。在此之前，CUDA 图捕获/回放路径仅能处理图像输入（`pixel_values` + `image_grid_thw`）。视频输入使用不同的键（`pixel_values_videos` + `video_grid_thw`）且需要更大的 `cu_seqlens` 缓冲区，因为每个视频项会贡献多个帧（`T` 个注意力序列）。该 PR 泛化了协议和管理器，使它们能通过单个共享的图管理器处理这两种模态。

!!! note "注意"
    当启用了 EVS（高效视频采样，Efficient Video Sampling）剪枝时，视频 CUDA 图会自动被禁用，因为 EVS 使得 Token 数量与数据相关，与 CUDA 图捕获不兼容。

    目前也支持每个提示词混合输入（图像 + 视频）。

## 通过 `SupportsEncoderCudaGraph` 进行模型集成

模型通过实现 [SupportsEncoderCudaGraph][vllm.model_executor.models.interfaces.SupportsEncoderCudaGraph] 协议来启用编码器 CUDA 图。该协议封装了所有特定于模型的逻辑，使得管理器保持模型无关性。该协议定义了以下方法：

* `get_encoder_cudagraph_config()` — 返回静态配置（支持的模态、缓冲区键、输出隐藏维度、填充逻辑、每个视频的最大帧数）。
* `get_encoder_cudagraph_budget_range(vllm_config)` — 返回 `(min_budget, max_budget)` 以自动推断 Token 预算。
* `get_encoder_cudagraph_item_specs(mm_kwargs)` — 返回 `list[EncoderItemSpec]`，描述每个项的输入尺寸、总输出 Token 数（`output_tokens`），以及对于双路径模型可选的逐路径 Token 数（`global_output_tokens`，`local_output_tokens`）。
* `select_encoder_cudagraph_items(mm_kwargs, indices)` — 通过索引提取子批次的项，在贪婪打包和 DP 分片期间使用。
* `prepare_encoder_cudagraph_capture_inputs(..., path="default")` — 为图捕获创建虚拟输入。`path` 参数（`"global"` 或 `"local"`）指示模型为哪条路径生成虚拟输入。返回包含单个 `values: dict[str, torch.Tensor]` 字典的 `EncoderCudaGraphCaptureInputs`，该字典包含所有要记录到图中的缓冲区。
* `prepare_encoder_cudagraph_replay_buffers(mm_kwargs, max_batch_size, max_frames_per_batch, path="default")` — 从实际批次输入中计算缓冲区的值。`path` 参数选择从 `mm_kwargs` 中提取哪些模态键。返回包含一个 `values` 字典的 `EncoderCudaGraphReplayBuffers`，该字典的键与捕获图的 `input_buffers.keys()` 匹配。
* `encoder_cudagraph_forward(inputs: dict[str, torch.Tensor], path="default")` — 仅接受固定形状输入张量（已捕获的 `values` 字典）的前向传播。在捕获和回放期间均会被调用。`path` 参数会分发到正确的编码器子模块（例如，DeepSeek-OCR 的全局或局部路径）。
* `encoder_eager_forward(mm_kwargs, path="default")` — 当没有匹配的图时，回退到 Eager 模式的前向传播。当 `path` 是 `"global"` 或 `"local"` 时，仅运行对应的编码器路径而不进行图捕获。
* `postprocess_encoder_output(..., local_output=None)` — 对编码器的输出进行后处理。`local_output` 参数接收局部路径编码器的输出张量（或 `None`），使得双路径模型能够将全局和局部特征组装为最终的单图像嵌入。

!!! note "注意"
    `SupportsEncoderCudaGraph` 协议被设计为模型无关的。新的视觉编码器模型只需实现该协议的方法即可选择启用此特性，而无需修改管理器。

**受支持的模型列表：**

| 架构名 | 模型名 | 图像支持 CUDA 图 | 视频支持 CUDA 图 | 双路径图支持 |
| ------------ | ------ | ------------ | ------------ | --------------- |
| `DeepseekOCRForCausalLM` | `DeepSeek-OCR` | ✅︎ | ❌︎ | ✅︎ |
| `Gemma3ForConditionalGeneration` | `Gemma3` | ✅︎ | ❌︎ | ❌︎ |
| `Glm4vForConditionalGeneration` | `GLM-4.1V, GLM-4.6V-Flash` | ✅︎ | ✅︎ | ❌︎ |
| `InternVLChatModel` | `InternVL3.5`, `InternVL3`, `InternVL2.5`, `InternVL2` | ✅︎ | ✅︎ | ❌︎ |
| `KimiVLForConditionalGeneration` | `Kimi-VL` | ✅︎ | ❌︎ | ❌︎ |
| `Llama4ForConditionalGeneration` | `Llama 4` | ✅︎ | ❌︎ | ❌︎ |
| `Qwen2VLForConditionalGeneration` | `Qwen2-VL` | ✅︎ | ✅︎ | ❌︎ |
| `Qwen2_5_VLForConditionalGeneration` | `Qwen2.5-VL` | ✅︎ | ✅︎ | ❌︎ |
| `Qwen3VLForConditionalGeneration` | `Qwen3-VL` | ✅︎ | ✅︎ | ❌︎ |
| `Qwen3_5ForConditionalGeneration` | `Qwen3.5`, `Qwen3.6` | ✅︎ | ✅︎ | ❌︎ |
| `Qwen3_5MoeForConditionalGeneration` | `Qwen3.5-MoE`, `Qwen3.6-MoE` | ✅︎ | ✅︎ | ❌︎ |
| `Step3VLForConditionalGeneration` | `Step3-VL` | ✅︎ | ❌︎ | ✅︎ |

!!! note "注意"
    编码器 CUDA 图目前已在 Blackwell GPU (GB200) 上搭配 `--mm-encoder-attn-backend=FLASH_ATTN` 和 `--mm-encoder-attn-backend=FLASHINFER` 完成测试。
    对于 Qwen2-VL 和 Qwen2.5-VL，仅测试了 FA2 和 FA3。

## 相关配置 (Configuration)

`CompilationConfig` 中的三个字段控制编码器 CUDA 图：

* `cudagraph_mm_encoder`（`bool`，默认 `False`）— 启用多模态编码器的 CUDA 图捕获。启用后，会为每个 Token 预算级别将整个编码器的前向传播捕获为 CUDA 图。
* `encoder_cudagraph_token_budgets`（`list[int]`，默认 `[]`）— 用于捕获的 Token 预算级别。如果为空（默认），则从模型架构中自动推断为 2 的幂次级别。用户提供的值将覆盖自动推断。
* `encoder_cudagraph_max_vision_items_per_batch`（`int`，默认 `0`）— 捕获期间每批的最大图像/视频数量。如果为 0（默认），则自动推断为 `max_budget // min_budget`。
* `encoder_cudagraph_max_frames_per_batch`（`int`，默认 `None`）— 捕获期间每批的最大视频帧数。如果是 `None`（默认），则自动推断为 `encoder_cudagraph_max_vision_items_per_batch * max_frames_per_video`（`max_frames_per_video` 是一个来自 `EncoderCudaGraphConfig` 的模型特定值，由模型上的 `get_max_frames_per_video()` 计算得出）。如果我们限制每个提示词的视频数为 `0`，则它也将被设置为 `0`（即回退到仅图像模式）。

双路径模式在模型级别通过 `EncoderCudaGraphConfig` 的字段（`enable_dual_path_graph`、`global_token_per_image`、`local_token_per_patch`）进行配置 —— 无需额外的用户配置。当模型选择启用时，管理器会自动生成独立的预算列表并路由到双路径执行。

## 使用指南 (Usage guide)

### 图像推理

通过 `compilation_config` 启用编码器 CUDA 图：

```bash
vllm serve Qwen/Qwen3-VL-32B \
  --compilation-config '{"cudagraph_mm_encoder": true}'
```

对于 `Llama 4`（仅图像）：

```bash
vllm serve meta-llama/Llama-4-Scout-17B-16E-Instruct \
  --limit-mm-per-prompt '{"image": 1}' \
  --compilation-config '{"cudagraph_mm_encoder": true}'
```

使用显式指定的预算：

```bash
vllm serve Qwen/Qwen3-VL-32B \
  --compilation-config '{"cudagraph_mm_encoder": true, "encoder_cudagraph_token_budgets": [2048, 4096, 8192, 13824], "encoder_cudagraph_max_vision_items_per_batch": 8}'
```

Python 示例：

```python
import vllm

compilation_config = {
    "cudagraph_mm_encoder": True,
    # 可选：覆盖自动推断的预算
    # "encoder_cudagraph_token_budgets": [2048, 4096, 8192, 13824],
    # "encoder_cudagraph_max_vision_items_per_batch": 8,
}

model = vllm.LLM(
    model="Qwen/Qwen3-VL-32B",
    compilation_config=compilation_config,
)
```

管理器会追踪命中/未命中统计数据并定期输出日志。“命中（hit）”意味着图像通过 CUDA 图回放处理；“未命中（miss）”意味着回退到 Eager 模式（图像超出了所有预算）。

### 视频推理

通过 `compilation_config` 启用编码器 CUDA 图：

```bash
vllm serve Qwen/Qwen3-VL-32B \
  --compilation-config '{"cudagraph_mm_encoder": true}'
```

使用显式指定的预算：

```bash
vllm serve Qwen/Qwen3-VL-32B \
  --compilation-config '{"cudagraph_mm_encoder": true, "encoder_cudagraph_token_budgets": [2048, 4096, 8192, 13824], "encoder_cudagraph_max_vision_items_per_batch": 8, "encoder_cudagraph_max_frames_per_batch": 64}'
```

Python 示例：

```python
import vllm

compilation_config = {
    "cudagraph_mm_encoder": True,
    # 可选：覆盖自动推断的预算
    # "encoder_cudagraph_token_budgets": [2048, 4096, 8192, 13824],
    # "encoder_cudagraph_max_vision_items_per_batch": 8,
    # "encoder_cudagraph_max_frames_per_batch": 64,
}

model = vllm.LLM(
    model="Qwen/Qwen3-VL-32B",
    compilation_config=compilation_config,
)
```

## 关于性能表现

以下基准测试是在 Blackwell GPU (GB200) 上使用 `vllm bench mm-processor` 运行的。完整细节请参见 [#35963](https://github.com/vllm-project/vllm/pull/35963)。

### 单 GPU 性能 (1x GB200)

模型：`Qwen/Qwen3-VL-30B-A3B-Instruct`，数据集：`lmarena-ai/VisionArena-Chat`（3000 个提示词，300 个热身），`max_model_len=32768`。

| 后端 | 平均延迟改善 | P99 延迟改善 |
| :------ | :----------------------- | :---------------------- |
| FLASH_ATTN | +11.8% (5.13→4.52ms) | +31.6% (9.16→6.26ms) |
| FLASHINFER | +19.6% (5.42→4.36ms) | +40.3% (10.87→6.49ms) |

重现命令：

```bash
vllm bench mm-processor \
  --model Qwen/Qwen3-VL-30B-A3B-Instruct \
  --dataset-name hf --dataset-path lmarena-ai/VisionArena-Chat \
  --num-prompts 3000 --num-warmups 300 \
  --max-model-len 32768 --seed 42 \
  --mm-encoder-attn-backend FLASH_ATTN \
  --compilation-config '{"cudagraph_mm_encoder": true, "encoder_cudagraph_token_budgets": [512, 1024, 1536, 2048, 2560, 3072, 3584, 4096, 4864], "encoder_cudagraph_max_vision_items_per_batch": 8}'
```

### 多 GPU 性能 (4x GB200, TP=4, DP=4)

模型：`Qwen/Qwen3-VL-32B-Instruct`，数据集：`random-mm`（1000 个提示词，200 个热身，每个请求包含 20 张 336x336 的图像），`max_model_len=8192`。

| 后端 | 平均延迟改善 | P99 延迟改善 |
| :------ | :----------------------- | :---------------------- |
| FLASH_ATTN | +18.4% (28.39→23.16ms) | +14.0% (238.78→205.28ms) |
| FLASHINFER | +44.4% (23.24→12.91ms) | +84.9% (172.41→26.05ms) |

重现命令：

```bash
vllm bench mm-processor \
  --model Qwen/Qwen3-VL-32B-Instruct \
  --dataset-name random-mm \
  --random-mm-base-items-per-request 20 \
  --random-mm-num-mm-items-range-ratio 0.0 \
  --random-mm-bucket-config '{"(336,336,1)": 1.0}' \
  --num-prompts 1000 --num-warmups 200 \
  --max-model-len 8192 --seed 42 \
  --mm-encoder-attn-backend FLASHINFER \
  --tensor-parallel-size 4 --mm-encoder-tp-mode data \
  --compilation-config '{"cudagraph_mm_encoder": true, "encoder_cudagraph_token_budgets": [512, 1024, 1536, 2048, 2560, 3072, 3584, 4096, 4864], "encoder_cudagraph_max_vision_items_per_batch": 8}'
```

!!! note "注意"
    有关 GPU (A100) 上视频推理基准测试的更多细节，请参见 [#38061](https://github.com/vllm-project/vllm/pull/38061)。
