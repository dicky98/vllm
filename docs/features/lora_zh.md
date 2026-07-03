# LoRA 适配器 (LoRA Adapters)

本文档向您展示如何在 vLLM 中在基座模型之上使用 [LoRA 适配器](https://arxiv.org/abs/2106.09685)。

LoRA 适配器可以与任何实现了 [SupportsLoRA][vllm.model_executor.models.interfaces.SupportsLoRA] 的 vLLM 模型一起使用。

适配器可以在每次请求时高效地进行服务，且额外开销极小。首先，我们下载适配器并使用以下代码将其保存在本地：

```python
from huggingface_hub import snapshot_download

sql_lora_path = snapshot_download(repo_id="jeeejeee/llama32-3b-text2sql-spider")
```

然后，我们实例化基座模型并传入 `enable_lora=True` 标志：

```python
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

llm = LLM(model="meta-llama/Llama-3.2-3B-Instruct", enable_lora=True)
```

现在，我们可以提交提示词并通过 `lora_request` 参数调用 `llm.generate`。`LoRARequest` 的第一个参数是一个易于识别的人类可读名称，第二个参数是适配器的全局唯一 ID，第三个参数是 LoRA 适配器的本地路径。

??? code

    ```python
    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=256,
        stop=["[/assistant]"],
    )

    prompts = [
        "[user] Write a SQL query to answer the question based on the table schema.\n\n context: CREATE TABLE table_name_74 (icao VARCHAR, airport VARCHAR)\n\n question: Name the ICAO for lilongwe international airport [/user] [assistant]",
        "[user] Write a SQL query to answer the question based on the table schema.\n\n context: CREATE TABLE table_name_11 (nationality VARCHAR, elector VARCHAR)\n\n question: When Anchero Pantaleone was the elector what is under nationality? [/user] [assistant]",
    ]

    outputs = llm.generate(
        prompts,
        sampling_params,
        lora_request=LoRARequest("sql_adapter", 1, sql_lora_path),
    )
    ```

查看 [examples/features/lora/multilora_offline.py](../../examples/features/lora/multilora_offline.py) 以获取如何与异步引擎一起使用 LoRA 适配器以及如何使用更高级配置选项的示例。

## 服务 LoRA 适配器 (Serving LoRA Adapters)

LoRA 适配模型也可以通过兼容 OpenAI 的 vLLM 服务器进行服务。为此，我们在启动服务器时使用 `--lora-modules {name}={path} {name}={path}` 来指定每个 LoRA 模块：

```bash
vllm serve meta-llama/Llama-3.2-3B-Instruct \
    --enable-lora \
    --lora-modules sql-lora=jeeejeee/llama32-3b-text2sql-spider
```

服务器入口点接受所有其他 LoRA 配置参数（如 `max_loras`、`max_lora_rank`、`max_cpu_loras` 等），这些参数将应用于所有后续请求。在查询 `/models` 端点时，我们应该能看到我们的 LoRA 以及它的基座模型（如果未安装 `jq`，您可以参考 [此指南](https://jqlang.org/download/) 进行安装）：

??? console "命令示例"

    ```bash
    curl localhost:8000/v1/models | jq .
    {
        "object": "list",
        "data": [
            {
                "id": "meta-llama/Llama-3.2-3B-Instruct",
                "object": "model",
                ...
            },
            {
                "id": "sql-lora",
                "object": "model",
                ...
            }
        ]
    }
    ```

请求可以通过 `model` 请求参数指定 LoRA 适配器，就像指定任何其他模型一样。请求将根据服务器范围的 LoRA 配置进行处理（即与基座模型请求并行处理，如果 `max_loras` 设置得足够高，还可能与其他 LoRA 适配器请求并行处理）。

以下是一个请求示例：

```bash
curl http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "sql-lora",
        "prompt": "San Francisco is a",
        "max_tokens": 7,
        "temperature": 0
    }' | jq
```

## 动态服务 LoRA 适配器 (Dynamically serving LoRA Adapters)

除了在服务器启动时提供 LoRA 适配器外，vLLM 服务器还支持在运行时通过专用 API 端点和插件动态配置 LoRA 适配器。当需要按需切换模型时，此功能非常有用。

!!! warning "警告"
    此功能伴随有安全风险。除非处于隔离且完全可信的环境中，否则不应在生产环境中使用。

要启用动态 LoRA 配置，请确保将环境变量 `VLLM_ALLOW_RUNTIME_LORA_UPDATING` 设置为 `True`。

```bash
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=True
```

### 使用 API 端点

加载 LoRA 适配器：

要动态加载 LoRA 适配器，请向 `/v1/load_lora_adapter` 端点发送 POST 请求，并附带要加载的适配器的必要详细信息。请求的 payload 应包含 LoRA 适配器的名称和路径。

动态加载 LoRA 适配器的示例请求：

```bash
curl -X POST http://localhost:8000/v1/load_lora_adapter \
-H "Content-Type: application/json" \
-d '{
    "lora_name": "sql_adapter",
    "lora_path": "/path/to/sql-lora-adapter"
}'
```

成功发送请求后，API 将从 `vllm serve` 返回 `200 OK` 状态码，`curl` 将返回响应体：`Success: LoRA adapter 'sql_adapter' added successfully`。如果发生错误（例如找不到或无法加载该适配器），将返回相应的错误信息。

卸载 LoRA 适配器：

要卸载先前加载的 LoRA 适配器，请向 `/v1/unload_lora_adapter` 端点发送包含要卸载的适配器名称或 ID 的 POST 请求。

成功发送请求后，API 会从 `vllm serve` 返回 `200 OK` 状态码，`curl` 将返回响应体：`Success: LoRA adapter 'sql_adapter' removed successfully`。

卸载 LoRA 适配器的示例请求：

```bash
curl -X POST http://localhost:8000/v1/unload_lora_adapter \
-H "Content-Type: application/json" \
-d '{
    "lora_name": "sql_adapter"
}'
```

### 使用插件 (Using Plugins)

或者，您也可以使用 `LoRAResolver` 插件来动态加载 LoRA 适配器。`LoRAResolver` 插件使您能够从本地和远程源（例如本地文件系统和 S3）加载 LoRA 适配器。在接收到每个请求时，如果包含了一个尚未加载的新模型名称，`LoRAResolver` 将尝试解析并加载相应的 LoRA 适配器。

如果您想从不同源加载 LoRA 适配器，可以设置多个 `LoRAResolver` 插件。例如，您可以配置一个用于本地文件的解析器，以及另一个用于 S3 存储的解析器。vLLM 将加载它找到的第一个 LoRA 适配器。

您可以安装现有插件，也可以实现自己的插件。默认情况下，vLLM 附带了一个[从本地目录加载 LoRA 适配器的解析器插件，以及一个从 Hugging Face Hub 上的存储库加载 LoRA 适配器的解析器插件](https://github.com/vllm-project/vllm/tree/main/vllm/plugins/lora_resolvers)。要启用其中任一解析器，您必须将 `VLLM_ALLOW_RUNTIME_LORA_UPDATING` 设置为 `True`。

- 要利用本地目录，请设置 `VLLM_PLUGINS` 以包含 `lora_filesystem_resolver`，并将 `VLLM_LORA_RESOLVER_CACHE_DIR` 设置为本地目录。当 vLLM 收到一个使用 LoRA 适配器 `foobar` 的请求时，它会首先在本地目录中查找名为 `foobar` 的目录，并尝试将其中的内容作为 LoRA 适配器加载。如果成功，请求将正常完成，并且该适配器将随即可供服务器的常规请求使用。
- 要利用 Hugging Face Hub 上的仓库，请设置 `VLLM_PLUGINS` 以包含 `lora_hf_hub_resolver`，并将 `VLLM_LORA_RESOLVER_HF_REPO_LIST` 设置为 Hugging Face Hub 上以逗号分隔的仓库 ID 列表。当 vLLM 收到对 LoRA 适配器 `my/repo/subpath` 的请求时，如果它存在且包含 `adapter_config.json`，它将下载 `my/repo` 中 `subpath` 处的适配器，然后为该缓存目录构建一个适配器请求，类似于 `lora_filesystem_resolver` 的工作方式。请注意，启用远程下载具有安全风险，且不适用于生产环境。

或者，您可以参考以下示例步骤来实现自己的插件：

1. 实现 `LoRAResolver` 接口。

    ??? code "一个简单的 S3 LoRAResolver 实现示例"

        ```python
        import os
        import s3fs
        from vllm.lora.request import LoRARequest
        from vllm.lora.resolver import LoRAResolver

        class S3LoRAResolver(LoRAResolver):
            def __init__(self):
                self.s3 = s3fs.S3FileSystem()
                self.s3_path_format = os.getenv("S3_PATH_TEMPLATE")
                self.local_path_format = os.getenv("LOCAL_PATH_TEMPLATE")

            async def resolve_lora(self, base_model_name, lora_name):
                s3_path = self.s3_path_format.format(base_model_name=base_model_name, lora_name=lora_name)
                local_path = self.local_path_format.format(base_model_name=base_model_name, lora_name=lora_name)

                # 将 LoRA 从 S3 下载到本地路径
                await self.s3._get(
                    s3_path, local_path, recursive=True, maxdepth=1
                )

                lora_request = LoRARequest(
                    lora_name=lora_name,
                    lora_path=local_path,
                    lora_int_id=abs(hash(lora_name)),
                )
                return lora_request
        ```

2. 注册 `LoRAResolver` 插件。

    ```python
    from vllm.lora.resolver import LoRAResolverRegistry

    s3_resolver = S3LoRAResolver()
    LoRAResolverRegistry.register_resolver("s3_resolver", s3_resolver)
    ```

    有关更多详细信息，请参阅 [vLLM 插件系统](../design/plugin_system.md)。

### 原地 LoRA 重新加载 (In-Place LoRA Reloading)

动态加载 LoRA 适配器时，您可能需要用更新后的权重替换现有适配器，同时保持名称相同。`load_inplace` 参数启用了此功能。这在异步强化学习设置中非常常见，其中适配器会不断更新和替换，而不会中断正在进行的推理。

当 `load_inplace=True` 时，vLLM 将用新适配器替换具有相同名称的现有适配器。

加载或替换同名 LoRA 适配器的示例请求：

```bash
curl -X POST http://localhost:8000/v1/load_lora_adapter \
-H "Content-Type: application/json" \
-d '{
    "lora_name": "my-adapter",
    "lora_path": "/path/to/adapter/v2",
    "load_inplace": true
}'
```

## `--lora-modules` 的新格式

在以前的版本中，用户会通过以下格式（键值对或 JSON 格式）提供 LoRA 模块。例如：

```bash
--lora-modules  sql-lora=jeeejeee/llama32-3b-text2sql-spider
```

这仅包含每个 LoRA 模块的 `name` 和 `path`，但无法指定 `base_model_name`。
现在，您可以使用 JSON 格式在指定名称和路径的同时指定 `base_model_name`。例如：

```bash
--lora-modules '{"name": "sql-lora", "path": "jeeejeee/llama32-3b-text2sql-spider", "base_model_name": "meta-llama/Llama-3.2-3B-Instruct"}'
```

为了提供向下兼容支持，您仍然可以使用旧的键值格式（name=path），但在这种情况下 `base_model_name` 将保持未指定状态。

## 混合使用 2D 和 3D MoE LoRA 适配器

要在同一个引擎实例中同时服务 2D 格式（基于 `megatron`）和 3D 格式（基于 `peft`）的适配器，请使用 `--enable-mixed-moe-lora-format` 启动服务器，并通过 `is_3d_lora_weight` 字段显式声明每个适配器的结构布局。

服务器启动（静态模块）：

```bash
vllm serve Qwen/Qwen3.6-35B-A3B \
    --enable-lora \
    --enable-mixed-moe-lora-format \
    --tensor-parallel-size 4 \
    --enable-expert-parallel \
    --lora-modules \
        '{"name": "lora-2d", "path": "jeeejeee/qwen36-35ba3b-2d-weights-poken-lora", "is_3d_lora_weight": false}' \
        '{"name": "lora-3d", "path": "jeeejeee/qwen36-35ba3b-moe-all-linear-poken-lora", "is_3d_lora_weight": true}'
```

通过 `/v1/load_lora_adapter` 动态加载：

```bash
curl -X POST http://localhost:8000/v1/load_lora_adapter \
-H "Content-Type: application/json" \
-d '{
    "lora_name": "lora-3d",
    "lora_path": "/path/to/3d-format-lora",
    "is_3d_lora_weight": true
}'
```

!!! warning "警告：您必须了解您的适配器的结构布局"
    在 `--enable-mixed-moe-lora-format` 下，vLLM 会信任调用者声明的任何 `is_3d_lora_weight` 值 —— 它**不会**检查 checkpoint 以进行验证。错误的声明将把权重加载到错误的堆叠缓冲区中，并在加载时没有任何报错的情况下默默产生垃圾输出。在服务前请确认布局：

    - **2D（每个专家独立，megatron 风格）** → 设置 `is_3d_lora_weight: false`。
      适配器的键（keys）类似于 `...experts.{idx}.gate_proj.lora_A.weight`、`...experts.{idx}.up_proj.lora_A.weight`、`...experts.{idx}.down_proj.lora_A.weight` —— 每个专家有一套对应的键。
    - **3D（融合，peft 风格）** → 设置 `is_3d_lora_weight: true`。
      适配器的键（keys）类似于 `...experts.gate_up_proj.lora_A.weight`、`...experts.down_proj.lora_A.weight` —— 这是一个在首个维度（leading dimension）堆叠了所有专家的单一张量。

当**未**设置 `--enable-mixed-moe-lora-format` 时，`is_3d_lora_weight` 会被忽略：vLLM 将从基座模型的 `is_3d_moe_weight` 选择包装器，且适配器必须与其匹配。对于非 MoE 模型，该字段也会被忽略。

## 模型卡片中的 LoRA 模型谱系 (LoRA model lineage in model card)

新格式的 `--lora-modules` 主要是为了支持在模型卡片中显示父模型信息。以下是说明您的当前响应如何支持此功能：

- LoRA 模型 `sql-lora` 的 `parent` 字段现在链接到其基座模型 `meta-llama/Llama-3.2-3B-Instruct`。这正确反映了基座模型与 LoRA 适配器之间的层级关系。
- `root` 字段指向 LoRA 适配器的文件位置。

??? console "命令输出示例"

    ```bash
    $ curl http://localhost:8000/v1/models

    {
        "object": "list",
        "data": [
            {
            "id": "meta-llama/Llama-3.2-3B-Instruct",
            "object": "model",
            "created": 1715644056,
            "owned_by": "vllm",
            "root": "meta-llama/Llama-3.2-3B-Instruct",
            "parent": null,
            "permission": [
                {
                .....
                }
            ]
            },
            {
            "id": "sql-lora",
            "object": "model",
            "created": 1715644056,
            "owned_by": "vllm",
            "root": "jeeejeee/llama32-3b-text2sql-spider",
            "parent": "meta-llama/Llama-3.2-3B-Instruct",
            "permission": [
                {
                ....
                }
            ]
            }
        ]
    }
    ```

## 多模态模型中的 Tower 和 Connector 的 LoRA 支持

目前，vLLM 实验性地支持多模态模型的 Tower 和 Connector 组件的 LoRA。要启用此功能，您需要为 Tower 和 Connector 实现对应的 Token 辅助函数。有关此方法背后原理的更多详细信息，请参阅 [PR 26674](https://github.com/vllm-project/vllm/pull/26674)。我们欢迎贡献者将 LoRA 支持扩展到其他模型的 Tower 和 Connector 上。请参阅 [Issue 31479](https://github.com/vllm-project/vllm/issues/31479) 以检查当前的模型支持状态。

## 多模态模型的默认 LoRA 模型

某些模型，例如 [Granite Speech](https://huggingface.co/ibm-granite/granite-speech-3.3-8b) 和 [Phi-4-multimodal-instruct](https://huggingface.co/microsoft/Phi-4-multimodal-instruct) 多模态模型，包含一些在给定模态存在时期望始终应用的 LoRA 适配器。若使用上述方法来管理可能会有些繁琐，因为这要求用户发送 `LoRARequest`（离线），或者在基座模型和 LoRA 模型之间根据请求的多模态数据内容进行请求过滤（服务器）。

为此，我们允许注册默认的多模态 LoRA 来自动处理此过程，用户可以将每种模态映射到一个 LoRA 适配器，以便在相应的输入存在时自动应用它。请注意，目前我们每个提示词仅允许使用一个 LoRA；如果提供了注册到不同模态的多种模态输入，它们都将不会被应用。

??? code "离线推理的示例用法"

    ```python
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.assets.audio import AudioAsset

    model_id = "ibm-granite/granite-speech-3.3-2b"
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    def get_prompt(question: str, has_audio: bool):
        """构建要发送给 vLLM 的输入提示词。"""
        if has_audio:
            question = f"<|audio|>{question}"
        chat = [
            {"role": "user", "content": question},
        ]
        return tokenizer.apply_chat_template(chat, tokenize=False)


    llm = LLM(
        model=model_id,
        enable_lora=True,
        max_lora_rank=64,
        max_model_len=2048,
        limit_mm_per_prompt={"audio": 1},
        # 只要请求数据中包含音频，就会自动传入一个以 `model_id` 为标识的 `LoRARequest`。
        default_mm_loras = {"audio": model_id},
        enforce_eager=True,
    )

    question = "can you transcribe the speech into a written format?"
    prompt_with_audio = get_prompt(
        question=question,
        has_audio=True,
    )
    audio = AudioAsset("mary_had_lamb").audio_and_sample_rate

    inputs = {
        "prompt": prompt_with_audio,
        "multi_modal_data": {
            "audio": audio,
        }
    }


    outputs = llm.generate(
        inputs,
        sampling_params=SamplingParams(
            temperature=0.2,
            max_tokens=64,
        ),
    )
    ```

您还可以传递一个 `--default-mm-loras` 的 JSON 字典，将各模态映射到对应的 LoRA 模型 ID。例如，在启动服务器时：

```bash
vllm serve ibm-granite/granite-speech-3.3-2b \
    --max-model-len 2048 \
    --enable-lora \
    --default-mm-loras '{"audio":"ibm-granite/granite-speech-3.3-2b"}' \
    --max-lora-rank 64
```

注意：默认的多模态 LoRA 当前仅适用于 `.generate` 和聊天补全接口（chat completions）。

## 使用建议 (Using Tips)

### 配置 `max_lora_rank`

`--max-lora-rank` 参数控制允许的 LoRA 适配器的最大 rank。此设置会影响内存分配和性能：

- **将其设置为您计划使用的所有 LoRA 适配器中的最大 rank**。
- **避免将其设置得过高** —— 使用远大于所需的值会浪费内存，并可能导致性能问题。

例如，如果您的 LoRA 适配器的 rank 分别为 [16, 32, 64]，请使用 `--max-lora-rank 64` 而不是 256：

```bash
# 推荐：与实际的最大 rank 匹配
vllm serve model --enable-lora --max-lora-rank 64

# 不推荐：不必要地过高，浪费内存
vllm serve model --enable-lora --max-lora-rank 256
```

### 限制 LoRA 仅作用于特定模块

`--lora-target-modules` 参数允许您在部署时限制哪些模型模块应用 LoRA。这在您仅需要对特定层进行性能微调时非常有用：

```bash
# 仅对输出投影层（output projection layers）应用 LoRA
vllm serve model --enable-lora --lora-target-modules o_proj

# 对多个特定模块应用 LoRA
vllm serve model --enable-lora --lora-target-modules o_proj qkv_proj down_proj
```

未指定 `--lora-target-modules` 时，LoRA 将默认应用于模型中所有支持的模块。此参数接受模块名称的后缀（即模块名称的最后一部分），例如 `o_proj`、`qkv_proj`、`gate_proj` 等。
