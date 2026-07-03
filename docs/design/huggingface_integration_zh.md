# 与 Hugging Face 的集成 (Integration with Hugging Face)

本文档描述了 vLLM 如何与 Hugging Face 库进行集成。我们将逐步解释运行 `vllm serve` 时后台发生的事情。

假设我们想通过运行 `vllm serve Qwen/Qwen2-7B` 来服务热门的 Qwen 模型：

1. `model` 参数为 `Qwen/Qwen2-7B`。vLLM 通过检查是否存在相应的配置文件 `config.json` 来确定该模型是否存在。具体实现请参见此[代码片段](https://github.com/vllm-project/vllm/blob/10b67d865d92e376956345becafc249d4c3c0ab7/vllm/transformers_utils/config.py#L162-L182)。在此过程中：
    - 如果 `model` 参数对应一个已存在的本地路径，vLLM 将直接从该路径加载配置文件。
    - 如果 `model` 参数是一个由用户名和模型名组成的 Hugging Face 模型 ID，vLLM 将首先尝试使用本地 Hugging Face 缓存中的配置文件，其中以 `model` 参数作为模型名，`--revision` 参数作为修订版本。更多关于 Hugging Face 缓存工作机制的信息，请参阅[其官方网站](https://huggingface.co/docs/huggingface_hub/en/package_reference/environment_variables#hfhome)。
    - 如果 `model` 参数是 Hugging Face 模型 ID，但在本地缓存中找不到，vLLM 将从 Hugging Face 模型中心（Model Hub）下载该配置文件。具体实现参见[此函数](https://github.com/vllm-project/vllm/blob/10b67d865d92e376956345becafc249d4c3c0ab7/vllm/transformers_utils/config.py#L91)。输入参数包括作为模型名称的 `model` 参数、作为修订版本的 `--revision` 参数，以及作为访问模型中心 Token 的环境变量 `HF_TOKEN`。在我们的示例中，vLLM 将下载 [config.json](https://huggingface.co/Qwen/Qwen2-7B/blob/main/config.json) 文件。

2. 在确认模型存在后，vLLM 会加载其配置文件并将其转换为字典。具体实现参见此[代码片段](https://github.com/vllm-project/vllm/blob/10b67d865d92e376956345becafc249d4c3c0ab7/vllm/transformers_utils/config.py#L185-L186)。

3. 接下来，vLLM [检查](https://github.com/vllm-project/vllm/blob/10b67d865d92e376956345becafc249d4c3c0ab7/vllm/transformers_utils/config.py#L189)配置字典中的 `model_type` 字段，以[生成](https://github.com/vllm-project/vllm/blob/10b67d865d92e376956345becafc249d4c3c0ab7/vllm/transformers_utils/config.py#L190-L216)要使用的配置对象。有一些 `model_type` 值是 vLLM 直接支持的，支持列表参见[这里](https://github.com/vllm-project/vllm/blob/10b67d865d92e376956345becafc249d4c3c0ab7/vllm/transformers_utils/config.py#L48)。如果 `model_type` 不在列表中，vLLM 将使用 [AutoConfig.from_pretrained](https://huggingface.co/docs/transformers/en/model_doc/auto#transformers.AutoConfig.from_pretrained) 加载配置类，并将 `model`、`--revision` 和 `--trust_remote_code` 作为参数。请注意：
    - Hugging Face 也有自己的逻辑来确定要使用的配置类。它会再次使用 `model_type` 字段在 transformers 库中搜索类名，支持的模型列表参见[这里](https://github.com/huggingface/transformers/tree/main/src/transformers/models)。如果找不到 `model_type`，Hugging Face 将使用配置文件 JSON 中的 `auto_map` 字段来确定类名。具体来说，是 `auto_map` 下的 `AutoConfig` 字段。示例可以参考 [DeepSeek](https://huggingface.co/deepseek-ai/DeepSeek-V2.5/blob/main/config.json)。
    - `auto_map` 下的 `AutoConfig` 字段指向模型仓库中的一个模块路径。为了创建配置类，Hugging Face 会导入该模块并使用 `from_pretrained` 方法加载配置类。这通常可能导致任意代码执行，因此它仅在启用了 `--trust_remote_code` 时执行。

4. 随后，vLLM 会对配置对象应用一些历史补丁。这些补丁大多与 RoPE 配置相关，具体实现参见[这里](https://github.com/vllm-project/vllm/blob/127c07480ecea15e4c2990820c457807ff78a057/vllm/transformers_utils/config.py#L244)。

5. 最后，vLLM 找到了我们需要初始化的模型类。vLLM 使用配置对象中的 `architectures` 字段来确定要初始化的模型类，因为它在[其注册表](https://github.com/vllm-project/vllm/blob/127c07480ecea15e4c2990820c457807ff78a057/vllm/model_executor/models/registry.py#L80)中维护着从架构名称到模型类的映射。如果在注册表中找不到该架构名称，则意味着 vLLM 不支持此模型架构。对于 `Qwen/Qwen2-7B`，`architectures` 字段是 `["Qwen2ForCausalLM"]`，它对应 [vLLM 代码](https://github.com/vllm-project/vllm/blob/127c07480ecea15e4c2990820c457807ff78a057/vllm/model_executor/models/qwen2.py#L364)中的 `Qwen2ForCausalLM` 类。该类将根据各种配置进行自我初始化。

除此之外，还有另外两件事 vLLM 依赖于 Hugging Face：

1. **分词器 (Tokenizer)**：vLLM 使用 Hugging Face 的分词器对输入文本进行分词（Tokenize）。分词器使用 [AutoTokenizer.from_pretrained](https://huggingface.co/docs/transformers/en/model_doc/auto#transformers.AutoTokenizer.from_pretrained) 加载，其中以 `model` 参数作为模型名，`--revision` 参数作为修订版本。也可以通过在 `vllm serve` 命令中指定 `--tokenizer` 参数来使用另一个模型的分词器。其他相关参数包括 `--tokenizer-revision` 和 `--tokenizer-mode`。设置 `VLLM_USE_FASTOKENS=1` 会把 vLLM 加载的任何 HF 快速分词器（Fast Tokenizer）替换为基于 Rust 的开箱即用 BPE 后端（参见 [fastokens 后端](../configuration/optimization.md#fastokens-backend)）。请查看 Hugging Face 的文档以了解这些参数的具体含义。这部分逻辑可以在 [get_tokenizer](https://github.com/vllm-project/vllm/blob/127c07480ecea15e4c2990820c457807ff78a057/vllm/transformers_utils/tokenizer.py#L87) 函数中找到。在获取分词器后，值得注意的是，vLLM 会将分词器的一些高开销属性缓存到 [vllm.tokenizers.hf.get_cached_tokenizer][] 中。

2. **模型权重 (Model weight)**：vLLM 从 Hugging Face 模型中心下载模型权重，其中以 `model` 参数作为模型名，`--revision` 参数作为修订版本。vLLM 提供了 `--load-format` 参数来控制要从模型中心下载哪些文件。默认情况下，它将尝试加载 safetensors 格式的权重，如果 safetensors 格式不可用，则回退到 PyTorch 的 bin 格式。我们也可以传递 `--load-format dummy` 来跳过权重的下载。
    - 建议使用 safetensors 格式，因为它在分布式推理中加载效率高，而且可以免受任意代码执行的安全风险。有关 safetensors 格式的更多信息，请参阅[官方文档](https://huggingface.co/docs/safetensors/en/index)。这部分逻辑可以在[这里](https://github.com/vllm-project/vllm/blob/10b67d865d92e376956345becafc249d4c3c0ab7/vllm/model_executor/model_loader/loader.py#L385)找到。

至此，vLLM 与 Hugging Face 的集成介绍就完成了。

总结来说，vLLM 从 Hugging Face 模型中心或本地目录读取配置文件 `config.json`、分词器和模型权重。它使用来自 vLLM、Hugging Face transformers 的配置类，或者从模型仓库加载配置类。
