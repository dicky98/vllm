# 多模态数据处理 (Multi-Modal Data Processing)

为了在 vLLM 中启用诸如[分块预填充（Chunked prefill）](../configuration/optimization.md#chunked-prefill)和[前缀缓存（Prefix caching）](../features/automatic_prefix_caching_zh.md)之类的各种优化，我们使用 [BaseMultiModalProcessor][vllm.multimodal.processing.BaseMultiModalProcessor] 根据 Hugging Face 处理器（HF processor）的输出，来提供占位符特征 Token（如 `<image>`）与多模态输入（如原始输入图像）之间的对应关系。

以下是 [BaseMultiModalProcessor][vllm.multimodal.processing.BaseMultiModalProcessor] 的主要特性：

## 提示词更新检测 (Prompt Update Detection)

HF 处理器的主要职责之一是使用占位符 Token 更新提示词（Prompt）。例如：

- 在字符串的开头插入特征占位符 Token（例如 `<image><image>...<image>`，其数量等于特征大小）。
- 将现有的输入占位符 Token（例如针对单张图像的 `<image>`）替换为特征占位符 Token（例如 `<image><image>...<image>`，其数量等于特征大小）。

有关哪些 Token 已被更新的信息，对于寻找占位符特征 Token 和多模态输入之间的对应关系至关重要。

在 vLLM 中，此信息是在 [_get_prompt_updates][vllm.multimodal.processing.BaseMultiModalProcessor._get_prompt_updates] 中使用 [PromptUpdate][vllm.multimodal.processing.PromptUpdate] 指定的。我们可以通过检查更新后 Token 的存在性来自动检测 HF 是否更新了提示词。

## 标记化的提示词输入 (Tokenized Prompt Inputs)

为了支持在单独的进程中进行 Token 标记化（Tokenization），我们支持在传入多模态数据的同时传入输入的 Token ID。

### 遇到的问题

考虑到 HF 处理器通常遵循以下几个主要步骤：

1. 对文本进行分词（Tokenize）
2. 处理多模态输入
3. 执行提示词更新

而我们的要求是：

- 对于“文本 + 多模态输入”，应用全部的步骤 1 到 3。
- 对于“已 Token 化的 ID + 多模态输入”，仅应用步骤 2 到 3。

如何在不重写 HF 处理器的前提下实现这一点？我们可以尝试针对不同输入多次调用 HF 处理器：

- 对于“文本 + 多模态输入”，直接调用 HF 处理器即可。
- 对于“已 Token 化的 ID + 多模态输入”，仅针对多模态输入调用处理器。

虽然 HF 处理器原生支持“文本 + 多模态输入”，但它并不支持“已 Token 化的 ID + 多模态输入”：如果输入占位符 Token 的数量与多模态输入的数量不对应，它会抛出错误。

此外，由于已 Token 化的文本没有经过 HF 处理器，我们必须自己应用“步骤 3”来保持输出 Token 和多模态数据之间的一致性。

### 虚拟文本 (Dummy text)

我们通过要求每个模型定义如何根据多模态输入的数量生成虚拟文本（通过 [get_dummy_text][vllm.multimodal.processing.BaseDummyInputsBuilder.get_dummy_text]）来规避第一个问题。这使我们能够生成与多模态输入相对应的虚拟文本并共同输入，从而获得处理后的多模态数据。

### 自动提示词更新 (Automatic prompt updating)

我们通过在 [_apply_prompt_updates][vllm.multimodal.processing.BaseMultiModalProcessor._apply_prompt_updates] 中实现与模型无关的代码来解决第二个问题，即根据 [_get_prompt_updates][vllm.multimodal.processing.BaseMultiModalProcessor._get_prompt_updates] 输出的规范，使用特征占位符 Token 自动更新提示词。

### 总结

借助虚拟文本和自动提示词更新，我们的多模态处理器最终可以同时接受文本提示词和 Token ID 提示词与多模态数据。详细逻辑参见 [_apply_hf_processor_main][vllm.multimodal.processing.BaseMultiModalProcessor._apply_hf_processor_main]。

## 处理器输出缓存 (Processor Output Caching)

一些 HF 处理器（例如 Qwen2-VL 的处理器）运行[非常缓慢](https://github.com/vllm-project/vllm/issues/9238)。为了缓解这个问题，我们缓存了 HF 处理器的多模态输出，以避免重复处理相同的多模态输入（如图像）。

当新数据传入时，我们首先检查哪些项已在缓存中，哪些项缺失。缺失的项将被打包在单个批次中传给 HF 处理器处理并进行缓存，然后与缓存中已有的项进行合并。

由于我们只处理缺失的多模态数据项，输入占位符 Token 的数量不再与多模态输入的数量对应，因此它们不能与文本提示词一起传递给 HF 处理器。因此，我们分别处理文本和多模态输入，使用[虚拟文本](#_1)来避免 HF 报错。由于这跳过了 HF 的提示词更新代码，我们随后会应用[自动提示词更新](#_2)，以保持输出 Token 和多模态数据之间的一致性。
