# 多模态编码器上的 torch.compile (torch.compile with Multimodal Encoders)

`torch.compile` 现在可以应用于 vLLM 中的多模态编码器（Multimodal Encoders）以及其他混杂的 `nn.Module`，包括像 LLaMA 4、Qwen-VL 这样的视觉语言模型以及类似的基于编码器结构的架构。

本文档涵盖了 vLLM 中多模态编码器的 `torch.compile` 集成基础，以及如何将该装饰器应用于新模型以提升性能。

!!! note "注意"
    关于 vLLM 中 `torch.compile` 集成的一般信息，请参阅 [torch.compile 设计文档](./torch_compile.md)。

## 概述 (Overview)

我们最近启用了 `@support_torch_compile` 装饰器，使其能够处理一个模型类型内的多个 `nn.Module` 组件；这使得能够为多模态编码器开启编译，从而为架构堆栈中的其他组件带来性能提升。

当应用于 [`Qwen2_5_vl`](https://github.com/vllm-project/vllm/pull/23207) 的视觉模块时，我们观察到端到端（e2e）性能提升约 4.5%，但编译时间有所增加。

此功能默认关闭。当模型被添加了 `@support_torch_compile` 装饰器后，可以在编译配置中设置 `compile_mm_encoder: true` 来启用此功能。

## 多模态组件的编译机制

### 启用的 API

要编译像编码器这样的多模态组件，我们遵循与大语言模型（LLM）文本主干网络（Text backbone）相同的机制，并辅以一些额外的脚手架设计：

1. `@support_torch_compile` 装饰器应当包含 `enable_if=should_torch_compile_mm_encoder`。这将把编译控制逻辑收拢到我们的 `compile_mm_encoder` 配置参数之后。

2. 对于编码器组件，`@support_torch_compile` 装饰器应当包含 `is_encoder=True`。这对于编译范围（Compile Range）集成是必需的（参见“编译范围集成”）。该装饰器会自动将类名作为缓存目录的前缀，从而避免在独立编译的子模块之间产生冲突（例如，视觉编码器组件与文本主干网络之间）。

### CompilationConfig (编译配置)

除了 `compile_mm_encoder: true` 之外，多模态编码器将继承与文本 LLM 相同的编译配置。未来我们可能会扩展此配置以引入更多配置参数。

## 将 torch.compile 应用于新的多模态模型/组件

要在新的通用 `nn.Module` 上应用 `support_torch_compile`，我们建议遵循 [`debug_vllm_compile`](./debug_vllm_compile.md) 中提到的相同步骤，这包括：

1. 首先将 `support_torch_compile` 应用在较小的模块上（例如基础的 MLP 层），然后逐步提高到更通用的模块，直至达到良好的性能权衡。

2. 利用 [`tlparse`](https://github.com/meta-pytorch/tlparse) 来识别并消除导致重新编译（Recompile）和图中断（Graph break）的根源。

3. 使用 `dynamic_arg_dims` 和适当的 `dynamic_shapes_config` 来处理动态变化。

### 常见陷阱

## VllmBackend 特性支持

### 编译范围 (Compile ranges)

`torch.compile` 的集成将尝试依赖于 `max_batch_size` 来推断动态形状（Dynamic shapes）的编译范围；然而，对于编码器中使用的模块，由于编码器作为输入时接收到的形状范围是不确定的，这一形状很难推断。因此，我们在 `@support_torch_compile` 装饰器中依赖于 `is_encoder=True`，以提醒 `torch.compile` 这一范围无法被推断，并且我们将该范围默认设置为 (1, MAX_INT)。

!!! note "注意"
    未来我们可能会缩窄这一范围以获得更好的性能。

### CUDA 图 (Cudagraphs)

我们尚未探索将多模态编码器的编译与 CUDA 图（CUDAGraph）进行集成；当前此行为是未定义的。

## 问题排查 (Troubleshooting)

### 视觉编码器中的图中断 (Graph Breaks in Vision Encoders)

某些视觉编码器操作可能会导致图中断。要识别它们，请运行：

```bash
TORCH_LOGS="+dynamo" vllm serve <MODEL>
```

多模态模型中导致图中断的常见原因：

- **动态图像尺寸**：使用 `dynamic_shapes_config` 处理可变的图像分辨率。
- **无法被追踪的操作**：某些操作（如 `to_list`）可能不被 Dynamo 支持。
- **条件处理**：基于图像属性的与数据相关的条件分支。

### 编译报错

如果多模态模型的编译失败：

1. **禁用并测试**：首先验证模型在不启用编译的情况下是否能正常工作：
   ```bash
   vllm serve <model> --compilation-config='{"mode":0,"compile_mm_encoder":"false"}'
   ```

2. **检查日志**：启用调试日志以查看详细的编译细节：
   ```bash
   VLLM_LOGGING_LEVEL=DEBUG vllm serve <model> --compilation-config='{"compile_mm_encoder":"true"}'
   ```

3. **报告问题**：如果您发现 bug，请在 [GitHub 上提交 Issue](https://github.com/vllm-project/vllm/issues/new/choose)。

## 另请参阅 (See Also)

- [torch.compile 集成](./torch_compile.md) - 核心设计文档
- [调试 torch.compile](./debug_vllm_compile.md) - 详细调试指南
- [多模态输入](../features/multimodal_inputs.md) - 如何传递多模态数据
- [解耦编码器](../features/disagg_encoder.md) - 扩展视觉编码器
- [支持的多模态模型](../models/supported_models.md#list-of-multimodal-language-models) - 模型兼容性
