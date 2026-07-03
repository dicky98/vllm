# 自定义算子 (CustomOp)

`CustomOp` 是一个抽象类，用于将各种操作的前向传播方法（forward method）分发到对应的计算后端。它还为 vLLM 内部和树外（OOT, Out-Of-Tree）插件提供了一种注册其自定义操作的机制。

本文档将介绍 `CustomOp` 在 vLLM 中的工作原理以及如何实现一个新的 `CustomOp`。

## CustomOp 在 vLLM 中的工作原理

`CustomOp` 类中管理了两个保存所有自定义算子的字典（以注册的名称为索引），分别用于 vLLM 自身和 OOT 插件。

我们可以使用 `@CustomOp.register("op_name")` 将算子类注册到 `CustomOp` 系统中。注册后，`op_name` 及其对应的类将被添加到 `op_registry` 字典中。此外，我们还可以通过 `@CustomOp.register_oot("op_name")` 注册 OOT 算子。我们稍后会详细介绍这一机制。

当调用 `CustomOp` 时（即调用其 `forward()` 方法），如果它被启用（即通过 `--compilation_config.custom_ops '["+op_name"]'`），它会根据当前平台 `current_platform` 自动将前向传播方法分发到适当的后端。否则（即它被禁用时），它将仅调用 `forward_native()` 方法以使用该前向传播方法的 PyTorch 原生实现。

具体的平台分发规则如下：

- **CPU 平台**：分发到 `forward_cpu()`。
- **CUDA 平台**：分发到 `forward_cuda()`。
- **ROCm 平台**：分发到 `forward_hip()`。如果未实现 `forward_hip()`，它将使用 `forward_cuda()` 作为回退方案。
- **XPU 平台**：分发到 `forward_xpu()`。
- **TPU 平台**：分发到 `forward_tpu()`。
- **OOT 平台**：分发到 `forward_oot()`。这只会在 OOT 平台上被调用。
- **默认规则**：分发到 `forward_native()` 作为所有平台的最底线回退方案。

!!! note "注意"
    由于类继承的存在，分发逻辑可能不是绝对的。派生类可能会重写此行为。

此外，vLLM 根据 `compilation_config.custom_ops` 决定是启用还是禁用某个 `CustomOp`。具体来说，如果一个 `CustomOp` 没有在 `compilation_config.custom_ops` 中被指定（即使用默认配置），若 `compilation_config.custom_ops` 中包含 `all`，则该算子会被启用；若包含 `none`，则会被禁用。

!!! note "注意"
    请注意，`all` 和 `none` 不能同时存在于 `compilation_config.custom_ops` 中。

默认情况下，如果 `compilation_config.backend == "inductor"` 且 `compilation_config.mode != CompilationMode.NONE`，系统会在 `compilation_config.custom_ops` 中追加一个 `none`，否则追加 `all`。换句话说，这意味着在以 PyTorch Compile 模式运行时，在某些平台（即那些默认使用 `inductor` 作为 `torch.compile` 后端的平台）上会禁用 `CustomOp`。在这种情况下，Inductor 会为那些被禁用的自定义算子生成（融合后的）Triton Kernel。

!!! note "注意"
    对于多模态模型，vLLM 强制启用了一些自定义算子，以在 ViT 部分使用设备特定的深度优化 Kernel 来获得更好的性能，例如 `MMEncoderAttention` 和 `ApplyRotaryEmb`。我们也可以向 `CustomOp` 的 `__init__()` 方法传递一个 `enforce_enable=True` 参数，以在对象级别上强制启用其自身。

    需要注意的是，在我们为多模态部分添加独立的 `compilation_config` 之后，这种 `enforce_enable` 机制将被移除。

## 如何为 CustomOp 自定义配置

vLLM 还为用户提供了对启用或禁用哪些自定义算子的细粒度控制，这可以通过在启动服务器时手动传递 `--compilation_config.custom_ops '["..."]'` 来实现。

例如：

- 使用 `--compilation_config.custom_ops '["all"]'` 启用所有自定义算子。
- 使用 `--compilation_config.custom_ops '["none"]'` 禁用所有自定义算子。
- 使用 `--compilation_config.custom_ops '["all,-op1"]'` 启用除 `op1` 以外的所有自定义算子（即前缀为 `-` 表示“禁用”）。
- 使用 `--compilation_config.custom_ops '["none,+op1,+op2"]'` 仅启用 `op1` 和 `op2`（即前缀为 `+` 表示“启用”）。

## vLLM 中支持的 CustomOp 类型

**1. 注意力操作 (Attention)：**

```python
--8<-- "vllm/model_executor/layers/mla.py:multi_head_latent_attention"
```

**2. 激活函数 (Activation)：**

```python
--8<-- "vllm/model_executor/layers/activation.py:silu_and_mul"

--8<-- "vllm/model_executor/layers/activation.py:mul_and_silu"

--8<-- "vllm/model_executor/layers/activation.py:gelu_new"

--8<-- "vllm/model_executor/layers/activation.py:gelu_fast"

--8<-- "vllm/model_executor/layers/activation.py:quick_gelu"

--8<-- "vllm/model_executor/layers/activation.py:gelu_and_mul"

--8<-- "vllm/model_executor/layers/activation.py:gelu_and_mul_sparse"

--8<-- "vllm/model_executor/layers/activation.py:relu2"

--8<-- "vllm/model_executor/layers/activation.py:xielu"

--8<-- "vllm/model_executor/layers/activation.py:swigluoai_and_mul"

--8<-- "vllm/model_executor/layers/activation.py:fatrelu_and_mul"
```

**3. 多模态卷积 (MM-Conv)：**

```python
--8<-- "vllm/model_executor/layers/conv.py:conv2d"

--8<-- "vllm/model_executor/layers/conv.py:conv3d"
```

**4. 嵌入层 (Embedding)：**

```python
--8<-- "vllm/model_executor/layers/vocab_parallel_embedding.py:vocab_parallel_embedding"

--8<-- "vllm/model_executor/layers/vocab_parallel_embedding.py:parallel_lm_head"
```

**5. 线性层 (Linear)：**

```python
--8<-- "vllm/model_executor/layers/linear.py:row_parallel_linear"

--8<-- "vllm/model_executor/layers/linear.py:column_parallel_linear"

--8<-- "vllm/model_executor/layers/linear.py:replicated_linear"
```

**6. Logits 处理器 (Logits Processor)：**

```python
--8<-- "vllm/model_executor/layers/logits_processor.py:logits_processor"
```

**7. Mamba 算子：**

```python
--8<-- "vllm/model_executor/layers/mamba/mamba_mixer.py:mamba_mixer"

--8<-- "vllm/model_executor/layers/mamba/mamba_mixer2.py:mamba_mixer2"

--8<-- "vllm/model_executor/layers/mamba/mamba_mixer2.py:mixer2_gated_rms_norm"

--8<-- "vllm/model_executor/models/plamo2.py:plamo2_mamba_mixer"

--8<-- "vllm/model_executor/layers/mamba/short_conv.py:short_conv"
```

**8. 混合专家模型 (MoE)：**

```python
--8<-- "vllm/model_executor/layers/fused_moe/layer.py:fused_moe"

--8<-- "vllm/model_executor/layers/fused_moe/fused_moe_modular_method.py:modular_fused_moe"

--8<-- "vllm/model_executor/layers/fused_moe/unquantized_fused_moe_method.py:unquantized_fused_moe"

--8<-- "vllm/model_executor/models/transformers/moe.py:transformers_fused_moe"

--8<-- "vllm/model_executor/layers/fused_moe/router/grouped_topk_router.py:grouped_topk"
```

**9. 归一化 (Norm)：**

```python
--8<-- "vllm/model_executor/layers/layernorm.py:rms_norm"

--8<-- "vllm/model_executor/layers/layernorm.py:rms_norm_gated"

--8<-- "vllm/model_executor/layers/layernorm.py:gemma_rms_norm"
```

**10. 量化 (Quantization)：**

```python
--8<-- "vllm/model_executor/layers/quantization/input_quant_fp8.py:quant_fp8"
```

**11. 位置编码 (Rope)：**

```python
--8<-- "vllm/model_executor/layers/rotary_embedding/base.py:rotary_embedding"

--8<-- "vllm/model_executor/layers/rotary_embedding/dual_chunk_rope.py:dual_chunk_rotary_embedding"

--8<-- "vllm/model_executor/layers/rotary_embedding/common.py:apply_rotary_emb"
```

**12. 编码器 (Encoder)：**

```python
--8<-- "vllm/model_executor/models/deepencoder2.py:qwen2_decoder"

--8<-- "vllm/model_executor/layers/attention/mm_encoder_attention.py:mm_encoder_attn"

--8<-- "vllm/model_executor/models/deepencoder.py:rel_pos_attention"
```

## 实现新 CustomOp 指南

### 在 vLLM 中实现一个新的 CustomOp

本部分是在 vLLM 中实现新 `CustomOp` 的教程。

步骤：

1. 实现一个新的算子类，该类继承自 `CustomOp` 基类。
2. 在该算子类上添加 `@CustomOp.register("op_name")` 装饰器，将其注册到 `CustomOp` 系统中。
3. 根据您的需要实现不同的 `forward_xxx()` 方法。

以 `MMEncoderAttention` 为例：

??? code

    ```python
    @CustomOp.register("mm_encoder_attn")
    class MMEncoderAttention(CustomOp):

        def __init__(
            self,
            num_heads: int,
            head_size: int,
            scale: float | None = None,
            num_kv_heads: int | None = None,
            prefix: str = "",
            multimodal_config: MultiModalConfig | None = None,
        ) -> None:
            super().__init__()
            # 初始化代码...

        def forward_native(
            self,
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            cu_seqlens: torch.Tensor | None = None,
            max_seqlen: torch.Tensor | None = None,  # 仅供 Flash Attention 使用
        ) -> torch.Tensor:
            # 调用 TORCH_SDPA 实现...

        def forward_cuda(
            self,
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            cu_seqlens: torch.Tensor | None = None,
            max_seqlen: torch.Tensor | None = None,  # 仅供 Flash Attention 使用
        ) -> torch.Tensor:
            # 调用 FA 或 TORCH_SDPA 实现...

        def forward_cpu(
            self,
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            cu_seqlens: torch.Tensor | None = None,
            max_seqlen: torch.Tensor | None = None,  # 仅供 Flash Attention 使用
        ) -> torch.Tensor:
            # 调用 TORCH_SDPA 实现...

        def forward_xpu(
            self,
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            cu_seqlens: torch.Tensor | None = None,
            max_seqlen: torch.Tensor | None = None,  # 仅供 Flash Attention 使用
        ) -> torch.Tensor:
            # 调用 FA 实现...

        def forward_tpu(
            self,
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            cu_seqlens: torch.Tensor | None = None,
            max_seqlen: torch.Tensor | None = None,  # 仅供 Flash Attention 使用
        ) -> torch.Tensor:
            # 调用 PALLAS 实现...
    ```

### 在 OOT 设备插件中注册新的 CustomOp

目前，得益于 [vLLM 的硬件插件机制](./plugin_system_zh.md)，涌现出了各种树外（OOT）设备插件，使得 vLLM 能够无缝运行在不同的硬件上。您还可以在 [介绍 vLLM 硬件插件：来自昇腾 NPU 的最佳实践](https://blog.vllm.ai/2025/05/12/hardware-plugin.html) 中了解有关该机制的更多细节。

- **官方设备插件**：[vllm-ascend](https://github.com/vllm-project/vllm-ascend)（针对华为昇腾 NPU）、[vllm-spyre](https://github.com/vllm-project/vllm-spyre)（针对 Spyre）、[vllm-gaudi](https://github.com/vllm-project/vllm-gaudi)（针对英特尔 Gaudi）、[vllm-neuron](https://github.com/vllm-project/vllm-neuron)（针对 AWS Neuron）、[vllm-metal](https://github.com/vllm-project/vllm-metal)（针对 Apple Silicon）等。
- **非官方设备插件**：[vllm-metax](https://github.com/MetaX-MACA/vLLM-metax)（针对沐曦 GPU）、[vllm-kunlun](https://github.com/baidu/vLLM-Kunlun)（针对百度昆仑 XPU）、[vllm-musa](https://github.com/MooreThreads/vllm-musa)（针对摩尔线程 GPU）等。

在这种情况下，`CustomOp` 可以使这些硬件制造商仅需注册一个 OOT `CustomOp` 并实现 `forward_oot()` 方法，便可以在运行时无缝地用他们针对特定设备进行深层优化的 Kernel 替换 vLLM 的算子操作。

现在，这部分将向您展示如何为设备插件注册一个 OOT `CustomOp`。

以 `MMEncoderAttention` 为例：

1. 实现一个继承自 `MMEncoderAttention` 的 `CustomMMEncoderAttention` 类，并实现其 `forward_oot()` 方法。
2. 在 vLLM 中注册您的 `CustomMMEncoderAttention` 以替换原来的 `MMEncoderAttention`。

??? code

    ```python
    from vllm.model_executor.layers.attention import MMEncoderAttention
    from vllm.model_executor.custom_op import CustomOp


    @CustomOp.register_oot("MMEncoderAttention")
    class CustomMMEncoderAttention(MMEncoderAttention):

        def __init__(...):
            super().__init__(...)

        def forward_oot(...):
            # 调用针对特定设备进行过深度优化的 Kernels
            ...
    ```

在这种情况下，一个新条目 `{"MMEncoderAttention": CustomMMEncoderAttention}` 将被添加到 `op_registry_oot` 中。当初始化 `MMEncoderAttention` 算子对象时，如果类名（即 `MMEncoderAttention`）包含在 `op_registry_oot` 的键中，vLLM 将会用我们注册的类（即 `CustomMMEncoderAttention`）替换并实例化它。

此后，当调用此 `MMEncoderAttention` 算子时，如果其被启用，您的 `forward_oot()` 将会被调用。这样，您就可以在不直接修改 vLLM 代码的情况下，在您的硬件上获得预期的性能。

此外，您还可以将所有的 `CustomOp` 注册在同一个地方，以便更好地进行管理。

??? code

    ```python
    from vllm.model_executor.custom_op import CustomOp


    REGISTERED_CUSTOM_OPS = {
        "CustomOP1": YourCustomOp1,
        "CustomOP2": YourCustomOp2,
        "CustomOP3": YourCustomOp3,
    }

    for op_name, op_cls in REGISTERED_CUSTOM_OPS.items():
        CustomOp.register_oot(_decorated_op_cls=op_cls, name=op_name)
    ```
