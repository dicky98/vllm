# 量化 (Quantization)

量化通过牺牲模型精度来换取更小的内存占用，从而使大型模型能够在更广泛的设备上运行。

!!! tip
    要开始使用量化，请参阅 [LLM Compressor](llm_compressor/README.md)。这是一个用于优化 vLLM 部署模型的库，支持 FP8、INT8、INT4 和其他量化格式。

以下是 vLLM 支持的量化格式：

- [AutoAWQ](auto_awq.md)
- [BitsAndBytes](bnb.md)
- [GPTQModel](gptqmodel.md)
- [Intel Neural Compressor](inc.md)
- [LLM Compressor](llm_compressor/README.md)
    - [FP8 W8A8](llm_compressor/fp8.md)
    - [INT4 W4A16](llm_compressor/int4.md)
    - [INT8 W4A8](llm_compressor/int8_w4a8.md)
    - [INT8 W8A8](llm_compressor/int8_w8a8.md)
- [NVIDIA Model Optimizer](modelopt.md)
- [在线量化 (Online Quantization)](online.md)
- [AMD Quark](quark.md)
- [量化 KV 缓存 (Quantized KV Cache)](quantized_kvcache.md)
- [TorchAO](torchao.md)
- [FP8 ViT 编码器注意力](fp8_vit_attn.md)

## 支持的硬件 (Supported Hardware)

下表展示了 vLLM 中各种量化实现与不同硬件平台的兼容性：

<style>
td:not(:first-child) {
  text-align: center !important;
}
td {
  padding: 0.5rem !important;
  white-space: nowrap;
}

th {
  padding: 0.5rem !important;
  min-width: 0 !important;
}

th:not(:first-child) {
  writing-mode: vertical-lr;
  transform: rotate(180deg)
}
</style>

| 实现方式 (Implementation) | Volta | Turing | Ampere | Ada | Hopper | AMD GPU | Intel GPU | x86 CPU | Arm CPU |
| ------------------------- | ----- | ------ | ------ | --- | ------ | ------- | --------- | ------- | ------- |
| AWQ                       | ❌    | ✅︎     | ✅︎     | ✅︎  | ✅︎     | ❌      | ✅︎        | ✅︎      | ❌      |
| GPTQ                      | ✅︎    | ✅︎     | ✅︎     | ✅︎  | ✅︎     | ❌      | ✅︎        | ✅︎      | ❌      |
| Marlin (GPTQ/AWQ/FP8/FP4) | ❌    | ✅︎*    | ✅︎     | ✅︎  | ✅︎     | ❌      | ❌        | ❌      | ❌      |
| llm-compressor INT8 (W8A8)| ❌    | ✅︎     | ✅︎     | ✅︎  | ✅︎     | ❌      | ❌        | ✅︎      | ✅︎      |
| llm-compressor INT8 (W4A8)| ❌    | ❌     | ❌     | ❌  | ❌     | ❌      | ❌        | ❌      | ✅︎      |
| llm-compressor FP8 (W8A8) | ❌    | ❌     | ❌     | ✅︎  | ✅︎     | ✅︎      | ❌        | ❌      | ❌      |
| bitsandbytes              | ✅︎    | ✅︎     | ✅︎     | ✅︎  | ✅︎     | ❌      | ❌        | ❌      | ❌      |
| DeepSpeedFP               | ✅︎    | ✅︎     | ✅︎     | ✅︎  | ✅︎     | ❌      | ❌        | ❌      | ❌      |
| GGUF                      | ✅︎    | ✅︎     | ✅︎     | ✅︎  | ✅︎     | ✅︎      | ❌        | ❌      | ❌      |

- Volta 指的是 SM 7.0，Turing 指的是 SM 7.5，Ampere 指的是 SM 8.0/8.6，Ada 指的是 SM 8.9，Hopper 指的是 SM 9.0。
- ✅︎ 表示指定的硬件支持该量化方法。
- ❌ 表示指定的硬件不支持该量化方法。
- 所有 Intel Gaudi 的量化支持已迁移至 [vLLM-Gaudi](https://github.com/vllm-project/vllm-gaudi) 仓库。
- *Turing 架构不支持 Marlin MXFP4。

!!! note
    关于 Google TPU 上的量化支持信息，请参考 [TPU 推理推荐模型和特性](https://docs.vllm.ai/projects/tpu/en/latest/recommended_models_features/) 文档。

!!! note
    随着 vLLM 的不断演进以及对不同硬件平台和量化方法的支持扩展，此兼容性表可能会发生变化。

    有关硬件支持和量化方法的最新信息，请参考 [vllm/model_executor/layers/quantization](../../../vllm/model_executor/layers/quantization) 或咨询 vLLM 开发团队。

## 树外量化插件 (Out-of-Tree Quantization Plugins)

vLLM 支持使用 `@register_quantization_config` 装饰器注册自定义的、树外（Out-of-tree）量化方法。这允许您在不修改 vLLM 代码库的情况下实现和使用自己的量化方案。

### 注册自定义量化方法

要注册自定义量化方法，请创建一个继承自 `QuantizationConfig` 的类，并使用 `@register_quantization_config` 对其进行装饰。其中的 `get_quant_method` 根据层类型分发到对应的量化方法中：

```python
import torch
from vllm.model_executor.layers.quantization import (
    register_quantization_config,
)
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from vllm.model_executor.layers.linear import LinearBase
from vllm.model_executor.layers.fused_moe import FusedMoE

@register_quantization_config("my_quant")
class MyQuantConfig(QuantizationConfig):
    """自定义量化配置。"""

    def get_name(self) -> str:
        return "my_quant"

    def get_supported_act_dtypes(self) -> list:
        return [torch.float16, torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        # 最小 GPU 计算能力需求，-1 表示无限制
        return -1

    @staticmethod
    def get_config_filenames() -> list[str]:
        # 在模型目录中寻找的配置文件名称列表
        return []

    @classmethod
    def from_config(cls, config: dict) -> "MyQuantConfig":
        # 从模型的量化配置字典中创建配置
        return cls()

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> QuantizeMethodBase | None:
        # 根据层类型进行分发
        # 注意：您只需要实现您关心的层类型的方法
        if isinstance(layer, LinearBase):
            return MyQuantLinearMethod()
        elif isinstance(layer, FusedMoE):
            return MyQuantMoEMethod(layer.moe_config)
        return None
```

### QuantizationConfig 必须实现的方法

您的自定义 `QuantizationConfig` 子类必须实现以下抽象方法：

| 方法名 | 描述 |
| ------ | ----------- |
| `get_name()` | 返回量化方法的名称 |
| `get_supported_act_dtypes()` | 返回支持的激活值数据类型列表（例如 `torch.float16`） |
| `get_min_capability()` | 返回最小 GPU 计算能力（例如 Ampere 架构为 80，-1 表示无限制） |
| `get_config_filenames()` | 返回要在模型目录中搜索的配置文件名列表 |
| `from_config(config)` | 类方法，用于从模型的量化配置字典创建配置对象 |
| `get_quant_method(layer, prefix)` | 返回给定层的量化方法，若跳过该层则返回 `None` |

### 实现量化线性层方法 (Quantized Linear Method)

对于线性层，需要从 `get_quant_method` 返回一个 `QuantizeMethodBase` 的子类。您可以继承 `UnquantizedLinearMethod` 作为起点：

```python
from vllm.model_executor.layers.linear import UnquantizedLinearMethod

class MyQuantLinearMethod(UnquantizedLinearMethod):
    """用于线性层的自定义量化方法。"""

    def create_weights(
        self, layer: torch.nn.Module, *weight_args, **extra_weight_attrs
    ):
        # 为该层创建量化权重
        ...

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # 在此应用自定义的量化计算逻辑
        ...
```

### 实现量化 MoE 层方法 (Quantized MoE Method)

对于混合专家（MoE）模型，需要从 `get_quant_method` 返回一个 `FusedMoEMethodBase` 子类。您可以使用 `UnquantizedFusedMoEMethod` 来跳过 MoE 的量化：

```python
from vllm.model_executor.layers.fused_moe.layer import UnquantizedFusedMoEMethod
from vllm.model_executor.layers.fused_moe.fused_moe_method_base import (
    FusedMoEMethodBase,
)
from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig

class MyQuantMoEMethod(FusedMoEMethodBase):
    """用于 MoE 层的自定义量化方法。"""

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        # 为 MoE 层创建量化权重
        ...

    def apply(
        self,
        layer: torch.nn.Module,
        router: "FusedMoERouter",
        x: torch.Tensor,
        router_logits: torch.Tensor,
    ) -> torch.Tensor:
        # 使用量化权重应用 MoE 计算
        ...

    def get_fused_moe_quant_config(
        self, layer: torch.nn.Module
    ) -> FusedMoEQuantConfig | None:
        # 返回 MoE 量化配置
        ...
```

参考现有的实现，例如 `vllm/model_executor/layers/quantization/fp8.py` 中的 `Fp8MoEMethod`。

### 使用插件

注册完成后，您就可以在 vLLM 中使用您的自定义量化方法：

```python
# 注册您的量化方法（导入包含您配置的模块）
import my_quant_plugin

from vllm import LLM

# 使用自定义量化方法
llm = LLM(model="your-model", quantization="my_quant")
```

有关插件系统的更多信息，请参阅 [插件系统文档](../../design/plugin_system.md)。
