# 输入输出处理器插件 (IO Processor Plugins)

输入输出处理器插件（IO Processor plugins）是一项允许对池化模型（Pooling models）的模型输入和输出进行前处理和后处理的功能。其设计思想是允许用户向 vLLM 传递自定义输入，该输入会被转换为一个或多个模型提示词，然后馈送给模型的 `encode` 方法。此类插件的一个潜在用例是在 vLLM 中用于生成多模态数据。例如，用户向 vLLM 输入一张图像，并在输出中得到另一张图像。

在使用 IO 处理器插件执行推理时，提示词类型由该插件定义，最终的请求输出也是如此。vLLM 不对输入/输出数据进行任何验证，由插件来确保将正确的数据提供给模型并返回给用户。截至目前，这些插件仅支持池化模型，并可以通过 `LLM` 和 `AsyncLLM` 中的 `encode` 方法触发，或者在在线服务模式下通过 `/pooling` 端点触发。

## 编写一个 IO 处理器插件

IO 处理器插件需要实现 [`IOProcessor`][vllm.plugins.io_processors.interface.IOProcessor] 抽象接口：

```python
IOProcessorInput = TypeVar("IOProcessorInput")
IOProcessorOutput = TypeVar("IOProcessorOutput")

class IOProcessor(ABC, Generic[IOProcessorInput, IOProcessorOutput]):
    """对引擎输入/输出进行前处理/后处理的抽象接口。"""

    def __init__(self, vllm_config: VllmConfig, renderer: BaseRenderer):
        super().__init__()

        self.vllm_config = vllm_config

    def parse_data(self, data: object) -> IOProcessorInput:
        raise NotImplementedError

    def merge_sampling_params(
        self,
        params: SamplingParams | None = None,
    ) -> SamplingParams:
        return params or SamplingParams()

    def merge_pooling_params(
        self,
        params: PoolingParams | None = None,
    ) -> PoolingParams:
        return params or PoolingParams(task="plugin")

    @abstractmethod
    def pre_process(
        self,
        prompt: IOProcessorInput,
        request_id: str | None = None,
        **kwargs,
    ) -> PromptType | Sequence[PromptType]:
        raise NotImplementedError

    async def pre_process_async(
        self,
        prompt: IOProcessorInput,
        request_id: str | None = None,
        **kwargs,
    ) -> PromptType | Sequence[PromptType]:
        return self.pre_process(prompt, request_id, **kwargs)

    @abstractmethod
    def post_process(
        self,
        model_output: Sequence[PoolingRequestOutput],
        request_id: str | None = None,
        **kwargs,
    ) -> IOProcessorOutput:
        raise NotImplementedError

    async def post_process_async(
        self,
        model_output: AsyncGenerator[tuple[int, PoolingRequestOutput]],
        request_id: str | None = None,
        **kwargs,
    ) -> IOProcessorOutput:
        # 我们无法保证返回的输出顺序与馈送给 vLLM 的顺序相同。
        # 让我们在后处理之前按 id 对它们进行排序
        sorted_output = sorted(
            [(i, item) async for i, item in model_output], key=lambda output: output[0]
        )
        collected_output = [output[1] for output in sorted_output]
        return self.post_process(collected_output, request_id=request_id, **kwargs)
```

- `parse_data` 方法用于验证用户数据并将其转换为 `pre_process*` 方法所期望的输入。
- `merge_sampling_params` 和 `merge_pooling_params` 方法将输入的 `SamplingParams` 或 `PoolingParams`（如果有）与默认参数合并。
- `pre_process*` 方法接收经验证的插件输入，生成供 vLLM 进行常规推理的模型提示词。
- `post_process*` 方法接收 `PoolingRequestOutput` 对象作为输入，生成自定义的插件输出。

在 PrithviGeospatialMAE 模型中生成 GeoTIFF 图像的插件示例实现参见[这里](https://github.com/IBM/terratorch/tree/main/terratorch/vllm/plugins/segmentation)。另外，也请参考我们的在线（[examples/pooling/plugin/prithvi_geospatial_mae_online.py](../../examples/pooling/plugin/prithvi_geospatial_mae_online.py)）和离线（[examples/pooling/plugin/prithvi_geospatial_mae_io_processor.py](../../examples/pooling/plugin/prithvi_geospatial_mae_io_processor.py)）推理示例。

## 使用一个 IO 处理器插件

IO 处理器插件是在引擎启动时加载的，有两种方法可以指定要加载的插件名称：

1. **通过 vLLM 的 `EngineArgs`**：在用于初始化 `AsyncLLM` 的 `EngineArgs` 中设置 `io_processor_plugin` 参数。在离线模式下，可以通过将 `io_processor_plugin` 参数传递给 `LLM` 来实现同样的目的；在在线服务模式下，也可以传递 `--io-processor-plugin` 参数。
2. **通过模型的 HF 配置**：在模型配置文件（`config.json`）中添加 `io_processor_plugin` 字段。

配置的顺序也决定了优先级。也就是说，通过 `EngineArgs` 设置的插件名称会覆盖在模型 HF 配置文件（`config.json`）中指定的任何插件名称。
