# 插件系统 (Plugin System)

社区经常会有扩展 vLLM 以实现自定义特性的需求。为了方便这一点，vLLM 包含了一个插件系统，允许用户添加自定义功能，而无需修改 vLLM 代码库。本文档解释了插件在 vLLM 中的工作原理以及如何为 vLLM 创建插件。

## 插件在 vLLM 中的工作原理

插件是用户注册并由 vLLM 执行的代码。鉴于 vLLM 的架构（参见 [架构概述](arch_overview_zh.md)），在运行分布式推理且使用各种并行技术时，可能会涉及多个进程。为了成功启用插件，vLLM 创建的每个进程都需要加载该插件。这由 `vllm.plugins` 模块中的 [load_plugins_by_group][vllm.plugins.load_plugins_by_group] 函数完成。

## vLLM 如何发现插件

vLLM 的插件系统使用标准的 Python `entry_points` 机制。此机制允许开发人员在他们的 Python 包中注册函数，以供其他包使用。一个插件的示例如下：

??? code

    ```python
    # setup.py 文件内部
    from setuptools import setup

    setup(name='vllm_add_dummy_model',
        version='0.1',
        packages=['vllm_add_dummy_model'],
        entry_points={
            'vllm.general_plugins':
            ["register_dummy_model = vllm_add_dummy_model:register"]
        })

    # vllm_add_dummy_model/__init__.py 文件内部
    def register():
        from vllm import ModelRegistry

        if "MyLlava" not in ModelRegistry.get_supported_archs():
            ModelRegistry.register_model(
                "MyLlava",
                "vllm_add_dummy_model.my_llava:MyLlava",
            )
    ```

有关如何在您的包中添加 entry points 的更多信息，请参考 [官方文档](https://setuptools.pypa.io/en/latest/userguide/entry_point.html)。

每个插件包含三个部分：

1. **插件组 (Plugin group)**：entry point 组的名称。vLLM 使用名为 `vllm.general_plugins` 的 entry point 组来注册通用插件。这是 `setup.py` 文件中 `entry_points` 的键（key）。对于 vLLM 的通用插件，请始终使用 `vllm.general_plugins`。
2. **插件名称 (Plugin name)**：插件的名称。这是 `entry_points` 字典中对应项的值字典的键名。在上面的例子中，插件名称是 `register_dummy_model`。可以使用 `VLLM_PLUGINS` 环境变量来过滤插件名称。如果只想加载特定的插件，请将 `VLLM_PLUGINS` 设置为该插件的名称。
3. **插件值 (Plugin value)**：要在插件系统中注册的函数或模块的完全限定名称（Fully qualified name）。在上面的例子中，插件值是 `vllm_add_dummy_model:register`，它指向 `vllm_add_dummy_model` 模块中名为 `register` 的函数。

## 支持的插件类型

- **通用插件 (General plugins)**（组名为 `vllm.general_plugins`）：这些插件的主要用例是将自定义的树外（Out-of-tree）模型注册到 vLLM 中。这是通过在插件函数中调用 `ModelRegistry.register_model` 注册模型来完成的。关于官方模型插件的示例，参见 [bart-plugin](https://github.com/vllm-project/bart-plugin)，它为 `BartForConditionalGeneration` 添加了支持。

- **平台插件 (Platform plugins)**（组名为 `vllm.platform_plugins`）：这些插件的主要用例是将自定义的树外平台注册到 vLLM 中。当当前环境不支持该平台时，插件函数应返回 `None`；当支持该平台时，应返回平台类的完全限定名称。

- **IO 处理器插件 (IO Processor plugins)**（组名为 `vllm.io_processor_plugins`）：这些插件的主要用例是为池化模型注册模型提示词和模型输出的自定义前/后处理。插件函数返回 `IOProcessor` 类的完全限定名称。

- **统计日志记录器插件 (Stat logger plugins)**（组名为 `vllm.stat_logger_plugins`）：这些插件的主要用例是将自定义的树外日志记录器注册到 vLLM 中。其 entry point 应为继承自 `StatLoggerBase` 的类。

## 编写插件指南

- **重入性要求 (Being re-entrant)**：在 entry point 中指定的函数应该是可重入的，即它可以被多次调用而不会引发问题。这在一些进程中多次调用该函数时是必要的。

### 平台插件指南

1. 创建一个平台插件项目，例如 `vllm_add_dummy_platform`。项目结构应如下所示：

    ```shell
    vllm_add_dummy_platform/
    ├── vllm_add_dummy_platform/
    │   ├── __init__.py
    │   ├── my_dummy_platform.py
    │   ├── my_dummy_worker.py
    │   ├── my_dummy_attention.py
    │   ├── my_dummy_device_communicator.py
    │   ├── my_dummy_custom_ops.py
    ├── setup.py
    ```

2. 在 `setup.py` 文件中，添加以下 entry point：

    ```python
    setup(
        name="vllm_add_dummy_platform",
        ...
        entry_points={
            "vllm.platform_plugins": [
                "my_dummy_platform = vllm_add_dummy_platform:register"
            ]
        },
        ...
    )
    ```

    请确保 `vllm_add_dummy_platform:register` 是一个可调用函数，并返回平台类的完全限定名称。例如：

    ```python
    def register():
        return "vllm_add_dummy_platform.my_dummy_platform.MyDummyPlatform"
    ```

3. 在 `my_dummy_platform.py` 中实现平台类 `MyDummyPlatform`。该平台类应该继承自 `vllm.platforms.interface.Platform`。请根据接口定义逐一实现各项函数。以下是至少应当被实现的一些重要函数和属性：

    - `_enum`：此属性是来自 [PlatformEnum][vllm.platforms.interface.PlatformEnum] 的设备枚举。通常它应当是 `PlatformEnum.OOT`，表示该平台是树外（Out-of-tree）平台。
    - `device_type`：此属性应当返回 PyTorch 使用的设备类型。例如 `"cpu"`、`"cuda"` 等。
    - `device_name`：此属性通常设置为与 `device_type` 相同。它主要用于日志记录。
    - `check_and_update_config`：该函数在 vLLM 的初始化流程中很早就被调用。它用于供插件更新 vLLM 配置（例如，在此函数中更新块大小、图模式配置等）。最重要的是，必须在此函数中设置 **`worker_cls`**，以便让 vLLM 知道工作进程要使用哪个 Worker 类。
    - `get_attn_backend_cls`：该函数应当返回注意力后端类的完全限定名称。
    - `get_device_communicator_cls`：该函数应当返回设备通信器类的完全限定名称。

4. 在 `my_dummy_worker.py` 中实现 Worker 类 `MyDummyWorker`。Worker 类应当继承自 [WorkerBase][vllm.v1.worker.worker_base.WorkerBase]。请根据接口定义逐一实现各项函数。基本上，基类中的所有接口都需要被实现，因为它们在 vLLM 的各个地方都会被调用。为了确保模型能够被成功执行，应当实现以下基础函数：

    - `init_device`：该函数被调用以对 Worker 的设备进行设置。
    - `initialize_cache`：该函数被调用以对 Worker 的缓存配置进行设置。
    - `load_model`：该函数被调用以将模型权重加载到设备中。
    - `get_kv_cache_spec`：该函数被调用以生成模型的 KV 缓存规范。
    - `determine_available_memory`：该函数被调用以分析模型的峰值内存占用情况，从而确定在不发生 OOM 的情况下，有多少内存可供 KV 缓存使用。
    - `initialize_from_config`：该函数被调用以使用指定的 `kv_cache_config` 分配设备端 KV 缓存。
    - `execute_model`：该函数在每一个推理步骤中被调用以执行模型推理。

    其他可选实现的函数包括：

    - 如果插件想要支持睡眠模式特性，请实现 `sleep` 和 `wakeup` 函数。
    - 如果插件想要支持图编译模式特性，请实现 `compile_or_warm_up_model` 函数。
    - 如果插件想要支持投机解码特性，请实现 `take_draft_token_ids` 函数。
    - 如果插件想要支持 LoRA 特性，请实现 `add_lora`、`remove_lora`、`list_loras` 和 `pin_lora` 函数。
    - 如果插件想要支持数据并行（DP）特性，请实现 `execute_dummy_batch` 函数。

    请参阅工作基类 [WorkerBase][vllm.v1.worker.worker_base.WorkerBase] 以了解更多可以实现的函数。

5. 在 `my_dummy_attention.py` 中实现注意力后端类 `MyDummyAttention`。注意力后端类应当继承自 [AttentionBackend][vllm.v1.attention.backend.AttentionBackend]。它用于在您的设备上计算注意力机制。您可以参考 `vllm.v1.attention.backends`，其中包含许多注意力后端的具体实现。

6. 实现自定义算子（Custom Ops）以获得高性能。虽然大多数算子都可以通过 PyTorch 的原生实现来运行，但性能可能不佳。在这种情况下，您可以为插件实现特定的自定义算子。目前，vLLM 支持以下类型的自定义算子：

    - **PyTorch 算子**：
      有 3 种 PyTorch 算子：
        - `通信器算子 (communicator ops)`：设备通信器算子，如 all-reduce、all-gather 等。请在 `my_dummy_device_communicator.py` 中实现设备通信器类 `MyDummyDeviceCommunicator`。该设备通信器类应当继承自 [DeviceCommunicatorBase][vllm.distributed.device_communicators.base_device_communicator.DeviceCommunicatorBase]。
        - `通用算子 (common ops)`：通用算子，如 matmul、softmax 等。请通过注册树外（OOT）算子的方式实现通用算子。详细细节参见 [CustomOp][vllm.model_executor.custom_op.CustomOp] 类。
        - `csrc 算子`：C++ 算子。这类算子在 C++ 中实现，并作为 PyTorch 自定义算子注册。遵循 csrc 模块和 `vllm._custom_ops` 来实现您的算子。
    - **Triton 算子**：
      目前对于 Triton 算子，自定义方式尚不生效。

7. （可选）实现其他可插拔模块，例如 LoRA、Graph 后端、量化、Mamba 注意力后端等。

## 兼容性保证 (Compatibility Guarantee)

vLLM 保证已文档化的插件接口（例如 `ModelRegistry.register_model`）将始终可供插件用于注册模型。然而，插件开发者有责任确保其插件与他们所针对的 vLLM 版本兼容。例如，`"vllm_add_dummy_model.my_llava:MyLlava"` 应当与该插件所针对的 vLLM 版本兼容。

在 vLLM 的开发过程中，模型/模块的接口可能会发生变化。如果您看到任何弃用（Deprecation）日志信息，请将您的插件升级到最新版本。

## 弃用声明 (Deprecation Announcement)

!!! warning "弃用警告"
    - `Platform.get_attn_backend_cls` 中的 `use_v1` 参数已废弃。它已在 v0.13.0 版本中被移除。
    - `vllm.attention` 中的 `_Backend` 已废弃。它已在 v0.13.0 版本中被移除。请改用 `vllm.v1.attention.backends.registry.register_backend` 将新的注意力后端添加到 `AttentionBackendEnum`。
    - `seed_everything` 平台接口已废弃。它已在 v0.16.0 版本中被移除。请改用 `vllm.utils.torch_utils.set_random_seed`。
    - `Platform.validate_request` 中的 `prompt` 已废弃。它已在 v0.18.0 版本中被移除。
