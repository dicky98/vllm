# 如何调试 vLLM-torch.compile 集成 (How to debug the vLLM-torch.compile integration)

**快速参考：**

- 使用 `tlparse` 获取 `torch.compile` 日志。在提交 Bug 报告和/或寻求技术支持时请附带这些日志。
- vLLM-torch.compile 的集成包含多个部分。vLLM 暴露了相应的标志来关闭其中的每个部分：

| 在线服务标志 (Online Flag) | 离线推理标志 (Offline Flag) | 对应结果 |
|--------------------------------|--------------------------------|--------------------------------------|
| `--enforce-eager` | `enforce_eager=True` | 关闭 `torch.compile` 和 CUDA 图 |
| `-cc.mode=0` | `compilation_config=CompilationConfig(mode=CompilationMode.NONE)` | 仅关闭 `torch.compile` |
| `-cc.mode=1` | `compilation_config=CompilationConfig(mode=CompilationMode.STOCK_TORCH_COMPILE)` | 关闭 vLLM 对 `torch.compile` 的修改（使用原生 `torch.compile`） |
| `-cc.cudagraph_mode=NONE` | `compilation_config=CompilationConfig(cudagraph_mode=CUDAGraphMode.NONE)` | 仅关闭 CUDA 图 |
| `-cc.backend=eager` | `compilation_config=CompilationConfig(backend='eager')` | 关闭 TorchInductor 后端 |
| `-cc.ir_enable_torch_wrap=False` | `compilation_config=CompilationConfig(ir_enable_torch_wrap=False)` | 关闭 vLLM IR 包装 (Wrapping) |

## vLLM-torch.compile 概述

为了提高性能，vLLM 利用 `torch.compile` 和 CUDA 图来加速推理。`torch.compile` 为 PyTorch 代码生成优化的算子核（Kernels），而 CUDA 图消除了主机端的启动开销。

最值得注意的是，**vLLM-compile 并不是常规的 torch.compile**，它是利用 PyTorch Compile 内部 API 构建的自定义编译器。

![vLLM-compile 架构图](../assets/design/debug_vllm_compile/design_diagram.png)

- 给定一个模型，我们通过 TorchDynamo 对其进行完整图捕获，该图在 Batch Size（Token 数量）上是动态的。
- 随后，vLLM 可以选择对该图进行切分（Splitting）和/或特化（Specializing），然后使用 TorchInductor 将每个子图编译为编译产物。此步骤可能会使用 vLLM 的自定义 Inductor Passes 来进一步优化图。这包括降低 vLLM IR 以消除分发开销。
- 编译产物会被保存到 vLLM 的编译缓存（Compile Cache）中，以便在未来直接加载。
- vLLM 应用 CUDA 图以减少 CPU 开销。

上述四个步骤中的任何一步都可能出错。一旦出错，请尽量隔离出错的子系统 —— 这能让您仅关闭最少的部分来维持可靠性目标，同时将对性能的影响降到最低，这也有助于我们（vLLM 官方团队）分析您提交的 Bug 报告。

有关设计细节的更多信息，请参考以下资源：

- [vLLM-torch.compile 集成介绍博客](https://blog.vllm.ai/2025/08/20/torch-compile.html)
- [vLLM-torch.compile 集成设计文档](./torch_compile_zh.md)
- [vLLM IR 设计文档](./vllm_ir_zh.md)
- [vLLM 答疑时间 #26 (Office Hours #26)](https://www.youtube.com/live/xLyxc7hxCJc?si=Xulo9pe53C6ywf0V&t=561)
- [在 PyTorch Conference 2025 上的演讲](https://youtu.be/1wV1ESbGrVQ?si=s1GqymUfwiwOrDTg&t=725)

## 使用 tlparse

使用 [tlparse](https://github.com/meta-pytorch/tlparse) 查看 `torch.compile` 的日志。这些日志展示了编译过程的各个阶段，包括 `torch.compile` 生成的融合算子核（Fused kernels）。

安装 tlparse：

```sh
pip install tlparse
```

要启用 `torch.compile` 日志，您可以设置环境变量 `TORCH_TRACE=<dir>`。在 Trace 过程中，该目录下会为每个 Rank 创建一个文件，每个文件都包含编译期间产生的各种伪影。如果可以，我们建议在提交 Bug 报告时附带这些日志文件 —— 它们非常有帮助。

离线推理用法：

```sh
TORCH_TRACE=~/trace_dir python my_script.py
tlparse ~/trace_dir/<rank_0_log_file>
```

在线服务用法：

```sh
TORCH_TRACE=~/trace_dir vllm serve
# 通过 ctrl-c 退出服务
tlparse ~/trace_dir/<rank_0_log_file>
```

在给定一个日志文件的情况下，`tlparse` 命令会输出一些 HTML 文件（例如输出到 `./tl_out/index.html`）。用浏览器打开它即可查看日志，界面类似于下图所示：

![tlparse 示例](../assets/design/debug_vllm_compile/tlparse_inductor.png)

## 关闭 vLLM-torch.compile 集成

传递 `--enforce-eager` 以完全关闭 vLLM-torch.compile 集成并彻底在 Eager 模式下运行，这也包括关闭 CUDA 图。

```sh
# 在线服务
vllm serve --enforce-eager
```

```py
# 离线推理
LLM(model, enforce_eager=True)
```

如果只想关闭 `torch.compile`，请在编译配置中传递 `mode = NONE`（`-cc` 是 `--compilation_config` 的简写）：

```sh
# 在线服务
vllm serve -cc.mode=0
```

```py
# 离线推理
from vllm.config.compilation import CompilationConfig, CompilationMode
LLM(model, compilation_config=CompilationConfig(mode=CompilationMode.NONE))
```

如果只想关闭 CUDA 图，请传递 `cudagraph_mode = NONE`：

```sh
# 在线服务
vllm serve -cc.cudagraph_mode=NONE
```

```py
# 离线推理
from vllm.config.compilation import CompilationConfig, CUDAGraphMode
LLM(model, compilation_config=CompilationConfig(cudagraph_mode=CUDAGraphMode.NONE))
```

vLLM IR 重度依赖编译流水线（包括函数化、自定义融合和 Lowering 转换）。若要关闭它并捕获 vLLM IR 的 Eager 模式分发行为，请在运行中加上 `ir_enable_torch_wrap=False`。IR torch wrap 仅在默认使用 `mode=VLLM_COMPILE` 且 `backend="inductor"`（默认值）时启用。

```sh
# 在线服务
vllm serve -cc.ir_enable_torch_wrap=False
```

```py
# 离线推理
from vllm.config.compilation import CompilationConfig
LLM(model, compilation_config=CompilationConfig(ir_enable_torch_wrap=False))
```

## 调试 TorchDynamo

vLLM 要求模型代码必须能够通过 TorchDynamo（`torch.compile` 的前端）被捕获为一个完整图。TorchDynamo 并不能支持所有的 Python 语法。在完整图（Fullgraph）模式下，如果遇到不支持的特性，它就会报错（通常被称为图中断 graph break）。

如果您遇到了图中断问题，请在 [pytorch/pytorch 仓库提交 Issue](https://github.com/pytorch/pytorch) 以便 PyTorch 开发者排期解决。在此期间，请尽最大努力重写代码以避免该图中断。
有关更多信息，请参考此 [Dynamo 指南](https://docs.pytorch.org/docs/stable/compile/programming_model.dynamo_core_concepts.html)。

## 调试动态形状完整图捕获 (Debugging Dynamic Shape full graph capture)

vLLM 要求模型的前向传播能够被捕获为一个在 Batch Size（即 Token 数量）上是动态的完整图。它在默认情况下会将这单个图编译为一个编译产物，并将该产物复用于所有的 Batch Size。

如果您的代码无法在启用动态形状（Dynamic Shapes）的情况下被捕获，您可能会遇到隐式错误、显式报错或 CUDA 非法内存访问。例如，以下代码就无法被捕获为单个图：

```py
if data.size[0] % 128 == 0:
    foo(...)
else:
    bar(...)
```

此问题很容易诊断。使用 `tlparse` 并点击 `compilation_metrics`：它会告诉您在 Batch Size 上的符号约束（Symbolic constraints）。如果存在任何限制了 Batch Size 的约束，那么事情就变得麻烦了。

![非良性 tlparse 示例](../assets/design/debug_vllm_compile/dynamic_shapes.png)

为了避免此问题，请采用以下方法之一：

1. 避免对 Token 数量进行条件分支判断（Branching）。
2. 将条件分支逻辑封装到一个自定义算子（Custom operator）中。TorchDynamo 不会追踪进入自定义算子的内部。

## 调试约束冲突和动态形状 Guards 问题

动态形状 Guards 是 Dynamo Guards 的一个特定类别。它们是 `torch.compile` 附加到动态维度（例如 `seq_len`）上的约束，以确保编译后的产物保持有效。当框架代码、自定义 Passes 或用户代码基于动态形状的值进行分支时，通常会出现这些 Guards。

**示例：**

```python
if x > 10:
    # 路径 A
else:
    # 路径 B
```

根据被追踪的是哪条路径，这将创建一个 Guard：`x > 10` 或 `x <= 10`。

**vLLM 的假设：**
vLLM 假设所有由 `torch.compile` 添加的 Guards 都可以被安全地丢弃，并且不会限制编译后的图必须对应特定的输入形状。当此假设被违反时，会引发需要用户调试的问题。暗示该假设被违反的一些副作用包括运行时错误或 `ConstraintViolationErrors`。

如果动态形状被约束为单个值，则会抛出 `ConstraintViolationErrors`。如果您遇到了约束冲突错误，或者怀疑某个动态形状 Guard 被错误地添加了，您可以使用更严格的动态形状模式来帮助调试该问题：

```sh
# 在线服务 - 使用 unbacked 模式
vllm serve meta-llama/Llama-3.2-1B -cc.dynamic_shapes_config.type=unbacked

# 在线服务 - 使用 backed_size_oblivious 模式
vllm serve meta-llama/Llama-3.2-1B -cc.dynamic_shapes_config.type=backed_size_oblivious
```

```py
# 离线推理 - 使用 unbacked 模式
from vllm.config.compilation import CompilationConfig, DynamicShapesConfig, DynamicShapesType
LLM(model, compilation_config=CompilationConfig(
    dynamic_shapes_config=DynamicShapesConfig(type=DynamicShapesType.UNBACKED)
))

# 离线推理 - 使用 backed_size_oblivious 模式
from vllm.config.compilation import CompilationConfig, DynamicShapesConfig, DynamicShapesType
LLM(model, compilation_config=CompilationConfig(
    dynamic_shapes_config=DynamicShapesConfig(type=DynamicShapesType.BACKED_SIZE_OBLIVIOUS)
))
```

这些模式更为严格，可以减少或消除对动态形状 Guards 的需要，从而有助于隔离问题：

- `unbacked`：使用不允许 Guards 的 Unbacked Symints，从而更容易找出是在何处被错误地添加了 Guards。
- `backed_size_oblivious`：使用一种对 Guarding 更加严格的模式。

有关动态形状模式的更多细节，请参阅 [动态形状和 vLLM 守卫丢弃](torch_compile_zh.md#vllm-guards)。

### 打印 Guards

为了查看在编译期间被添加的所有 Guards，您可以使用 `TORCH_LOGS=+dynamic`：

```sh
TORCH_LOGS=+dynamic vllm serve meta-llama/Llama-3.2-1B
```

在日志中查找 `[guard added]` 可以看到在何处添加了 Guards。这可以帮助您识别是哪些操作导致了 Guards 被错误地添加。

## 调试 TorchInductor

TorchInductor 接收捕获到的图，然后将其编译为可能会调用一个或多个 Triton Kernel 的 Python 代码。在少数（但极其不幸的）情况下，它可能会生成错误的 Triton Kernel。这可能会表现为隐式错误、CUDA 非法内存访问或显式报错。

### Inductor 运行时断言 (Inductor runtime assertions)

默认情况下（在 `torch < 2.12` 上），vLLM 会禁用 Inductor 的运行时断言（`assert_size_stride`、`assert_alignment`），以避免在大模型上每次前向传播引入约 2 毫秒的开销。设置 `VLLM_LOGGING_LEVEL=DEBUG` 会自动重新启用它们，从而使得调试会话能获得完整的形状/步长（Shape/Stride）校验：

```sh
VLLM_LOGGING_LEVEL=DEBUG vllm serve <model>
```

您也可以通过 `--compilation-config` 显式覆盖它们：

```sh
vllm serve <model> -cc.inductor_compile_config='{"size_asserts": true, "alignment_asserts": true, "scalar_asserts": true}'
```

在 `torch >= 2.12` 上，PyTorch 采用了高效的“单次断言（assert-once）”策略，这些标志不再被 vLLM 所压制。

要调试是否是 TorchInductor 出了问题，您可以通过在编译配置中传入 `backend='eager'` 来禁用它：

```sh
# 在线服务
vllm serve -cc.backend=eager
```

```py
# 离线推理
LLM(compilation_config=CompilationConfig(backend='eager'))
```

如果是 Inductor 的问题，请向 [PyTorch 官方提交 Bug](https://github.com/pytorch/pytorch)。如果您觉得富有冒险精神，可以直接调试 Inductor 输出代码中的 Triton Kernel（您可以通过使用 `tlparse` 来定位这些代码）。

![tlparse 示例](../assets/design/debug_vllm_compile/tlparse_inductor.png)

您也可以使用 `TORCH_LOGS=output_code <command>` 来打印 Inductor 输出的代码。

### 可编辑的 TorchInductor 代码

您可以通过设置 `VLLM_COMPILE_CACHE_SAVE_FORMAT=unpacked` 或传入 `-cc.compile_cache_save_format=unpacked`，来编辑随后运行的 TorchInductor 代码。默认值是 `binary`，意味着它是不可编辑的。

这是一种非常有用的技术：您可以在输出代码中放入断点（例如 `torch.distributed.breakpoint()`）和打印语句。

## 调试 vLLM-compile 缓存

vLLM 为 `torch.compile` 产物构建了它自己的缓存。其设计思想是，产物在编译一次后，可以在之后被重复使用。这是在 [torch.compile 自身编译器缓存](https://docs.pytorch.org/tutorials/recipes/torch_compile_caching_tutorial.html) 之上封装的又一层缓存。

虽然 `torch.compile` 的编译器缓存极其稳定，但不幸的是，vLLM 的编译器缓存并不总是正确的。您可以通过设置 `VLLM_DISABLE_COMPILE_CACHE=1` 来禁用它。

您也可以手动清除此缓存：

- 通过 `rm -rf ~/.cache/vllm` 移除 vLLM 的编译缓存（请查看日志以确认缓存位置是否改变）。
- 通过 `rm -rf /tmp/torchinductor_$(whoami)` 移除 `torch.compile` 内置的缓存。

vLLM 的缓存是从缓存键（Cache key）到已编译产物的映射。vLLM 通过结合多个因素（例如配置标志和模型名称）来计算缓存键。如果 vLLM 的编译缓存出错，这通常意味着遗漏了某个考量因素。关于 vLLM 如何计算部分缓存键，请参见 [此示例](https://github.com/vllm-project/vllm/blob/18b39828d90413d05d770dfd2e2f48304f4ca0eb/vllm/config/model.py#L310)。

vLLM 的编译缓存要求被编译的代码最终是可序列化的（Serializable）。如果不是这样，它在保存时会报错。通常的修复方法是：

- 重写不可序列化的部分（这可能有些困难，因为目前很难判断哪些可序列化，哪些不可序列化）。
- 提交 Bug 报告。
- 通过设置 `VLLM_DISABLE_COMPILE_CACHE=1` 忽略此错误（注意：这会使服务器热启动变慢很多）。

## 调试 CUDA 图

CUDA 图（CUDAGraphs）特性允许您：

- 将调用了一个或多个 CUDA Kernel 的可调用对象捕获到 CUDA 图中。
- 重新回放（Replay）该 CUDA 图。

被捕获的 CUDA 图包含了在捕获过程中所使用的所有内存。回放该 CUDA 图时会读写完全相同的内存区域。

这带来了一些限制：

1. 为了在新的数据上使用 CUDA 图，您需要将数据复制到 CUDA 图在读取的缓冲区中。
2. CUDA 图仅捕获 CUDA Kernel，它们不会捕获在 CPU 上完成的工作。

vLLM 使用了原生 CUDA 图 API，如果使用不当，这是不安全的。

如果只想关闭 CUDA 图，请传递 `cudagraph_mode = NONE`：

```sh
# 在线服务
vllm serve -cc.cudagraph_mode=NONE
```

```py
# 离线推理
from vllm.config.compilation import CompilationConfig, CUDAGraphMode
LLM(model, compilation_config=CompilationConfig(cudagraph_mode=CUDAGraphMode.NONE))
```
