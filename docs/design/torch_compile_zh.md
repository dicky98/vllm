# `torch.compile` 集成设计 (torch.compile integration)

在 vLLM 的 V1 架构中，`torch.compile` 是默认启用的，且是该框架的关键组成部分。本文档通过一个简单的逐步示例，来展示如何理解 `torch.compile` 的使用。

在整个示例中，我们将运行一个通用的 Llama 模型，并开启 Debug 级别的日志记录以显示所有细节。所使用的命令为：`VLLM_LOGGING_LEVEL=DEBUG vllm serve meta-llama/Llama-3.2-1B`。

!!! note "注意"
    有关 `torch.compile` 集成的更多信息和最新进展，请参阅这篇 [博客文章](https://blog.vllm.ai/2025/08/20/torch-compile.html)。

## 编译缓存 (Compilation Cache)

在非常详尽的日志中，我们可以看到：

```console
INFO 03-07 03:06:55 [backends.py:409] Using cache directory: ~/.cache/vllm/torch_compile_cache/1517964802/rank_0_0 for vLLM's torch.compile
```

vLLM 会综合考虑所有可用因素，并决定一个用于存储所有编译产物（Compilation artifact）的目录。这意味着，在您的部署场景中，您可以直接复制整个 `~/.cache/vllm/torch_compile_cache` 目录以节省大量的编译时间，从而加速 vLLM 实例的启动。

所考量的因素包括：

- 所有相关的配置（参阅 [config 目录](../../vllm/config) 下各自配置中的 `compute_hash` 函数）。
- PyTorch 配置（参阅 [compiler_interface.py](../../vllm/compilation/compiler_interface.py) 中的 `compute_hash` 函数）。
- 模型的 forward 函数以及被 forward 函数调用的相关函数（见下文）。

在考量了所有这些因素后，我们通常可以保证该缓存是安全可用的，不会导致任何异常行为。因此，该缓存默认是启用的。如果您想调试编译过程，或者怀疑缓存引起了某些问题，可以通过设置环境变量 `VLLM_DISABLE_COMPILE_CACHE=1` 来禁用它。

vLLM 中 `torch.compile` 集成的一个独到之处在于，我们保证在服务任何请求之前完成所有的编译。任何请求都不会触发新的编译。否则，引擎会因为该请求而阻塞，响应时间会出现意料之外的峰值。

默认情况下，缓存会将编译后的产物保存为二进制文件。如果您希望与生成的代码进行交互以进行调试，可以在编译配置中设置 `compile_cache_save_format=unpacked`，或者不设置它而直接设置环境变量 `VLLM_COMPILE_CACHE_SAVE_FORMAT=unpacked`。

## 动态形状与 vLLM 守卫丢弃 (Dynamic shapes and vllm guard dropping)

`torch.compile` 的设计旨在需要时毫不犹豫地对动态形状进行守卫校验（Guards）。这与 vLLM 中丢弃 Guards 的 `torch.compile` 方法相冲突，因为许多此类 Guards 可能是实质性的。

`torch.compile` 提供了两种动态形状：`backed`（有后端支持的）和 `unbacked`（无后端支持的）。

`torch.compile` 对 `backed` 动态形状进行 Guards 校验，并不保证不会对其添加 Guards。用户代码、Dynamo、Inductor 和 Autograd 都可能添加 Guards。此外，对于 0/1 特化，即使在没有遇到这些范围的分支的情况下，Backed 符号也会被无条件特化为 0、1 或 >=2。

相反，`unbacked` 动态形状保证不会被 Guards 校验，并且不进行 0/1 特化。然而，当遇到需要它们数值的分支且未定义显式的 Unbacked 处理时，可能会抛出与数据相关的错误（Data dependent error, DDE）。该框架正在向不抛出 DDE 而是选择通用路径（General paths）的状态收敛。使用 Unbacked 的一个缺点是，由于性能 Bug 或选择通用路径，可能会错失一些优化机会，而且使用了一个固定的、非基于示例输入的 Hint（这很快将通过 `override_hint` API 进行修复）。选择通用路径的一个例子是，当无法在符号上证明有克隆引入的改变时，在函数调用 `contiguous()` 和 `reshape()` 中假设输入是不连续的。

`backed_size_oblivious` 是一个 Flag，它允许在定义了对 Unbacked 的显式处理的地方，将 Backed 符号视作 Unbacked 处理。在这种模式下，框架代码中基本上避免了 0/1 特化，并且不会发生默认的 0/1 特化。然而，仍然无法保证 `torch.compile` 不会进行 Guards 校验，特别是由于用户代码或自定义 Passes。`backed_size_oblivious` 在 PyTorch 编译中是实验性的，未来可能会被废弃。尽管如此，它仍然是一个比 `backed` 更安全的选择，且降低性能的概率低于 `unbacked`。

### 配置动态形状 (Configuring Dynamic Shapes)

`DynamicShapesConfig` 允许您通过设置 `type` 字段来控制动态形状的行为。您可以从三种模式中进行选择：`BACKED`（默认值）、`UNBACKED` 和 `BACKED_SIZE_OBLIVIOUS`。

#### 离线推理示例（使用 LLM 类）

当使用 `LLM` 类进行离线推理时，您可以通过 `compilation_config` 参数来配置动态形状：

```python
from vllm import LLM, SamplingParams
from vllm.config.compilation import CompilationConfig, DynamicShapesConfig, DynamicShapesType

# 示例：使用 backed_size_oblivious（实验性，比 backed 更安全）
llm = LLM(
    model="meta-llama/Llama-3.2-1B",
    compilation_config=CompilationConfig(
        dynamic_shapes_config=DynamicShapesConfig(
            type=DynamicShapesType.BACKED_SIZE_OBLIVIOUS
        )
    )
)

# 示例：使用 unbacked（防范 Guards 的最强保证）
llm = LLM(
    model="meta-llama/Llama-3.2-1B",
    compilation_config=CompilationConfig(
        dynamic_shapes_config=DynamicShapesConfig(
            type=DynamicShapesType.UNBACKED
        )
    )
)

# 生成输出
prompts = ["Hello, my name is", "The future of AI is"]
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
outputs = llm.generate(prompts, sampling_params)
```

#### 在线服务示例（使用 vllm serve）

当使用 `vllm serve` 进行在线服务时，您可以通过 `--compilation-config` 标志配置动态形状：

```bash
# 示例：使用 unbacked
vllm serve meta-llama/Llama-3.2-1B \
  --compilation-config '{"dynamic_shapes_config": {"type": "unbacked"}}'

# 备选：使用点号记法（对于单一值更简便）
vllm serve meta-llama/Llama-3.2-1B -cc.dynamic_shapes_config.type=unbacked
```

#### 选择合适的模式

- **`BACKED`**（默认）：当您愿意接受潜在不安全的 Guards 丢弃以换取最大性能时使用。Guards 可能会被不合理地添加，然后被忽略。
- **`UNBACKED`**：当您需要最强有力的防范 Guards 的保证时使用。这是最保守的选择，但可能会错失一些优化机会。
- **`BACKED_SIZE_OBLIVIOUS`**：当您想在避免 Guards 与性能之间取得平衡时使用。此实验性模式比 `BACKED` 更安全，但仍没有 `UNBACKED` 那么保守。

## Python 代码编译 (Python Code Compilation)

在非常详尽的日志中，我们可以看到：

??? console "日志详情"

      ```text
      DEBUG 03-07 03:06:52 [decorators.py:203] Start compiling function <code object forward at 0x7f08acf40c90, file "xxx/vllm/model_executor/models/llama.py", line 339>

      DEBUG 03-07 03:06:54 [backends.py:370] Traced files (to be considered for compilation cache):
      DEBUG 03-07 03:06:54 [backends.py:370] xxx/torch/_dynamo/polyfills/builtins.py
      DEBUG 03-07 03:06:54 [backends.py:370] xxx/torch/nn/modules/container.py
      DEBUG 03-07 03:06:54 [backends.py:370] xxx/torch/nn/modules/module.py
      DEBUG 03-07 03:06:54 [backends.py:370] xxx/vllm/attention/layer.py
      DEBUG 03-07 03:06:54 [backends.py:370] xxx/vllm/distributed/communication_op.py
      DEBUG 03-07 03:06:54 [backends.py:370] xxx/vllm/distributed/parallel_state.py
      DEBUG 03-07 03:06:54 [backends.py:370] xxx/vllm/model_executor/custom_op.py
      DEBUG 03-07 03:06:54 [backends.py:370] xxx/vllm/model_executor/layers/activation.py
      DEBUG 03-07 03:06:54 [backends.py:370] xxx/vllm/model_executor/layers/layernorm.py
      DEBUG 03-07 03:06:54 [backends.py:370] xxx/vllm/model_executor/layers/linear.py
      DEBUG 03-07 03:06:54 [backends.py:370] xxx/vllm/model_executor/layers/rotary_embedding.py
      DEBUG 03-07 03:06:54 [backends.py:370] xxx/vllm/model_executor/layers/vocab_parallel_embedding.py
      DEBUG 03-07 03:06:54 [backends.py:370] xxx/vllm/model_executor/models/llama.py

      DEBUG 03-07 03:07:07 [backends.py:462] Computation graph saved to ~/.cache/vllm/torch_compile_cache/1517964802/rank_0_0/computation_graph.py
      DEBUG 03-07 03:07:07 [wrapper.py:105] Dynamo transformed code saved to ~/.cache/vllm/torch_compile_cache/1517964802/rank_0_0/transformed_code.py
      ```

这涉及 Python 代码编译，即 Dynamo 的图捕获（Graph capture）。它尝试追踪代码为 `xxx/vllm/model_executor/models/llama.py:339` 的函数，也就是我们编译的模型的 `forward` 函数。在前向传播期间，还有其他一些函数被 Dynamo 调用并内联，如日志所示，这包括来自 `xxx/torch/nn/modules/module.py` 的一些 PyTorch 函数（被 PyTorch `nn.Module` 使用，因为模块属性访问会触发函数调用），以及 vLLM 的一些通信 / 注意力 / 激活函数。当我们确定要使用的缓存目录时，所有被追踪的文件都会被考量在内。这样，上述文件中的任何代码更改都将触发编译缓存未命中，进而引发重新编译。

Dynamo 编译的结果是一个保存在 `~/.cache/vllm/torch_compile_cache/1517964802/rank_0_0/transformed_code.py` 中的新函数。通常，该函数从模块中解包张量，然后将其传递给被追踪的计算图。计算图保存在 `~/.cache/vllm/torch_compile_cache/1517964802/rank_0_0/computation_graph.py` 中。

## 计算图处理 (Computation Graph Processing)

计算图对每个张量都进行了形状标注。输入包括来自模型的输入 ID、位置 ID、权重和缓冲区，输出为最终的隐藏状态（Hidden states）。请注意，LM Head 投影和采样操作并未包含在图中。

计算图的大多数输入具有静态形状，因为它们是模型权重和缓冲区，在模型的整个生命周期内不会改变。只有输入 ID 和位置 ID 具有符号形状（Symbolic shapes），即形状可能会随批次而改变。然而，它们将共享相同的符号形状。也就是说，计算图唯一改变的大小就是 Batch Size（当前前向传播中处理的 Token 数量）。

注意力操作非常复杂，并且需要与具有复杂形状的 KV 缓存进行交互。幸运的是，注意力操作的输出与其输入 Query 的形状完全相同。因此，我们将整个注意力操作封装在一个 PyTorch 自定义算子 `torch.ops.vllm.unified_attention_with_output` 中，这样 Dynamo 就不会去检查任何内部操作。通过这种方式，尽管注意力操作非常复杂，但从 Dynamo 的视角来看，我们仍然能将模型的计算图作为一个完整图（Full-graph）来捕获。

计算图通过 `splitting_ops`（通常是注意力操作）被进一步切分为多个碎片。因此，在 `~/.cache/vllm/torch_compile_cache/1517964802/rank_0_0/computation_graph.py` 文件中，我们可以看到许多子模块，每个子模块都是切分后的一段计算图：

- 注意力操作本身是一个子模块。
- 从一个注意力操作到下一个注意力操作的计算图部分是一个子模块。

每个子模块都可以通过其索引进行标识，并将被单独处理。

## 计算图编译 (Computation Graph Compilation)

在非常详尽的日志中，我们还可以看到：

```console
DEBUG 03-07 03:52:37 [backends.py:134] store the 0-th graph for shape None from inductor via handle ('fpegyiq3v3wzjzphd45wkflpabggdbjpylgr7tta4hj6uplstsiw', '~/.cache/vllm/torch_compile_cache/1517964802/rank_0_0/inductor_cache/iw/ciwzrk3ittdqatuzwonnajywvno3llvjcs2vfdldzwzozn3zi3iy.py')
DEBUG 03-07 03:52:39 [backends.py:134] store the 1-th graph for shape None from inductor via handle ('f7fmlodmf3h3by5iiu2c4zarwoxbg4eytwr3ujdd2jphl4pospfd', '~/.cache/vllm/torch_compile_cache/1517964802/rank_0_0/inductor_cache/ly/clyfzxldfsj7ehaluis2mca2omqka4r7mgcedlf6xfjh645nw6k2.py')
...
DEBUG 03-07 03:52:45 [backends.py:134] store the 15-th graph for shape None from inductor via handle ('f7fmlodmf3h3by5iiu2c4zarwoxbg4eytwr3ujdd2jphl4pospfd', '~/.cache/vllm/torch_compile_cache/1517964802/rank_0_0/inductor_cache/ly/clyfzxldfsj7ehaluis2mca2omqka4r7mgcedlf6xfjh645nw6k2.py')
DEBUG 03-07 03:52:45 [backends.py:134] store the 16-th graph for shape None from inductor via handle ('fvj3ccoi7m34f3dnr4itmu55mmun44l5xymwhrjlwisylsk7q6jy', '~/.cache/vllm/torch_compile_cache/1517964802/rank_0_0/inductor_cache/tf/ctfftkglj7b4lcttq5cymx6cew372uoauupqn6ldsvpiucavqcjc.py')
```

这意味着第一段计算图（具有针对符号形状的 `None` 形状）由 Inductor 编译（键值为 `fpegyiq3v3wzjzphd45wkflpabggdbjpylgr7tta4hj6uplstsiw`）。编译后的 Kernel 保存在 `~/.cache/vllm/torch_compile_cache/1517964802/rank_0_0/inductor_cache/iw/ciwzrk3ittdqatuzwonnajywvno3llvjcs2vfdldzwzozn3zi3iy.py`。您可以打开该文件查看 Inductor 最终执行的代码。

还有一个细节：您可以看到第 1 个图和第 15 个图具有相同的键值，而第 0 个图和第 16 个图则是不同的。这是符合预期的，因为我们通过注意力操作对图进行了切分，从而得到了 3 个独特的子图：

- 注意力操作之前的首层。
- 从一个注意力操作到下一个注意力操作的每个中间层。
- 注意力操作之后的最后一层。

如果我们已经有了缓存目录（例如第二次运行相同的代码），我们将看到以下日志：

```console
DEBUG 03-07 04:00:45 [backends.py:86] Directly load the 0-th graph for shape None from inductor via handle ('fpegyiq3v3wzjzphd45wkflpabggdbjpylgr7tta4hj6uplstsiw', '~/.cache/vllm/torch_compile_cache/1517964802/rank_0_0/inductor_cache/iw/ciwzrk3ittdqatuzwonnajywvno3llvjcs2vfdldzwzozn3zi3iy.py')
```

这一次，Inductor 编译被完全旁路，我们将直接从磁盘加载上一次得到的编译产物。

上面的例子只是使用 Inductor 编译一个通用形状（即符号形状）。我们也可以使用 Inductor 来编译一些特定的形状，例如：

```bash
vllm serve meta-llama/Llama-3.2-1B \
  --compilation-config '{"compile_sizes": [1, 2, 4, 8]}'
```

然后它还会专门针对 Batch Size `1, 2, 4, 8` 编译一个特定的 Kernel。此时，计算图中的所有形状都是静态且已知的，我们将开启自动调优（Auto-tuning）以调优最大性能。这在您第一次运行时可能会很慢，但下一次运行时，我们可以直接跳过调优并运行调优好的 Kernel。

当所有形状已知时，`torch.compile` 可以比较不同的配置，并通常能找到一些更好的配置来运行 Kernel。例如，我们可以看到以下日志：

??? console "自动调优日志"

    ```
    AUTOTUNE mm(8x2048, 2048x3072)
      triton_mm_4 0.0130 ms 100.0% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=128, BLOCK_M=16, BLOCK_N=32, B_PROLOGUE_CAST_TYPE=None, EVEN_K=True, GROUP_M=8, num_stages=5, num_warps=2
      triton_mm_8 0.0134 ms 97.4% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=128, BLOCK_M=16, BLOCK_N=64, B_PROLOGUE_CAST_TYPE=None, EVEN_K=True, GROUP_M=8, num_stages=5, num_warps=4
      triton_mm_12 0.0148 ms 87.7% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=128, BLOCK_M=16, BLOCK_N=128, B_PROLOGUE_CAST_TYPE=None, EVEN_K=True, GROUP_M=8, num_stages=4, num_warps=4
      mm 0.0160 ms 81.6%
      triton_mm_16 0.0165 ms 78.7% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=64, BLOCK_M=16, BLOCK_N=128, B_PROLOGUE_CAST_TYPE=None, EVEN_K=True, GROUP_M=8, num_stages=5, num_warps=8
      triton_mm_3 0.0199 ms 65.4% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=32, BLOCK_M=16, BLOCK_N=32, B_PROLOGUE_CAST_TYPE=None, EVEN_K=True, GROUP_M=8, num_stages=5, num_warps=2
      triton_mm_1 0.0203 ms 64.2% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=128, BLOCK_M=16, BLOCK_N=32, B_PROLOGUE_CAST_TYPE=None, EVEN_K=True, GROUP_M=8, num_stages=2, num_warps=2
      triton_mm_7 0.0203 ms 64.1% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=64, BLOCK_M=16, BLOCK_N=64, B_PROLOGUE_CAST_TYPE=None, EVEN_K=True, GROUP_M=8, num_stages=3, num_warps=4
      triton_mm_2 0.0208 ms 62.5% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=32, BLOCK_M=16, BLOCK_N=64, B_PROLOGUE_CAST_TYPE=None, EVEN_K=True, GROUP_M=8, num_stages=5, num_warps=4
      triton_mm_11 0.0215 ms 60.5% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=64, BLOCK_M=16, BLOCK_N=128, B_PROLOGUE_CAST_TYPE=None, EVEN_K=True, GROUP_M=8, num_stages=3, num_warps=4
    SingleProcess AUTOTUNE benchmarking takes 2.0428 seconds and 7.5727 seconds precompiling
    ```

这意味着，对于一个形状为 `8x2048x3072` 的矩阵乘法，`torch.compile` 尝试了具有各种配置的 Triton 模板，它比默认代码（分发到 cuBLAS 库）要快得多。

不幸的是，由于自动调优需要相当长的时间（从几秒到几分钟，具体取决于模型大小和批次大小），尽管它可以缓存起来以供日后使用，为了用户友好起见，我们默认关闭了它。如果您追求极限性能，建议通过编译特定形状来尝试此功能。

## CUDA 图捕获 (Cudagraph Capture)

vLLM 的 V1 架构使用了与分段式编译保持一致的分段式 CUDA 图（piecewise cudagraph）。完整的计算图被如上所述切分，我们仅对注意力操作之间的那部分计算图进行 CUDA 图捕获（包括注意力操作之前的首个图，以及所有注意力操作之后的最后一个图）。这是基于一个常见的观察：注意力之间的计算通常是逐 Token 的，非常适合 CUDA 图；而注意力操作本身在实现 CUDA 图兼容时非常复杂。因此，通过以 Eager 模式运行注意力操作而对其余操作采用 CUDA 图，我们保留了注意力操作的灵活性。

分段式 CUDA 图也具有细粒度的内存管理。其目的是仅将注意力 Kernel 排除在 CUDA 图之外，而将所有其余模块和内存分配操作保留在 CUDA 图中。这就是为什么 V1 中的注意力操作将输出张量作为注意力输入的原因。

CUDA 图由编译器后端捕获并管理，并在批次大小与捕获的对应 CUDA 图相符时进行回放。模型的调用方（Model runner）仅需确保正确管理输入缓冲区。所有中间缓冲区均由编译器后端自动管理。

默认情况下，vLLM 会尝试确定一组大小来捕获 CUDA 图。您也可以使用配置 `cudagraph_capture_sizes` 来覆盖它：

```bash
vllm serve meta-llama/Llama-3.2-1B \
  --compilation-config '{"cudagraph_capture_sizes": [1, 2, 4, 8]}'
```

然后它将仅对指定的大小捕获 CUDA 图。这对于细粒度地控制 CUDA 图捕获非常有用。

### 完整 CUDA 图捕获

如果使用的注意力后端与 CUDA 图兼容，则可以将注意力作为 CUDA 图的一部分包含进去。这在某些情况下可以提高性能，例如小模型或 MoE 的解码速度。详情请参阅 [CUDA 图设计文档](cuda_graphs_zh.md)。
