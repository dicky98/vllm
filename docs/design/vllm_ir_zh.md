# vLLM IR：函数式中间表示 (vLLM IR: Functional Intermediate Representation)

## 动因 (Motivation)

vLLM IR 是一种**函数式中间表示 (IR)**，它填补了低级 `torch` 算子与 vLLM 高级层（如 `RMSNorm` 和量化算子）之间的空白。通过将算子的**语义 (Semantics)** 与其**实现 (Implementation)** 和**分发 (Dispatching)** 进行分离，vLLM IR 同时简化了编译和算子内核的注册与分发。它作为 torch FX 表示中的一个**方言 (Dialect)** 运行，允许与“常规”的 torch 算子及自定义 torch 算子/内核完全互操作，并支持从以前的 `CustomOp` 方法逐步迁移。

关键设计原则：

- **Eager 与编译模式一致性 (Eager-compile consistency)**：在 Eager 模式和编译模式下行为完全一致（极微小的数值差异除外）。
- **简单、透明且强大的内核选择 (Simple, transparent, yet powerful kernel selection)**：提供良好的可见性和控制力，便于调试。
- **约定优于配置 (Convention over configuration)**：注册算子和实现几乎不需要样板代码（Zero boilerplate）。
- **可扩展性 (Extensibility)**：算子和实现可以在任何地方注册，无论在树内（In-tree）还是树外（Out-of-tree）。
- **互操作性 (Interoperability)**：与“常规” torch 算子及自定义 torch 算子/内核完全兼容，减少开发摩擦并允许逐步迁移。

干净的语义与实现分离带来了一个统一且可扩展的分发机制，允许每个平台有多个内核，并提供强大的内核选择能力。这种分离还促进了更干净的测试和基准测试，去除了遗留方法中大部分的标准样板代码。

通过将内核选择延迟到编译过程的后期，编译器可以在更高级别的表示上运行，这具有以下主要好处：

- 在融合（Fusion）/转换（Transformation）Passes 中进行模式匹配时，每个算子只需要一个简单、唯一的模式（Pattern）。
- 树外（OOT）编译器后端可以直接从更高级别的表示进行 Lowering 转换（正在开发中）。
- 编译器可以对可用的实现进行自动调优（Autotune，未来特性）。

## 快速概述 (Quick Overview)

### 声明一个 IR 操作

使用带有原生 PyTorch 实现的 `@register_op` 装饰器来声明一个 IR 操作，该函数定义了该算子的语义：

```python
# vllm/ir/ops/layernorm.py
from torch import Tensor
from vllm.ir import register_op

@register_op
def rms_norm(x: Tensor, weight: Tensor | None, epsilon: float, variance_size: int | None = None) -> Tensor:
    """加权均方根层归一化 (Weighted root-mean-square layer normalization)"""
    orig_dtype = x.dtype
    x = x.to(torch.float32)
    x_var = x if variance_size is None else x[..., :variance_size]
    variance = x_var.pow(2).mean(dim=-1, keepdim=True)
    x = x * torch.rsqrt(variance + epsilon)
    x = x.to(orig_dtype)
    if weight is not None:
        x = x * weight
    return x
```

原生 PyTorch 实现有三个目的：

1. **语义定义**：指定操作的确切语义，包括张量形状（Shapes）和步长（Strides）。
2. **默认实现**：当没有其他（更好）的实现可用时使用。
3. **测试参考基准**：其他具体内核的实现必须与这些语义匹配。

### 注册具体实现

使用 IR 算子对象上的 `register_impl` 装饰器来注册内核实现：

```python
# vllm/kernels/vllm_c.py
from vllm import ir

rms_norm_no_var = lambda x, weight, epsilon, variance_size=None: variance_size is None

@ir.ops.rms_norm.register_impl("vllm_c", supports_args=rms_norm_no_var, supported=current_platform.is_cuda_alike())
def rms_norm(x: Tensor, weight: Tensor | None, epsilon: float, variance_size: int | None = None) -> Tensor:
    output = torch.empty_like(x)
    torch.ops._C.rms_norm(output, x, weight, epsilon)
    return output
```

实现可以指定：

- `supported`：静态布尔值，指示该实现是否可用。
- `supports_args`：一个函数，用于检查该实现是否支持特定的输入参数。
- `inplace`：该实现是否重用输入内存以用于输出。

### 在模型中使用 IR 操作

在模型代码中直接导入并调用 IR 操作：

```python
# vllm/model_executor/layers/layernorm.py
from vllm import ir

class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x: Tensor, residual: Tensor | None = None):
        if residual is None:
            return ir.ops.rms_norm(x, self.weight, self.variance_epsilon)

        # 使用 maybe_inplace 重载以允许实现重用输入内存用于输出
        # （在此调用之后使用 x 或 residual 将是未定义行为）
        return ir.ops.fused_add_rms_norm.maybe_inplace(
            x, residual, self.weight, self.variance_epsilon
        )
```

### 配置内核选择 (Configuring Kernel Selection)

内核选择通过配置中的优先级列表（Priority lists）进行控制。优先级列表指定了考量实现的顺序，系统会选择列表里第一个受支持的实现。这包含了静态支持检查（`supported=...`）和动态参数支持检查（`supports_args=...`）。

#### 命令行配置

使用 `--ir-op-priority.<op_name>=<provider1>,<provider2>,...`：

```bash
# CUDA：为 rms_norm 使用 vllm_c 实现
vllm serve meta-llama/Llama-3.2-1B \
  --ir-op-priority.rms_norm=vllm_c

# ROCm：首先尝试 aiter，回退到 vllm_c，最后使用 native
vllm serve meta-llama/Llama-3.2-1B \
  --ir-op-priority.rms_norm=aiter,vllm_c,native

# 配置多个操作
vllm serve meta-llama/Llama-3.2-1B \
  --ir-op-priority.rms_norm=vllm_c \
  --ir-op-priority.fused_add_rms_norm=vllm_c
```

#### Python 代码中配置

```python
from vllm import LLM
from vllm.config import VllmConfig, KernelConfig

llm = LLM(
    model="meta-llama/Llama-3.2-1B",
    vllm_config=VllmConfig(
        kernel_config=KernelConfig(
            ir_op_priority={
                "rms_norm": ["vllm_c", "native"],
                "fused_add_rms_norm": ["vllm_c", "native"],
            }
        )
    )
)
```

#### 平台默认配置

每个平台都提供了自动应用的默认优先级列表：

```python
# CUDA/XPU/ROCm 平台默认配置（使用 Inductor 编译时）
{
  "rms_norm": ["native"],  # 默认使用原生 torch
  "fused_add_rms_norm": ["native"],
}

# CUDA 平台默认配置（Eager 或仅限 Dynamo 模式）
{
  "rms_norm": ["vllm_c", "native"],
  "fused_add_rms_norm": ["vllm_c", "native"],
}

# ROCm 平台默认配置（未来规划 - 目前与 CUDA 相同）
{
    "rms_norm": ["aiter", "vllm_c", "native"],
    "fused_add_rms_norm": ["aiter", "vllm_c", "native"],
}

# XPU 平台默认配置（Eager 或仅限 Dynamo 模式）
{
    "rms_norm": ["xpu_kernels", "native"],
    "fused_add_rms_norm": ["xpu_kernels", "native"],
}
```

用户指定的优先级会前置于平台默认配置中，因此您只需指定顺序不同的实现，其他实现会自动追加到末尾。

## 编译流水线 (Compilation Pipeline)

vLLM IR 深度定制了基于 `torch.compile` 的编译流程，以允许自定义的编译 Passes 在高级 IR 上运行，同时在最后仍能生成高效的底层代码。该编译流水线由几个阶段组成：

### 1. Dynamo 追踪 (Dynamo Tracing)

当 `torch.compile` 追踪模型的 `forward` 方法时，vLLM IR 操作会作为自定义操作出现在 `vllm_ir` torch 库中。这些操作对 Dynamo 来说是不透明的，这意味着它们会直接出现在 FX 图中而不进行分解（Decomposition）：

```python
# Python 代码 (epsilon=1e-5)
x1 = ir.ops.rms_norm(x, weight, epsilon)
x2, residual_out = ir.ops.fused_add_rms_norm.maybe_inplace(x1, residual, weight, epsilon)

# Dynamo 追踪后的 FX 图
x1 = torch.ops.vllm_ir.rms_norm.default(x, weight, 1e-5); x = None
out = torch.ops.vllm_ir.fused_add_rms_norm.maybe_inplace(x1, residual, weight, 1e-5); x1 = residual = None
x2 = out[0]
residual_out = out[1]
```

### 2. AOTAutograd 与函数化 (AOTAutograd and Functionalization)

AOTAutograd 会将图函数化（Functionalizes），将任何带有就地修改（Mutation）的操作转换为等价的纯函数。对于带有 `maybe_inplace` 重载的 vLLM IR 操作，我们在 AOTAutograd 之前手动执行此操作，在 pre-grad 自定义 Pass 钩子中，使用 `default` 重载将它们转换为无损的函数形式。

```python
# 函数化之后
x1 = torch.ops.vllm_ir.rms_norm.default(x, weight, 1e-5); x = None
out = torch.ops.vllm_ir.fused_add_rms_norm.default(x1, residual, weight, 1e-5); x1 = residual = None
x2 = out[0]
residual_out = out[1]
```

该 Pass 还会追踪哪些输入被“捐赠”（Donated，即被传递给 `maybe_inplace`），将此信息存储在 vLLM 的 `PassContext` 中，供稍后克隆消除 Pass 使用。

### 3. IR 融合与转换 Passes (IR Fusion and Transformation Passes)

在函数化之后，自定义的 vLLM Passes 对包含高级 IR 操作的函数式 FX 图进行操作。这些 Passes 可以执行算子融合、将操作分发用于序列并行等各种转换：

```python
# 示例：序列并行 (参见 SequenceParallelismPass)
# SP Pass 执行前
all_reduce = torch.ops.vllm.all_reduce(x, "tp:0")
rms_norm = torch.ops.vllm_ir.rms_norm(all_reduce, weight, 1e-5)

# SP Pass 执行后
reduce_scatter = torch.ops.vllm.reduce_scatter(x, "tp:0")
rms_norm = torch.ops.vllm_ir.rms_norm(all_reduce, weight, 1e-5)
all_gather = torch.ops.vllm.all_gather(x, "tp:0")
```

融合 Passes 受益于这种高级表示：它们不需要与底层 PyTorch 操作进行匹配，不需要单独处理不同的内核实现，也不需要处理自定义内核的函数化。

### 4. IR Lowering 转换 (IR Lowering)

Lowering Pass (`VllmIRLoweringPass`) 使用选择的具体内核实现来替换每个 vLLM IR 操作。实现的选择基于优先级列表和支持谓词（Support predicates），在图的元数据中使用**虚拟张量 (Fake tensors)** 代替实际的算子参数：

```python
# 实现选择逻辑，在 Eager 分发和编译 Lowering 中相同
def dispatch(*args) -> IrOpImpl:
  for provider in priority_list:  # 例如 ["vllm_c", "native"]
    impl = ir_op.impls[provider]
    if not impl.supported:
      continue
    if impl.supports_args and not impl.supports_args(*args):
      continue
    return impl

# make_fx 使用 torch.fx.symbolic_trace
impl_graph = make_fx(selected_impl.impl_fn)
# 用 impl_graph 的节点替换 IR op 节点
match.replace_by_example(selected_impl.impl_fn, node.args)
```

例如，使用 `vllm_c` 实现来 Lowering 转换 `rms_norm`：

```python
# Lowering 转换前 (IR 算子)
rms_norm = torch.ops.vllm_ir.rms_norm.default(x, weight, 1e-5)

# Lowering 转换后 (追踪 vllm_c 实现)
# 注意：Lowering 目前不进行函数化，未来这可能会改变。
empty =  torch.ops.aten.empty.memory_format(x.shape, ...)
rms_norm = torch.ops._C.rms_norm(empty, x, weight, 1e-5)
```

在 Lowering 转换一个会修改输入的实现（`inplace=True`）时，Lowering Pass 会插入克隆（Clones）以保持函数语义：

```python
# 用于 fused_add_rms_norm 的 vllm_c 实现会修改其前两个参数
# 为了安全起见，在此插入 clones 进行 Lowering
clone_default = torch.ops.aten.clone.default(x)
clone_default_1 = torch.ops.aten.clone.default(residual)
fused_add_rms_norm = torch.ops._C.fused_add_rms_norm.default(clone_default, clone_default_1, weight, 1e-5)
```

### 5. 克隆清理 (Clone Cleanup)

在 Lowering 转换之后，克隆消除 Pass (`UnsafeCloneEliminationPass`) 会移除在 Lowering 过程中引入的不必要克隆。当搭配 `maybe_inplace` 使用就地（In-place）内核时，这个 Pass 对于实现零拷贝（Zero-copy）行为至关重要。

当满足以下条件时，该 Pass 将移除克隆操作：
- 克隆的输入是在图内部创建的，且在图中此后没有被再次使用。
- 克隆的输入是一个图参数，且被标记为了已捐赠（Donated）。

```python
# 清理之后 (已捐赠的输入，此后无任何使用)
fused_add_rms_norm = torch.ops._C.fused_add_rms_norm.default(x, residual, weight, 1e-5)
```

就地函数化（追踪捐赠的输入）与克隆清理的结合，使得编译器能够安全地使用就地内核，而不会增加冗余拷贝或增加内存开销。

### 6. Inductor 优化与代码生成 (Inductor Optimization and Codegen)

在 IR Lowering 和清理之后，图只包含标准的 PyTorch 操作和特定于平台的自定义算子。随后 Inductor 执行其标准代码生成：

- **Inductor Lowering 和逐点融合**：融合逐元素（Element-wise）操作、Reduction 规约操作等。
- **内存规划**：确定缓冲区的分配和复用。
- **代码生成**：为融合操作生成 Triton 或 C++ 代码。
- **自动调优 (Autotuning)**：选择最佳的算子内核配置。

### 流水线总结 (Pipeline Summary)

```text
模型前向传播 (Model Forward Pass)
    ↓
[Dynamo 追踪] → 带有 vllm_ir.* 算子的 FX 图
    ↓
[Pre-grad: 就地函数化] → maybe_inplace 转换为 default, 并追踪被捐赠的输入
    ↓
[AOTAutograd] → 函数化
    ↓
[Post-grad: IR 融合 Passes] → 融合高级 IR 算子 (例如 rms_norm + quant)
    ↓
[Post-grad: IR Lowering] → vllm_ir.* 算子 转换为 impl 算子 (必要时带 clones)
    ↓
[Post-grad: 克隆清理] → 利用捐赠输入信息，移除不必要的 clones
    ↓
[Inductor] → 模式匹配、融合、内存规划、代码生成
    ↓
已编译好的代码 (Compiled Code)
```

## 核心 vLLM IR 概念

### 算子声明 (Operation Declaration)

操作使用 `@register_op` 装饰器进行声明，这会创建一个 `IrOp` 对象：

```python
@register_op(
    name=None,           # 操作名称 (默认使用函数名)
    activations=None,    # 激活参数列表 (默认使用以 'x' 开头的参数)
    allow_inplace=False, # 是否创建 maybe_inplace 重载
)
def op_name(...):
    ...
```

**参数说明：**

- `activations`：被认为是“激活值（Activations）”的参数名列表（通常由 `maybe_inplace` 消耗）。默认使用以 `x` 开头的参数。
- `allow_inplace`：创建一个 `maybe_inplace` 重载，以便进行内存高效的执行（见下文）。

### `maybe_inplace` 重载

`maybe_inplace` 重载是 LLM 推理中提升内存效率的关键特性。它向调用方发出信号：调用方在执行此操作后无需保留传入的激活值输入，从而允许就地实现直接重用输入内存作为输出。

#### 语义与用法

```python
# 标准用法：保留输入内容
out, res_out = ir.ops.fused_add_rms_norm(x, residual, weight, epsilon)
# x 和 residual 保持不变，out 和 res_out 是新的张量

# maybe_inplace：输入内容可能被修改
out, res_out = ir.ops.fused_add_rms_norm.maybe_inplace(x, residual, weight, epsilon)
# x 和 residual 可能会被修改 (在此之后使用它们是未定义行为)
# out 和 res_out 可能会与 x 和 residual 共享内存别名
```

在将输入传递给 `maybe_inplace` 之后继续使用它属于**未定义行为**：

```python
# 错误写法：在捐赠了 x 之后继续使用它
out, res_out = ir.ops.fused_add_rms_norm.maybe_inplace(x, residual, weight, epsilon)
result = out + x  # 错误：x 已经被捐赠了！
```

如果需要保留输入，可以使用默认重载或手动克隆：

```python
# 方案 1：使用默认重载
out, res_out = ir.ops.fused_add_rms_norm(x, residual, weight, epsilon)
result = out + x  # 正确：x 被保留下来

# 方案 2：在 maybe_inplace 之前进行克隆
out, res_out = ir.ops.fused_add_rms_norm.maybe_inplace(x.clone(), residual, weight, epsilon)
result = out + x  # 正确：x 保持完好，捐赠的是克隆出来的张量
```

#### 编译行为

在编译期间，就地函数化 Pass 会校验已捐赠的输入没有在后文被再次使用，并将 `maybe_inplace` 转换为无损的 `default` 重载：

```python
# 就地函数化 Pass (pre-grad)
for node in graph.nodes:
    if node.target == torch.ops.vllm_ir.fused_add_rms_norm.maybe_inplace:
        # 校验在该节点之后，激活值输入没有被使用
        for activation_arg in activation_inputs:
            for user in activation_arg.users:
                if user appears after node:
                    raise ValueError(f"Input {activation_arg} donated but used again")

        # 转换为 default 重载
        node.target = torch.ops.vllm_ir.fused_add_rms_norm.default

        # 追踪捐赠的图输入，用于后续的克隆消除
        for i, arg in enumerate(node.args):
            if arg.op == "placeholder" and i in activation_indices:
                pass_context.donated_input_ids.add(node_to_idx[arg])
```

随后，捐赠的输入信息将被克隆清理 Pass 采用，以便在 Lowering 转换就地内核时消除多余的拷贝。

#### Eager 模式行为

在 Eager 模式下（不使用 `torch.compile`），`maybe_inplace` 通过允许 IR 操作直接分发到就地实现，来实现**最大内存效率**的执行：

```python
# 用于 maybe_inplace 的 Eager 分发逻辑
impl: IrOpImpl = ir_op.dispatch(*args)
return impl.impl_fn(*args)

# 用于 default 的 Eager 分发逻辑：
impl: IrOpImpl = ir_op.dispatch(*args)
if impl.inplace:
  args = [
    arg.clone() if i in ir_op.activations else arg
    for i, arg in enumerate(args)
  ]
return impl.impl_fn(*args)
```

在模型代码中使用 `maybe_inplace` 并结合就地内核实现，在 Eager 模式和编译模式下均能提供最佳的内存效率，且在两种模式下语义完全一致。

#### 节省内存示例

考虑一个带有残差连接的 Transformer 层：

```python
# 不使用 maybe_inplace (每层有 2 次分配)
hidden_states = self.attention(input)
normed, residual = ir.ops.fused_add_rms_norm(hidden_states, input, weight, eps)
# 内存情况：input (保留), hidden_states (保留), normed (新分配), residual (新分配)

# 使用 maybe_inplace (使用就地内核时，每层 0 次内存分配)
hidden_states = self.attention(input)
normed, residual = ir.ops.fused_add_rms_norm.maybe_inplace(hidden_states, input, weight, eps)
# 内存情况：normed (重用 hidden_states 的内存), residual (重用 input 的内存)
```

### 具体实现注册 (Implementation Registration)

具体实现使用 `register_impl` 方法进行注册：

```python
@ir.ops.op_name.register_impl(
    provider="provider_name",  # 唯一标识符 (例如 "vllm_c", "aiter", "triton")
    supported=True,            # 静态可用性检查
    supports_args=None,        # 动态参数支持检查
)
def impl_fn(...):
    ...
```

**提供商 (Provider) 命名约定：**

- `native`：保留给原生 torch 实现（通过 `@register_op` 声明）。
- `vllm_c`：通过 `torch.ops._C` 调用的 C++/CUDA 内核。
- `aiter`：AMD AITER 库。
- `xpu_kernels`：在 `vllm-xpu-kernels` 中实现的 SYCL/SYCLTLA 内核。
- `triton_*`：Triton 内核。
- 其他实现对应的平台/库名称。

**支持性校验：**

- `supported`：静态布尔值，在导入时检查一次（例如 `HAS_TRITON`, `is_cuda_alike()`）。
- `supports_args`：一个形如 `(*args, **kwargs) -> bool` 的函数，用于检查参数兼容性。
    - 在编译期间使用**虚拟张量 (Fake tensors)** 进行调用，实现零开销检查。
    - 在 Eager 模式分发期间使用**真实张量 (Real tensors)** 进行调用。
    - 不应该检查 Batch Size 或根据值添加 Guards。

示例支持谓词（Predicate）：

```python
def aiter_rms_norm_supports(x, weight, epsilon, variance_size=None):
    # 检查 dtype (合理：不依赖于 batch size)
    if x.dtype not in [torch.float16, torch.bfloat16]:
        return False
    # 检查可选参数 (合理：静态检查)
    if variance_size is not None:
        return False
    return True

@ir.ops.rms_norm.register_impl("aiter", supports_args=aiter_rms_norm_supports)
def rms_norm(...):
    ...
```

当设置了环境变量 `VLLM_BATCH_INVARIANT=1` 时，会自动选择 Batch-invariant 的内核。

### Eager 模式 vs 编译模式

vLLM IR 操作在 Eager 模式和编译模式下的行为完全一致：

**Eager 模式：**
- 基于优先级列表直接分发到对应实现。
- 使用真实张量参数检查支持性。
- 引入极低的开销（如果需要，还可以进一步优化）。

**编译模式：**
- IR 算子在 FX 图中作为 `torch.ops.vllm_ir.*` 自定义算子出现。
- Lowering 阶段使用虚拟张量来选择实现。
- 与 Inductor 优化完全集成。

这种一致性使得：
- 可以充满信心地在 Eager 模式下进行原型设计。
- 可以通过禁用编译来进行调试。
- 可以从 Eager 执行平滑且逐步地迁移到编译执行。

## 其他主题 (Other Topics)

### 树外 (OOT) 实现 (Out-of-Tree Implementations)

外部平台无需修改 vLLM 代码即可注册具体实现：

```python
# 在外部包中
from vllm import ir

@ir.ops.rms_norm.register_impl("my_platform", supported=is_my_platform())
def rms_norm(x, weight, epsilon, variance_size=None):
    return my_platform.rms_norm(x, weight, epsilon)
```

然后配置优先级以使用您的实现：

```python
class MyPlatform(Platform):
  def get_default_ir_op_priority(self):
    return IrOpPriorityConfig(rms_norm=['my_platform', 'native'])

# 用户仍可以以相同方式覆盖优先级
llm = LLM(ir_op_priority=IrOpPriorityConfig(rms_norm=['custom_oot_kernel']))
```

### 调试与可观测性 (Debugging and Observability)

!!! note "注意"
    请随时向我们反馈如何针对您的使用场景改进可观测性！

启用 Debug 级别日志以查看内核选择过程：

```bash
VLLM_LOGGING_LEVEL=DEBUG vllm serve ...
```

这将记录：
- 每个操作选择了哪些实现。
- 为什么实现被拒绝（不支持、参数不支持等）。
- 编译缓存的命中/未命中情况。
- IR Lowering 统计数据。

在已编译的图中检查选定的实现：

```python
# 编译完成后，检查 Lowering Pass
lowering_pass = backend.lowering_pass
print(lowering_pass.selected_impls)
# 输出示例: {'rms_norm': {'node_123': 'vllm_c', 'node_456': 'vllm_c'}}
```

## 从 CustomOp 迁移

vLLM IR 被设计为与 `CustomOp` 共存并逐步取代它：

1. **算子声明**：转换 `CustomOp` 类 `PluggableLayer` 并将 `forward_native` 移至 `@register_op` 函数。
2. **实现注册**：使用 `@ir.ops.op_name.register_impl` 代替重写方法。
3. **层的使用**：用 `ir.ops.op_name(...)` 替换 `self.op(...)`。
4. **配置**：将 `--compilation-config.custom-ops` 迁移到 `--ir-op-priority`。

迁移可以增量进行，一次只迁移一个操作。

## 参见 (See Also)

- [torch.compile 集成](torch_compile_zh.md) — 通用编译基础设施
- [算子融合](fusions_zh.md) — vLLM 中的自定义融合和转换 Passes
- [自定义算子](custom_op_zh.md) — 遗留的自定义算子系统
