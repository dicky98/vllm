# Logits 处理器 (Logits Processors)

!!! important "重要"
    部分 Logits 处理器的设计更改仍在进行中，其 API 在不久的将来可能会发生变化。我们希望尽快稳定这部分 API。

本文档介绍了 vLLM 引擎如何与 Logits 处理器交互，以及 vLLM 所支持的用于实现 Logits 处理器的编程模型。

## 背景介绍 (Logits Processors Background)

Logits 处理器用于调整下一个 Token 的概率分布，通常目的是引导模型朝着期望的行为方式发展。

在 vLLM 中，Logits 处理器在批次（Batch）粒度下运行。在给定的引擎步骤中，Logits 处理器消耗一个由模型输出的 `(num_requests) x (vocab_size)` 的原始 Logits 张量。对于启用了该 Logits 处理器的所有请求，Logits 处理器将对 Logits 张量的对应行应用转换，而保持其他行未被修改。转换后的 Logits 张量随后被传递给 Softmax。

## vLLM 引擎中的 Logits 处理器

vLLM 引擎的常驻批次（Persistent batch）数据结构中维护了已加载的 Logits 处理器列表。

为了能够同时处理整个批次，每个 Logits 处理器都可以维护关于批次中请求的元数据（即每个请求特定的 Logits 处理器配置设置）。因此，Logits 处理器是有状态（Stateful）的。

在每个引擎步骤中，vLLM 引擎会：（1）更新每个 Logits 处理器的内部状态，以及（2）将 Logits 处理器应用到模型输出的 Logits 上。

### 更新 Logits 处理器内部状态

在每个引擎步骤开始时，常驻批次可能会根据调度器（Scheduler）的输出来添加、丢弃和/或重新排序请求。在常驻批次重组之后，vLLM 引擎会调用每个 Logits 处理器的 `update_state()` 方法。这对于确保 Logits 处理器的内部状态在引擎步骤开始时重组，以匹配新的常驻批次状态是必需的。

下面的伪代码展示了 vLLM 常驻批次向每个 Logits 处理器通知批次状态发生变化的流程：

??? code "Model Runner 更新 Logits 处理器状态"

    ``` python
    # gpu_model_runner.py

    class GPUModelRunner(...):

        ...

        def execute_model(self, scheduler_output, ...):
            self._update_states(scheduler_output)

            ...

        def _update_states(...):

            ...

            # ...更新常驻批次以反映新请求/已完成请求以及批次内请求的重新排序...

            ...

            self.input_batch.refresh_metadata()


    # gpu_input_batch.py

    class InputBatch:

        ...

        def refresh_metadata(self):

            ...

            # 更新每个 Logits 处理器的状态以反映常驻批次状态
            batch_update = self.batch_update_builder.get_and_reset(self.num_reqs)
            for logit_proc in self.logitsprocs.all:
                logit_proc.update_state(batch_update)

            ...


    # vllm/v1/sample/logits_processor/interface.py

    @dataclass(frozen=True)
    class BatchUpdate:
        # 传递给 Logits 处理器 update_state() 方法的批次状态变更数据结构
        batch_size: int

        removed: Sequence[RemovedRequest]
        added: Sequence[AddedRequest]
        moved: Sequence[MovedRequest]
    ```

### 将 Logits 处理器应用到模型输出的 Logits

在更新常驻批次状态后，vLLM Model Runner 进行模型推理以获取 Logits。然后，Model Runner 对 Logits 调用采样器（Sampler）。而采样器操作的一部分，就是对模型输出的 Logits 调用 Logits 处理器的 `apply()` 方法，从而产生转换后的 Logits（`apply()` 方法可以就地 in-place 或离地 out-of-place 修改 Logits，但就地修改内存效率更高）。此过程如下伪代码所示：

请注意，采样器将通过 `SamplingMetadata.logitsprocs` 访问 Logits 处理器。当 vLLM 引擎构建 `SamplingMetadata` 时（下文未示出），对 Logits 处理器列表的引用会从常驻批次数据结构传递到 `SamplingMetadata` 中。

??? code "将 Logits 处理器应用到模型输出的 Logits"

    ``` python
    # gpu_model_runner.py

    class GPUModelRunner(...):

        ...

        def execute_model(self, scheduler_output, ...):
            # （在上一节中讨论过）
            self._update_states(scheduler_output)

            ...

            # ...运行模型推理以获取 Logits...

            ...

            # 调用采样器，采样器内部会应用 Logits 处理器
            sampler_output = self.sampler(logits=logits,
                                          sampling_metadata=sampling_metadata)

            ...


    # sampler.py

    class Sampler(nn.Module):

        ...

        def forward(self, logits, sampling_metadata):

            ...

            # 将非 Argmax 不变的 Logits 处理器应用到模型输出的 Logits
            for processor in (sampling_metadata.logitsprocs.non_argmax_invariant):
                logits = processor.apply(logits)

            sampled = self.sample(logits, sampling_metadata)

            ...

            # ...返回采样器输出的数据结构...


        def sample(self, logits, sampling_metadata)

            ...

            # ...如果所有请求都是贪婪采样，则提前退出...

            ...

            # 应用 Argmax 不变的 Logits 处理器
            for processor in sampling_metadata.logitsprocs.argmax_invariant:
                logits = processor.apply(logits)

            ...

            # ...执行采样并返回采样结果...
    ```

在采样时，采样器会检查常驻批次中的所有请求是否均采用贪婪采样。如果是这样，采样器会通过跳过“Argmax 不变（argmax-invariant）”的 Logits 处理器来节省计算开销。在这里，“argmax” 是给定行中具有最高 Logit 值的 Token ID（即模型对给定请求赋予最高权重的 Token）的缩写。

* **Argmax 不变的 Logits 处理器** 是一种不会修改 Argmax 的 Logits 处理器（例如 Min-P）。例如，屏蔽掉概率最低的 Token 的 Logits 处理器不会改变哪个 Token ID 具有最大 Logit 值。贪婪采样总是挑选 Logit 值最高的 Token ID，因此在概念上，对于贪婪采样请求，可以跳过 Argmax 不变的 Logits 处理器。

* **非 Argmax 不变的 Logits 处理器** 是一种可能会修改 Argmax 的 Logits 处理器。例如，为了强制终止解码，在一定步数后屏蔽除 EOS 之外的所有 Token 的 Logits 处理器可能会屏蔽掉最大 Logit 值的 Token，从而改变 Argmax。在概念上，对于贪婪采样请求，不能跳过这些 Logits 处理器。

vLLM Logits 处理器抽象要求引擎在批次粒度下应用 Logits 处理器；因此在实践中，只有当整个批次都使用贪婪采样时，才能跳过 Argmax 不变的 Logits 处理器。

## Logits 处理器编程模型 (Logits Processor Programming Model)

前几节暗示了 vLLM Logits 处理器必须支持的接口。本节将全面介绍用于实现与 vLLM 引擎兼容的 Logits 处理器的编程模型，包括 `LogitsProcessor` 基类及其接口方法，以及表示常驻批次状态变更的 `BatchUpdate` 数据结构，两者均如下代码所示：

??? code "`LogitsProcessor` 基类和 `BatchUpdate` 数据结构"

    ``` python
    from abc import ABC, abstractmethod
    from collections.abc import Sequence
    from dataclasses import dataclass
    from enum import Enum, auto
    from typing import TYPE_CHECKING

    import torch

    from vllm import SamplingParams

    if TYPE_CHECKING:
        from vllm.config import VllmConfig


    class MoveDirectionality(Enum):
        # 批次内请求的单向 i1->i2 移动
        UNIDIRECTIONAL = auto()
        # 批次内请求的双向 i1<->i2 交换
        SWAP = auto()


    # 添加到批次中的新请求的 (index, params, prompt_tok_ids, output_tok_ids) 元组
    AddedRequest = tuple[int, SamplingParams, list[int], list[int]]

    # 代表批次内请求单向移动或双向交换的 (index 1, index 2, directionality) 元组
    MovedRequest = tuple[int, int, MoveDirectionality]

    # 被移除请求的批次索引列表
    RemovedRequest = int


    @dataclass(frozen=True)
    class BatchUpdate:
        """Logitsprocs 所需的常驻批次状态变更信息"""
        batch_size: int  # 当前批次中的请求数量

        # 对添加到常驻批次、从常驻批次移除以及在常驻批次中移动的请求的元数据。
        #
        # 关键假设：`added` 中每个元组的 `output_tok_ids` 列表元素是对该请求正在运行的输出 Token 列表的引用；
        # 通过此引用，Logits 处理器始终能看到最新生成的输出 Token 列表。
        removed: Sequence[RemovedRequest]
        moved: Sequence[MovedRequest]
        added: Sequence[AddedRequest]


    class LogitsProcessor(ABC):

        @abstractmethod
        def __init__(self, vllm_config: "VllmConfig", device: torch.device,
                    is_pin_memory: bool) -> None:
            raise NotImplementedError

        @abstractmethod
        def apply(self, logits: torch.Tensor) -> torch.Tensor:
            raise NotImplementedError

        @abstractmethod
        def is_argmax_invariant(self) -> bool:
            """如果 Logits 处理器对贪婪采样中的 Argmax 计算没有影响，则返回 True。
            注意：基于子类的实现，这在给定 LogitsProcessor 子类的不同实例中可能有相同或不同的值。
            """
            raise NotImplementedError

        @abstractmethod
        def update_state(
            self,
            batch_update: "BatchUpdate" | None,
        ) -> None:
            """在每次前向传播之前、新输出 Token 产生时被调用。

            参数：
                仅当批次构成发生变化时，batch_update 才为非 None。
            """
            raise NotImplementedError

        @classmethod
        def validate_params(cls, sampling_params: SamplingParams):
            """校验此 Logits 处理器的采样参数。

            若参数无效则抛出 ValueError。
            """
            return None
    ```

vLLM Logits 处理器必须继承 `LogitsProcessor` 并定义（至少）以下方法：

* `__init__(self, vllm_config: VllmConfig, device: torch.device, is_pin_memory: bool)`：
    * `vllm_config`：引擎配置数据结构。
    * `device`：硬件加速器设备信息。
    * `is_pin_memory`：指示固定内存（Pin memory）是否可用以支持 Logits 处理器实现的 Flag。

* `apply(self, logits: torch.Tensor) -> torch.Tensor`：
    * 消耗一个 `(num_requests) x (vocab_size)` 的 Logits 张量 (`logits`)。
    * 在批次粒度下应用 Logits 处理器转换。
    * 返回一个转换后的 `(num_requests) x (vocab_size)` Logits 张量。
    * 您可以就地（in-place）或离地（out-of-place）修改输入的 Logits；就地修改更加节省内存。

* `is_argmax_invariant(self) -> bool`：
    * 如果 Logits 处理器是 Argmax 不变的（绝不改变给定请求具有最高 Logit 值的 Token ID），返回 `True`；如果 Logits 处理器可能会修改 Argmax，返回 `False`。
    * `is_argmax_invariant()` 在启动时被评估一次；如果为 `True`，当所有请求均使用贪婪采样时，vLLM 将在给定的步骤中跳过应用此 Logits 处理器。

* `update_state(self, batch_update: "BatchUpdate" | None) -> None`：
    * 在当前引擎步骤开始时，消耗一个表示常驻批次状态变化的 `BatchUpdate` 数据结构。
    * 使用 `BatchUpdate` 成员来更新 Logits 处理器的内部状态。
    * **注意**：批次更新数据结构可能为 `None`，表示批次构成没有变化。在这种情况下，Logits 处理器可能仍需要根据它在添加时保留的已更新 `output_token_ids` 列表来更新其状态。

* `validate_params(cls, sampling_params: SamplingParams)`：
    * 如果 `SamplingParams` 包含了该 Logits 处理器使用的无效参数（尤其是自定义参数），则抛出 `ValueError`。
    * 当请求发送到入口点时，`validate_params()` 将校验 `SamplingParams` 并拒绝带有无效参数的请求。

### `BatchUpdate` 数据结构详解

`BatchUpdate` 抽象将常驻批次建模为请求列表，支持以下更改批次状态的操作（注意，下面提到这些操作的顺序反映了它们在 `update_state()` 中应该被处理的顺序）：

* **Remove (移除)**：移除（不替换）索引为 `i` 的请求。
    * 移除在 `BatchUpdate.removed` 中由一个 `int`（代表 `i`）表示。
    * 移出操作对批次的影响：
        ``` text
        批次: [A, B, C]
        在 i=1 处移除:
        =>
        新批次: [A, x, C] # 丢弃 B 并留出一个空槽
        ```

* **Add (添加)**：在索引 `i` 处添加（或用其替换现有请求）一个新请求。如果替换了某个请求，其关联的状态应该被丢弃。
    * 添加在 `BatchUpdate.added` 中表示为以下形式的元组：
        ``` text
        (索引, 新请求的 SamplingParams, prompt token ID 列表引用, output token ID 列表引用)
        ```
    * `prompt token IDs` 和 `output token IDs` 分别是对请求的 Prompt Token ID 列表和输出 Token ID 列表的引用。注意，输出 Token ID 列表会随着每个引擎步骤而增长，并且这种增长对 Logits 处理器是可见的，因为输出 Token ID 是通过引用传递的。**这对于考量目前为止已生成 Token 的 Logits 处理器至关重要**。
    * 具体 Logits 处理器子类的实现决定了添加的请求元组中的字段如何消化到内部表示中。例如，不使用 Prompt 或输出 Token ID 的 Logits 处理器可能仅需利用 `index` 和 `SamplingParams`，而丢弃其他元组字段。
    * 如果索引 `i` 处当前持有请求，则发生替换：
        ``` text
        批次: [A, B, C]
        要在 i=1 处添加新请求: D
        =>
        新批次: [A, D, C] # 添加 D，丢弃 B
        ```
    * 如果索引 `i` 处当前不持有请求（因为 `i` 超出了当前批次大小的范围）：
        ``` text
        批次: [A, B, C]
        要在 i=3 处添加新请求: D
        =>
        新批次: [A, B, C, D] # 添加 D，扩展批次
        ```

* **Move (移动/交换)**：将索引 `s` 处的请求移动到索引 `d` 处，或者交换索引 `s` 和 `d` 处的请求。
    * 移动在 `BatchUpdate.moved` 中表示为以下形式的元组：
        ``` text
        (s, d, UNIDIRECTIONAL 或 SWAP)
        ```
    * 如果 Move 指定为 `UNIDIRECTIONAL` (单向移动)：
        * 索引 `s` 处的请求被移到索引 `d` 处；索引 `s` 变成空槽。
            ``` text
            批次: [A, x, C, D]
            单向移动 s -> d: 3 -> 1
            =>
            新批次: [A, D, C, x] # 将 D 移到 1，在 3 处留下空槽
            ```
        * 如果索引 `d` 处已经驻留了另一个请求，它将被替换并丢弃：
            ``` text
            批次: [A, B, C, D]
            单向移动 s -> d: 3 -> 1
            =>
            新批次: [A, D, C, x] # 将 D 移到 1，丢弃 B 并在 3 处留下空槽
            ```
    * 如果 Move 指定为 `SWAP` (双向交换)，则 `s` 和 `d` 处的请求交换索引位置：
        ``` text
        批次: [A, B, C, D]
        交换移动 s <-> d: 3 <-> 1
        =>
        新批次: [A, D, C, B] # 交换 B 和 D
        ```

此外，`BatchUpdate` 数据结构包含一个在引擎步骤开始时对常驻批次大小的表示（`batch_size`）。

### vLLM 引擎如何构建 `BatchUpdate` 数据结构

Logits 处理器 `update_state()` 的实现应该假定 Model Runner 以下述模式更新常驻批次状态（以 `BatchUpdate` 抽象来表达）：

1. 识别在当前引擎步骤中已完成的请求的索引。
2. 识别在当前步骤中引入的新请求。
3. 使用 Add 操作将新请求替换已完成的请求，按照被替换请求的索引递增的顺序（从最低索引开始）进行。
4. 根据新请求和已完成请求的相对数量：
    1. 如果新请求和已完成请求的数量相同，则转到下一步。
    2. *如果新请求多于已完成请求*：应用 Add 操作将未替换已完成请求的剩余新请求追加到批次末尾。为这些新请求分配连续的索引，从 `current_max_batch_index + 1` 开始。
    3. *如果新请求少于已完成请求*：
        * 对未被新请求替换的已完成请求应用 Remove 操作。这些移除的请求索引必然大于前一步中被替换的已完成请求的最大索引。移除操作可能会使批次处于非连续状态。
        * **“压缩” 批次以使其连续**：从索引最低的空槽（由 Remove 操作引起）开始，应用单向移动（Unidirectional Move）操作，将当前批次中最高非空槽的内容移来填满该空槽。依次以目标空槽索引递增、源非空槽索引递减的顺序进行额外的单向移动操作，直到批次连续为止。
        * **收缩批次**：压缩批次的一个副作用是，由 Remove 操作产生的空槽被归拢在批次数组末尾的一个连续块中。因此，在压缩后，更新 `BatchUpdate.batch_size` 以反映非空槽的数量。
5. 为了提高效率，对批次进行重新排序。基于注意力后端的实现和当前批次的特征，可能会应用零个或多个交换移动（Swap Move）操作来重新排序批次。

补充说明：

* Logits 处理器的 `update_state()` 方法必须按照以下顺序处理批次更新操作：removes（移除）、adds（添加）、moves（移动/交换）。
* Add 操作的索引参数是指**发生添加操作时的索引**，即在任何 Move 操作执行之前。
    * 例如：如果一个请求在索引 5 处被添加，随后与索引 3 进行了交换，则 `BatchUpdate.added` 中的添加操作仍与索引 5 相关联，而不是索引 3。
    * 换句话说，可以假定 Move 操作是在 Adds 和 Removes 之后应用的。
* 可以假定 Move 操作按照它们在 `BatchUpdate.moved` 中出现的顺序依次应用。
* 如果没有新/已完成的请求且没有批次重排，则 Logits 处理器的批次更新将为 `None`。

#### 示例 1：新请求少于已完成请求时的批次更新

以下示例模拟了一个引擎步骤：引入了 1 个新请求，消除了 2 个已完成的请求，此外注意力后端执行了一次交换以优化批次顺序。

``` text
批次状态 (引擎步骤开始时): [A, B, C, D]
批次大小: 4

新请求: E

已完成的请求: A, C

处理步骤 (使用 BatchUpdate 抽象):

1. 在索引 0 处添加 E
[E, B, C, D] # 丢弃 A
批次大小: 4

2. 在索引 2 处移除
[E, B, x, D] # 丢弃 C，索引 2 处产生空槽
批次大小: 4

3. 通过单向移动 3 -> 2 压缩批次并收缩批次大小
[E, B, D] x # 空槽现在处于批次之外
批次大小: 3

4. 注意力后端优化：通过 Swap 0 <-> 1 重排批次
[B, E, D]
批次大小: 3
```

最终生成的 `BatchUpdate` 数据结构将类似于：

``` text
BatchUpdate 实例
* added: [(0, E 的 SamplingParams, E 的 prompt tokens 引用, E 的 output tokens 引用)]
* removed: [2] # 请求 C 被移除且未被替换
* moved: [(3, 2, UNIDIRECTIONAL), (0, 1, SWAP)]
```

#### 示例 2：新请求多于已完成请求时的批次更新

以下示例模拟了一个引擎步骤：引入了 2 个新请求，消除了 1 个已完成的请求，此外注意力后端执行了一次交换以优化批次顺序。

``` text
批次状态 (引擎步骤开始时): [A, B, C, D]
批次大小: 4

新请求: E, F

已完成的请求: C

处理步骤 (使用 BatchUpdate 抽象):

1. 在索引 2 处添加 E
[A, B, E, D] # 丢弃 C
批次大小: 4

2. 在索引 4 (当前最大批次索引 + 1) 处添加 F
[A, B, E, D, F] # 批次大小扩展 1
批次大小: 5

3. 注意力后端优化：通过 Swap 0 <-> 1 重排批次
[B, A, E, D, F]
批次大小: 5
```

请注意，由于 Remove 操作没有留下空槽，因此跳过了批次压缩步骤。

最终生成的 `BatchUpdate` 数据结构将类似于：

``` text
BatchUpdate 实例
* added: [(2, E 的 SamplingParams, E 的 prompt tokens 引用, E 的 output tokens 引用), (4, F 的 SamplingParams, F 的 prompt tokens 引用, F 的 output tokens 引用)]
* removed: [] # 没有请求被移除且未被替换
* moved: [(0, 1, SWAP)]
```

## 如何向 vLLM 引入新的 Logits 处理器

### 编写内置 Logits 处理器的最佳实践

* 鉴于 Logits 处理器是在批次粒度下运行的，请编写高效的 `apply()` 和 `update_state()` 实现。
    * 例如，您可以尝试使用高效的向量化操作来实现 `apply()` 或在 `update_state()` 中更新内部状态向量。
    * 然而，如果您认为某个 Logits 处理器可能很少被使用，那么使用请求状态的“稀疏（sparse）”表示可能是合适的。即该类可以使用字典来表示请求配置，该字典仅存储启用了该 Logits 处理器的请求的元数据。

* 应当由 Logits 处理器的作者来决定：
    1. **用于针对该请求配置 Logits 处理器行为的逐请求属性**。例如，如果您正在为 vLLM 编写一个新的内置 Logits 处理器，您可能需要或不需要向 `SamplingParams` 和 vLLM REST API 添加额外的字段。
    2. **在逐请求的基础上启用或禁用该 Logits 处理器的条件**。除非您的目的是让内置 Logits 处理器始终对所有请求都起作用，否则您应当以这样一种方式编写您的 Logits 处理器：使其能够针对给定请求被禁用，即通过将某个参数默认设置为 `None`，或者通过传入一个特定的不起作用的参数值（如 `0.0`）。尽量为禁用了该 Logits 处理器的请求节省计算和内存开销。
    3. **在批次级别上短路（Short-circuit）Logits 处理器的条件**。即使您已经定义了在请求级别禁用内置 Logits 处理器的方法，如果您在 `update_state()` 和 `apply()` 中的实现使用了在单个命令中对整个常驻批次起作用的高效向量化操作，那么这可能很难转化为计算开销的节省。例如，您不能仅仅因为一个请求禁用了 Logits 处理器而跳过 `apply()` 中的整个向量化操作。为了在没有任何运行中的请求使用该内置 Logits 处理器的边界情况下节省计算开销，我们建议在所有请求都禁用了该 Logits 处理器时，将 `apply()` 设计为直接返回未修改的输入张量。类似地，如果没有任何请求启用该 Logits 处理器，考虑是否可以在 `update_state()` 中跳过一些步骤。
        * 此外，在 `update_state()` 中节省计算开销的一个简单方法是当 `batch_update` 为 `None` 时提前退出。

* 确保 Logits 处理器的 `update_state` 方法丢弃了关于已完成请求的信息（即被 Add 替换的请求或被 Remove 移除的请求）。

* 如果 Logits 处理器的行为是一致的，`is_argmax_invariant()` 可以硬编码为 `True` 或 `False`。然而，Argmax 不变性也可能需要通过编程方式确定（即如果您的 Logits 处理器以某种影响其是否为 Argmax 不变性的方式允许用户自定义）。因此，`is_argmax_invariant()` 不是类方法（Classmethod）。

### 内置 Logits 处理器 (Built-In Logits Processors)

当 vLLM 引擎启动时，内置的 Logits 处理器总是会被加载。可以参考 `vllm/v1/sample/logits_processor/builtin.py` 中现有的 vLLM 内置 Logits 处理器，以了解如何编写新的内置 Logits 处理器。如果某个 Logits 处理器可能会对广大受众有用，那么提交 PR 将其引入为内置 Logits 处理器是有意义的。基于上述编程模型，vLLM 目前采用了以下内置的 Logits 处理器：

- Min-P
- Logit bias (Logit 偏置)
- Min-tokens (最小 Token 数)

请审阅这些 Logits 处理器的具体实现，以获取编写内置 Logits 处理器的指导。

此外，以下类似于 Logits 处理器的功能目前被硬编码在采样器中，尚未利用上述编程模型。它们中的大多数都将被重构以使用上述 Logits 处理器编程模型：

- Allowed token IDs (允许的 Token ID)
- Bad words (禁用词)
- Repetition penalty (重复惩罚)
- Frequency penalty (频率惩罚)
- Presence penalty (存在惩罚)
- Temperature (温度)
- Top-K
- Top-P

### 自定义 Logits 处理器 (Custom Logits Processors)

可以使用 [用户提供的自定义 Logits 处理器](../features/custom_logitsprocs.md) 来增强 vLLM 的功能。
