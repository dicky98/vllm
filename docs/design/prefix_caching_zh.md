# 自动前缀缓存 (Automatic Prefix Caching)

对 KV 缓存块（KV-Cache Blocks）进行前缀缓存是大语言模型（LLM）推理中一种非常流行的优化手段，旨在避免冗余的提示词（Prompt）计算。其核心思想非常简单 —— 我们缓存已处理请求的 KV 缓存块，并在新请求传入时，若其与之前的请求具有相同的前缀，则直接复用这些块。由于前缀缓存几乎是一种无成本的优化，并且不会改变模型输出，因此它已被许多公共 API 端点（例如 OpenAI、Anthropic 等）和大多数开源 LLM 推理框架（例如 SGLang）广泛使用。

虽然前缀缓存有多种实现方案，但 vLLM 选择了一种基于哈希（Hash-based）的方法。具体来说，我们通过块中的 Token 以及该块之前的前缀 Token 来对每个 KV 缓存块进行哈希处理：

```text
                     Block 1                  Block 2                  Block 3
          [A gentle breeze stirred] [the leaves as children] [laughed in the distance]
Block 1: |<--- 块内 Token 元素 --->|
Block 2: |<------- 前缀 ------->| |<--- 块内 Token 元素 --->|
Block 3: |<------------------ 前缀 -------------------->| |<--- 块内 Token 元素 --->|
```

在上面的示例中，第一个块中的 KV 缓存可以通过 Token “A gentle breeze stirred” 来唯一标识。第三个块可以通过该块中的 Token “laughed in the distance” 以及前缀 Token “A gentle breeze stirred the leaves as children” 来唯一标识。因此，我们可以构建一个 `hash(tuple[components])` 结构的块哈希，其中组件（components）包括：

* **父块哈希值 (Parent hash value)**：当前块的父哈希块的哈希值。
* **块内 Token (Block tokens)**：此块中 Token 的元组（Tuple）。包含精确 Token 的目的是为了减少潜在的哈希值冲突。
* **额外哈希 (Extra hashes)**：使该块保持唯一所需的其他值，例如 LoRA ID、多模态输入哈希（参见下面的示例）以及用于在多租户环境中隔离缓存的缓存盐（Cache salts）。

!!! note "注意 1"
    我们只缓存已存满的块（Full blocks）。

!!! note "注意 2"
    在以前的版本中，哈希键（Hash key）不能保证完全没有冲突。自 v0.11 版本起，默认的哈希算法为 `sha256`，解决了冲突风险。

    对于 `vllm serve`，您可以通过 `--prefix-caching-hash-algo` 控制哈希算法：
    - `sha256`（默认）：使用 Python 的 `pickle` 进行序列化。哈希值在不同的 Python 或 vLLM 版本之间可能无法复现。
    - `sha256_cbor`：使用 `cbor2` 进行序列化，提供可复现的、跨语言兼容的哈希。推荐用于跨环境的确定性缓存。
    - `xxhash`：使用基于 xxHash（128位）的 Pickle 序列化以实现更快的非密码学哈希。需要安装可选包 `xxhash`。**重要提示**：使用不被认为密码学安全的哈希算法理论上会增加哈希冲突的风险，这可能会在多租户环境中导致未定义行为甚至泄露隐私信息。即使冲突的概率依然极低，在开启此选项之前，也必须在性能收益与您的安全风险承受能力之间进行权衡。
    - `xxhash_cbor`：结合了规范 CBOR 序列化与 xxHash，用于生成可复现的哈希。需要安装可选包 `xxhash`。

**多模态输入哈希示例**  
在本例中，我们阐述前缀缓存如何与多模态输入（如图像）一起工作。假设我们有一个包含以下消息的请求：

```text
messages = [
    {"role": "user",
     "content": [
         {"type": "text",
          "text": "What's in this image?"
         },
         {"type": "image_url",
          "image_url": {"url": image_url},
         },
    ]},
]
```

它将变成如下的提示词（Prompt）：

```text
提示词 (Prompt):
    <s>[INST]What's in this image?\n[IMG][/INST]

标记化提示词 (Tokenized prompt):
    [1, 3, 7493, 1681, 1294, 1593, 3937, 9551, 10, 4]

带占位符的提示词 (Prompt with placeholders (<P>)):
    [1, 3, 7493, 1681, 1294, 1593, 3937, 9551, <P>, <P>, ..., <P>, 4]
```

正如我们所看到的，在分词（Tokenization）之后，`[IMG]` 将被一序列占位符 Token 替换，这些占位符在预填充（Prefill）期间会被图像嵌入（Image embeddings）所替代。前缀缓存在此场景下面临的挑战是，我们需要将图像从占位符中区分出来。为了解决这个问题，我们编码了由前端图像处理器生成的图像哈希。例如，上述提示词中块的哈希将是（假设块大小为 16，我们有 41 个占位符 Token）：

```text
Block 0
    父哈希：None
    Token ID: 1, 3, 7493, 1681, 1294, 1593, 3937, 9551, <p>, ..., <p>
    额外哈希：<图像哈希>
Block 1
    父哈希：Block 0 哈希
    Token ID: <p>, ..., <p>
    额外哈希：<图像哈希>
Block 2
    父哈希：Block 1 哈希
    Token ID: <p>, ..., <p>
    额外哈希：<图像哈希>
Block 3
    父哈希：Block 2 哈希
    Token ID: <p>, ..., <p>, 4
    额外哈希：<图像哈希>
```

在本文档的其余部分，我们首先介绍 vLLM v1 中用于前缀缓存的数据结构，然后介绍主要 KV 缓存操作（例如分配、追加、释放、淘汰）的前缀缓存工作流。最后，我们通过一个例子来阐述端到端的前缀缓存工作流程。

**用于安全性的缓存隔离**  
为了提高共享环境中的隐私性，vLLM 支持通过可选的单请求加盐（Salting）来隔离前缀缓存的复用。通过在请求中包含 `cache_salt`，该值将被注入到第一个块的哈希中，从而确保只有具有相同 Salt 的请求才能复用缓存的 KV 块。这可以防止基于时间差的旁路攻击（攻击者通过观察延迟差异来推断缓存内容）。这在不牺牲性能的前提下提供了保护。

```json
{
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Here is a document with details about the world series: ..."},
    {"role": "user", "content": "Who won the world series in 2020?"}
  ],
  "cache_salt": "your-cache-salt"
}
```

通过此设置，缓存共享仅限于显式同意使用相同 Salt 的用户或请求，从而在信任组内实现缓存复用，同时隔离其他用户。

## 数据结构 (Data Structure)

vLLM v1 中的前缀缓存是在 KV 缓存管理器（KV cache manager）中实现的。其基本构建块是 “Block” 数据类（已简化）：

```python
class KVCacheBlock:
    # 块 ID（不可变）
    block_id: int
    # 块哈希（当块存满时分配，
    # 并在块被淘汰时重置）。
    block_hash: BlockHash
    # 当前正在使用此块的请求数量。
    ref_cnt: int

    # 用于为空闲队列构建双向链表的指针。
    prev_free_block: "KVCacheBlock | None" = None
    next_free_block: "KVCacheBlock | None" = None
```

这里有两个设计要点需要强调：

1. 我们在初始化 KV 缓存管理器时分配所有的 `KVCacheBlock`，使其成为一个块池（Block Pool）。这避免了 Python 对象创建的开销，并可以随时轻松地追踪所有块。
2. 我们直接在 `KVCacheBlock` 中引入了双向链表指针，以便我们能够直接构建一个空闲队列。这带来了两个好处：
    1. 我们可以以 O(1) 的复杂度将中间的元素移动到末尾。
    2. 我们可以避免引入另一个会对元素进行包装的 Python 队列（例如 `deque`）。

因此，在初始化 KV 缓存管理器时，我们将拥有以下组件：

![组件概述](../assets/design/prefix_caching/overview.png)

* **块池 (Block Pool)**：`KVCacheBlock` 的列表。
* **空闲块队列 (Free Block Queue)**：仅存储用于操作的头块和尾块的指针。
* **缓存块 (Cache blocks)**：从哈希键到块 ID 的映射。
* **请求块 (Request blocks)**：从请求 ID 到已分配块 ID 的映射。

## 相关操作 (Operations)

### 块分配 (Block Allocation)

**新请求**：调度器调度新请求并分配 KV 缓存块的工作流：

1. 调度器调用 `kv_cache_manager.get_computed_blocks()` 来获取已经计算过的块序列。这是通过对请求中的提示词 Token 进行哈希处理并在缓存块中进行查找来完成的。
2. 调度器调用 `kv_cache_manager.allocate_slots()`。它执行以下步骤：
    1. 计算所需新块的数量，如果可分配的块不足则返回。
    2. “触碰（Touch）”已计算出的块。它将已计算块的引用计数增加 1，并且如果该块未被其他请求使用，则将其从空闲队列中移除。这是为了避免这些已计算的块被淘汰。具体演示参见下一节的示例。
    3. 通过从空闲队列头部弹出块来分配新块。如果弹出的头部块是一个已缓存的块，这也会“淘汰（evict）”该块，从而使其他请求从现在起无法再复用它。
    4. 如果分配的块已经存满了 Token，我们立即将其添加到缓存块中，以便该块可以被同一批次中的其他请求复用。

**运行中的请求**：调度器调度正在运行的请求并分配 KV 缓存块的工作流：

1. 调度器调用 `kv_cache_manager.allocate_slots()`。它执行以下步骤：
    1. 计算所需新块的数量，如果可分配的块不足则返回。
    2. 通过从空闲队列头部弹出块来分配新块。如果弹出的头部块是一个已缓存的块，这也会“淘汰”该块，从而使其他请求从现在起无法再复用它。
    3. 将 Token ID 追加到现有块以及新块的插槽（slots）中。如果一个块存满了，我们将其添加到缓存块中进行缓存。

**重复块 (Duplicated blocks)**  
假设块大小为 4，您发送一个请求（请求 1），其提示词为 ABCDEF，解码长度为 3：

```text
提示词: [A, B, C, D, E, F]
输出: [G, H, I]

时刻 0:
  Tokens: [A, B, C, D, E, F, G]
  块表 (Block Table): [0 (ABCD), 1 (EFG)]
  缓存块 (Cache Blocks): 0
时刻 1:
  Tokens: [A, B, C, D, E, F, G, H]
  块表 (Block Table): [0 (ABCD), 1 (EFGH)]
  缓存块 (Cache Blocks): 0, 1
时刻 2:
  Tokens: [A, B, C, D, E, F, G, H, I]
  块表 (Block Table): [0 (ABCD), 1 (EFGH), 2 (I)]
  缓存块 (Cache Blocks): 0, 1
```

现在块 0 和块 1 已被缓存，我们再次发送相同的请求（请求 2）并使用贪婪采样（Greedy Sampling），因此它将产生与请求 1 完全相同的输出：

```text
提示词: [A, B, C, D, E, F]
输出: [G, H, I]

时刻 0:
  Tokens: [A, B, C, D, E, F, G]
  块表 (Block Table): [0 (ABCD), 3 (EFG)]
  缓存块 (Cache Blocks): 0, 1
时刻 1:
  Tokens: [A, B, C, D, E, F, G, H]
  块表 (Block Table): [0 (ABCD), 3 (EFGH)]
  缓存块 (Cache Blocks): 0, 1, 3
```

可以看出，块 3 是一个新的已存满的块并被缓存。然而，它与块 1 是重复的，这意味着我们两次缓存了相同的块。在 v0 中，当检测到块 3 重复时，我们会释放块 3 并让请求 2 改为使用块 1，因此在时刻 1 其块表变为 `[0, 1]`。然而，vLLM v1 中的块表是只准追加（Append-only）的，这意味着不允许将块表从 `[0, 3]` 更改为 `[0, 1]`。因此，对于哈希键 E-H，我们将拥有重复的块。这种重复将在释放请求时被消除。

### 释放 (Free)

当一个请求完成时，如果没有其他请求正在使用它们（引用计数 = 0），我们会释放其所有块。在此示例中，我们释放请求 1 以及与之关联的块 2、3、4、8。我们可以看到，释放的块以*相反*的顺序被添加到空闲队列的尾部。这是因为请求的最后一个块必须对更多的 Token 进行哈希，被其他请求复用的可能性较低。因此，它应该被优先淘汰。

![请求被释放后的空闲队列](../assets/design/prefix_caching/free.png)

### 淘汰 - LRU (Eviction)

当空闲队列的头部块（最近最少使用的块）被缓存时，我们必须淘汰该块以防止它被其他请求使用。具体来说，淘汰包括以下步骤：

1. 从空闲队列的头部弹出该块。这是要被淘汰的 LRU 块。
2. 从缓存块中移除该块的 ID。
3. 移除该块的哈希值。

## 示例 (Example)

在此示例中，我们假设块大小为 4（每个块可以缓存 4 个 Token），并且我们在 KV 缓存管理器中总共有 10 个块。

**时刻 1：缓存为空，一个新请求到来。** 我们分配 4 个块。其中 3 个已经存满并被缓存。第四个块部分存满，包含 4 个 Token 中的 3 个。

![示例时刻 1](../assets/design/prefix_caching/example-time-1.png)

**时刻 2：请求 0 使块 3 存满，并请求一个新块以继续解码。** 我们缓存块 3 并分配块 4。

![示例时刻 2](../assets/design/prefix_caching/example-time-3.png)

**时刻 3：请求 1 携带 14 个提示词 Token 到来，其中前 10 个 Token 与请求 0 相同。** 我们可以看到只有前 2 个块（8 个 Token）命中了缓存，因为第 3 个块仅匹配 4 个 Token 中的 2 个。

![示例时刻 3](../assets/design/prefix_caching/example-time-4.png)

**时刻 4：请求 0 运行结束并被释放。** 块 2、3 和 4 以相反的顺序被添加到空闲队列中（但块 2 和 3 仍然被缓存）。块 0 和 1 没有被添加到空闲队列中，因为它们正被请求 1 使用。

![示例时刻 4](../assets/design/prefix_caching/example-time-5.png)

**时刻 5：请求 1 运行结束并被释放。**

![示例时刻 5](../assets/design/prefix_caching/example-time-6.png)

**时刻 6：请求 2 携带 29 个提示词 Token 到来，其中前 12 个 Token 与请求 0 相同。** 请注意，即使空闲队列中的块顺序为 `7 - 8 - 9 - 4 - 3 - 2 - 6 - 5 - 1 - 0`，命中缓存的块（即 0、1、2）在分配前会被触碰并从队列中移除，因此空闲队列变为 `7 - 8 - 9 - 4 - 3 - 6 - 5`。结果，分配的块为 0（已缓存）、1（已缓存）、2（已缓存）、7、8、9、4，而 3 被淘汰。

![示例时刻 6](../assets/design/prefix_caching/example-time-7.png)
