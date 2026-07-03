# 混合 KV 缓存管理器 (Hybrid KV Cache Manager)

!!! warning "警告"
    本篇文档是基于 Commit [458e74](https://github.com/vllm-project/vllm/commit/458e74eb907f96069e6d8a4f3c9f457001fef2ea) 撰写的。此特性目前仍处于早期阶段，后续可能会发生变化。

## 什么是混合模型？ (What is a hybrid model?)

最近的许多“混合”大语言模型（LLMs）在单个模型中结合了多种注意力机制（Attention types）。例如：

1. **滑动窗口注意力 (Sliding window attention, sw) + 全文注意力 (Full attention)**：gpt-oss、Gemma 2/3、Ministral、Cohere 等。
2. **Mamba + 全文注意力**：Bamba、Jamba、Minimax 等。
3. **局部块注意力 (Local chunked attention) + 全文注意力**：Llama 4。

为了高效地为这些模型提供服务，我们的 [KVCacheManager][vllm.v1.core.kv_cache_manager.KVCacheManager] 必须：

1. **为不同的层类型分配不同的插槽 (Slots)**，例如：
    - 全文注意力层：为**所有** Token 保留插槽。
    - 滑动窗口注意力层：仅为最近的 **`sliding_window_size`** 个 Token 保留插槽。
2. **支持特定于层的前缀缓存（Prefix-cache）规则**，例如：
    - 全文注意力：缓存命中的前缀要求**所有** Token 都保留在 KV 缓存中。
    - 滑动窗口注意力：缓存命中的前缀仅要求最后 **`sliding_window_size`** 个 Token 保留在 KV 缓存中。

## 术语定义 (Definitions)

1. **kv 隐藏大小 (kv hidden size)**：存储单层中一个 Token 的 KV 缓存所需的字节数。
2. **块 (block)**：预留给 KV 缓存的内存被划分为多个具有相同 *页面大小（Page size）*（定义见下文）的 *块*。
3. **块大小 (block size)**：一个块中包含的 Token 数量。
4. **页面大小 (page size)**：一个块的物理内存大小，定义为：

    $$
    \text{num_layers} \times \text{block_size} \times \text{kv_hidden_size}
    $$

    `num_layers` 并不意味着模型中的总层数。其确切的数值取决于本文档中的上下文。

    !!! note "注意"
        这与代码中的 `KVCacheSpec.page_size_bytes` 不同，后者被定义为：

        $$
        \text{block_size} \times \text{kv_hidden_size}
        $$

## 缓存分配 (Allocation)

### 核心思想

我们对所有层类型使用单一内存池。该内存池被划分为多个具有相同页面大小的块。[KVCacheManager][vllm.v1.core.kv_cache_manager.KVCacheManager] 根据各层的注意力类型分配不同数量的块。

核心挑战是确保每个层类型都使用相同的**页面大小**。对于仅使用全文注意力的模型，页面大小非常直观，定义为：

$$
\text{page_size} = \text{block_size} \times \text{num_hidden_layers} \times \text{kv_hidden_size}
$$

然而，在混合模型中，不同注意力类型的 `num_hidden_layers` 是不同的，这通常会导致页面大小不匹配。下面的几个案例展示了我们如何将它们统一起来。

### 案例 1：玩具模型 (Toy Model)

让我们从一个简单的玩具示例开始：一个模型具有 1 个全文注意力层和 3 个滑动窗口注意力层。所有层具有相同的 `kv_hidden_size`。

我们让每个块持有单层中 `block_size` 个 Token，因此：

$$
\text{page_size} = \text{kv_hidden_size} \times \text{block_size}
$$

[KVCacheManager][vllm.v1.core.kv_cache_manager.KVCacheManager] 为每一层分配不同数量的块。

此案例仅是一个玩具示例。对于真实模型，请参考以下案例。

### 案例 2：相同 `kv_hidden_size` 且具有规律模式

当模型拥有更多层时（例如，20 个滑动窗口注意力层和 10 个全文注意力层，具有相同的 `kv_hidden_size`），如果为每一层都调用一次分配器（共 30 次调用）虽然可行，但效率低下。作为解决方案，我们对需要相同块数的层进行分组分配，以减少调用次数。

这种分组是可行的，因为不同类型层的数量之间通常存在一个优雅的比率。例如：

- Gemma-2：1 sw : 1 full
- Llama 4：3 local : 1 full

我们的示例可以看作是 2 sw : 1 full。我们可以分配块，就好像模型中只有 2 个 sw 和 1 个 full，然后将结果重复 10 次，以生成这 30 层的 `block_ids`。此时页面大小变为：

$$
10 \times \text{kv_hidden_size} \times \text{block_size}
$$

假设 `block_size` 为 16，滑动窗口大小为 32，请求长度为 112，那么对于上述示例模型，我们需要分配 11 个块（全文使用 0-6，sw 组 1 使用 7-8，sw 组 2 使用 9-10）。

![分配结果](../assets/design/hybrid_kv_cache_manager/basic_grouping_example.png)

在此图中，“/” 表示不需要分配块（滑动窗口层不需要为早期的 Token 分配插槽）。

下面是正式的定义。层被划分为多个 *KV 缓存组 (KV Cache Groups)*，使得：

1. **每个组内注意力类型完全一致**：每个组仅包含具有相同注意力类型的层，因此在给定请求下它们需要相同数量的块。这使得同一组内的层可以共享相同的块 ID，而不会造成内存浪费。
2. **各组之间的页面大小完全一致**：因为我们的内存池只支持单一页面大小。

我们的示例模型被划分为 3 个 KV 缓存组：

- 组 0：10 个全文注意力层 (full.0 - full.9)
- 组 1：10 个滑动窗口注意力层 (sw.0 - sw.9)
- 组 2：10 个滑动窗口注意力层 (sw.10 - sw.19)

显然，它满足规则 1。对于规则 2，所有 3 个组都将：

$$
10 \times \text{kv_hidden_size} \times \text{block_size}
$$

作为它们的页面大小。

### 案例 3：相同 `kv_hidden_size` 且没有规律模式

不幸的是，并非所有的模型都拥有如此完美的比例，案例 2 中的方法可能会产生过多的小分组。例如，Gemma-3-27b 拥有 52 个滑动窗口注意力层和 10 个全文注意力层。在案例 2 的约束下，它将产生 26 个滑动窗口组和 5 个全文注意力组，每个组仅包含 2 层。这种分配依然效率低下。为了减少 KV 缓存组的数量，我们使用所有注意力类型中最小的层数来对层进行分组。例如，在 Gemma-3-27b 中，每个组包含 min(52, 10) = 10 层。那么分组结果为：

- 组 0：10 个全文注意力层 (full.0 - full.9)
- 组 1：10 个滑动窗口注意力层 (sw.0 - sw.9)
- 组 2：10 个滑动窗口注意力层 (sw.10 - sw.19)
- ...
- 组 6：10 个滑动窗口注意力层 (sw.40 - sw.49)
- 组 7：2 个滑动窗口注意力层 (sw.50 - sw.51) 和 8 个填充层 (padding layers)

当新模型推出时，如果该启发式算法导致了不佳的结果（例如，20 个 full + 30 个 sw，组大小应该是 10 而不是 20），我们将更新此算法。

该案例发生在 Gemma-3 系列模型中，以及案例 2 模型但引入了一个全文注意力层的 Eagle 投机解码中。此解决方案存在一些内存浪费，并不完美。如果填充开销变得不可接受，请向我们报告，以便我们改进该算法。

### 案例 4：不同的 `kv_hidden_size`（主要是混合 Mamba 模型）

某些架构（例如 Bamba、Jamba、Minimax）将标准注意力层与 Mamba 层交错排列，其中每个 Mamba 层每个 Token 的状态大小（State size）可能远大于注意力层的 `kv_hidden_size`。由于我们只支持在所有组中使用单一页面大小，我们必须调和这些不同的隐藏大小。

当前的算法是：

1. 增加注意力层的 `block_size` 直到满足：
    $$
    \text{block_size} \times \text{kv_hidden_size}_{\text{att}} \ge \text{state_size}_{\text{mamba}}
    $$
2. 将每层 Mamba 状态填充到：
    $$
    \text{block_size} \times \text{kv_hidden_size}_{\text{att}}
    $$
3. 应用案例 3 中的分组策略。

!!! note "注意"
    这可能会导致注意力层的 `block_size` 超过 400，这实在太大了。另一种可行的填充策略是增加 `block_size` 直到满足：

    $$
    \text{block_size} \times \text{kv_hidden_size}_{\text{att}} \times \text{num_attn_layers} \ge \text{state_size}_{\text{mamba}}
    $$

    此填充策略目前仍在开发中。

### 案例 5：KV 共享 (KV Sharing)

KV 共享是指一个层使用另一个层的 KV 缓存，例如 gemma-3n。
在这些模型中，[KVCacheManager][vllm.v1.core.kv_cache_manager.KVCacheManager] 会忽略所有使用 KV 共享的层，只为需要 KV 缓存的层分配缓存，并在 Model Runner 中进行一些补丁修补，将分配结果应用到 KV 共享层。

## 前缀缓存 (Prefix caching)

为了简便起见，本节中我们假设 `block_size=1`。

### 核心思想

块池使用类似于 `tuple(block_hash, group_id) -> block` 的字典来缓存已存满的块。这意味着不同组的相同 Token 是独立缓存和淘汰的。

当有新请求传入时，我们检查每个组的缓存命中前缀，并返回这些组的交集（Intersection）作为请求的已缓存前缀。下面是检查一个组的缓存命中并执行求交集操作的详细算法。

### 案例 0：仅全文注意力模型

对于全文注意力层，需要为请求中的所有 Token 分配块。关于底层设计的详细信息，请参阅 [前缀缓存](prefix_caching_zh.md)。

为了找到请求的最长缓存命中前缀，我们从左（第一个块）到右（最后一个块）进行遍历，检查该块是否被缓存，并在缓存未命中时提前退出。例如，在下述示例中，我们将返回前 7 个 Token (0-6) 作为缓存命中前缀（蓝色块代表已缓存）：

![全文注意力的前缀缓存](../assets/design/hybrid_kv_cache_manager/full_attn.png)

### 案例 1：仅滑动窗口注意力模型

对于滑动窗口注意力层，一种简单的内存分配实现是分配 `sliding_window_size` 个块，并以循环（Round-robin）方式填满它们。但这种简单的实现与前缀缓存不兼容，因此我们没有选择这种设计。在 vLLM 中，我们为不同的 Token 分配不同的块，并释放滑动窗口之外的块。

对于新请求，缓存命中前缀仅需要最后 `sliding_window_size - 1` 个 Token 被缓存。
假设 `sliding_window_size = 4` 且 `block_size = 1`，请求是一个 15 个 Token 的 Prompt（蓝色块代表已缓存）：

![滑动窗口注意力的前缀缓存](../assets/design/hybrid_kv_cache_manager/sw_attn.png)

这里有 3 种可能的缓存命中前缀：

- 缓存命中长度为 5，使用 [2, 3, 4] 计算 Prefill → [5, 6, …, 14]
- 缓存命中长度为 6，使用 [3, 4, 5] 计算 Prefill → [6, 7, …, 14]
- 缓存命中长度为 14，使用 [11, 12, 13] 计算 Prefill → [14]（最有效）

我们可以从右到左检查缓存命中，并在找到匹配时提前退出。这与全文注意力正好相反，全文注意力是从左到右检查并在匹配失败时提前退出。滑动窗口注意力的一个潜在缺点（相比于全文注意力）是，在没有匹配时，我们最终会遍历整个 Token 列表，而这往往是常见的情况。这可能会导致不可忽视的开销，但在“全文注意力 + 滑动窗口注意力”的情况下表现良好，如下文所述。

### 案例 2：滑动窗口注意力 + 全文注意力模型

第一个问题是如何找到缓存命中前缀。我们需要通过以下方式“求交”全局注意力层和滑动窗口注意力层的缓存命中：

1. 获取全文注意力的最长缓存命中长度（从左到右扫描）。
2. 在该长度范围内，获取滑动窗口注意力的最长缓存命中长度。通过从全文注意力缓存命中长度的位置开始，从右到左检查缓存命中来实现。

这可以确保所得的滑动窗口注意力层的缓存命中也一定是全文注意力层的缓存命中。这比寻找每个组的所有可能前缀然后再进行求交集更为高效，因为如果没有任何缓存命中，我们的方法可以提前退出。

该算法适用于恰好包含全文注意力 + X 两种注意力类型的模型，其中 X 可以是任何高效的注意力算法，如滑动窗口、Llama 4 局部注意力以及 Mamba。它目前不支持没有全文注意力层的模型，以及拥有 2 种以上注意力类型的模型。在撰写本文档时，这对于大多数混合模型来说已经足够。

第二个问题是缓存淘汰策略。目前，我们对所有 KV 缓存组使用同一个 LRU 队列。当块被释放时（无论是因为请求完成还是块超出了滑动窗口），它们都会被添加到该 LRU 队列中。

### 案例 3：Mamba 模型

Mamba 模型的前缀缓存支持目前正在开发中。一旦实现，具有 Mamba 层 + 全文注意力层的模型将可以通过案例 2 中的“全文注意力 + X”算法来支持。

## 系统实现 (Implementation)

### 概述

![混合 KV 缓存管理器概述](../assets/design/hybrid_kv_cache_manager/overview.png)

`KVCacheManager` 被组织为 3 层：

- **[KVCacheManager][vllm.v1.core.kv_cache_manager.KVCacheManager]**：调度器与 KV 缓存管理系统之间的接口。
- **[KVCacheCoordinator][vllm.v1.core.kv_cache_coordinator.KVCacheCoordinator]**：协调各个组的 `SingleTypeKVCacheManager` 以生成请求的分配结果。根据模型的配置，会选择以下协调器之一：
    - **[KVCacheCoordinatorNoPrefixCache][vllm.v1.core.kv_cache_coordinator.KVCacheCoordinatorNoPrefixCache]**：在前缀缓存被禁用时使用。
    - **[UnitaryKVCacheCoordinator][vllm.v1.core.kv_cache_coordinator.UnitaryKVCacheCoordinator]**：若只有一个 KV 缓存组，由于不需要求交集，前缀缓存逻辑被简化。
    - **[HybridKVCacheCoordinator][vllm.v1.core.kv_cache_coordinator.HybridKVCacheCoordinator]**：处理恰好两个 KV 缓存组（必须包括一个全文注意力组加上另一个高效注意力组）。其他情况暂未实现。您可以禁用前缀缓存以使用 `KVCacheCoordinatorNoPrefixCache`。
- **[SingleTypeKVCacheManager][vllm.v1.core.single_type_kv_cache_manager.SingleTypeKVCacheManager]**：每个实例管理一个 KV 缓存组的分配和前缀缓存，实现特定于注意力机制的逻辑（例如全文注意力、滑动窗口、Mamba）。

上图中的蓝色框显示了具有 10 个全文注意力层和 20 个滑动窗口注意力层的情况，因此它会：

- 使用 `HybridKVCacheCoordinator`。
- 为 3 个 `KVCacheGroup` 使用 1 个 `FullAttentionManager` 和 2 个 `SlidingWindowManager`。

### 内存布局 (Memory Layout)

对于拥有 $n$ 个 `KVCacheGroup` 且每个组有 $m$ 层的模型，我们分配 $m$ 个缓冲区。每个缓冲区由 $n$ 层共享，每个组共享一层。

下图针对一个包含 10 个全文注意力层 (full.0 - full.9) 和 20 个滑动窗口注意力层 (sw.0 - sw.19) 的模型。它遵循“缓存分配”章节中的“案例 2”，被划分为 3 个组：

- 组 0：10 个全文注意力层 (full.0 - full.9)
- 组 1：10 个滑动窗口注意力层 (sw.0 - sw.9)
- 组 2：10 个滑动窗口注意力层 (sw.10 - sw.19)

对于某个请求，我们分配 11 个块，将 `block_id` 0-6 分配给组 0，7-8 分配给组 1，9-10 分配给组 2。

在该示例中，物理内存被划分为 10 个缓冲区（`KVCacheTensor` 0 - `KVCacheTensor` 9）。每个报告的缓冲区由 3 层共享（例如，`KVCacheTensor` 0 由组 0 的 full.0、组 1 的 sw.0 和组 2 的 sw.10 共享），并被划分为大小为 `block_size * kv_hidden_size` 的片（pieces）。这 3 个注意力层的 KV 缓存会根据所分配的 `block_ids` 被保存到缓冲区的不同片段中：

![内存布局示例](../assets/design/hybrid_kv_cache_manager/memory_layout.png)

!!! note "注意"
    一个逻辑上的“块”会被映射到物理内存 10 个缓冲区中的 10 个片段。
