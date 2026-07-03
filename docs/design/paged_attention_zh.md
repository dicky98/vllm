# Paged Attention (分页注意力)

!!! warning "警告"
    这是一篇基于 [vLLM 原始论文](https://arxiv.org/abs/2309.06180) 的历史文档。
    它已不再描述 vLLM 今天所使用的实际代码。

目前，vLLM 利用其自身实现的多头查询注意力核（Multi-head query attention kernel，源于 `csrc/attention/attention_kernels.cu`）。
该算子核设计为与 vLLM 的分页 KV 缓存（Paged KV Caches）兼容，其中键（Key）缓存和值（Value）缓存被存储在独立的块（Blocks）中（注意，这里的“块”概念与 GPU 线程块 Thread Block 不同。因此在后文中，我将把 vLLM 的分页注意力块简称为“块 Block”，而将 GPU 线程块称为“线程块 Thread Block”）。

为了实现高性能，该算子核依赖于一种专门设计的内存布局和访问方法，特别是当线程将数据从全局内存（Global Memory）读取到共享内存（Shared Memory）时。
本文档的目的是逐步提供该算子核实现的高层解释，以帮助那些希望了解 vLLM 多头查询注意力核的人。在阅读完本文档后，用户可能会有更好的理解，并觉得更容易跟进实际的代码实现。

请注意，本文档可能不会涵盖所有细节，例如如何计算对应数据的正确索引或点积的具体实现。然而，在阅读本文档并熟悉了高层逻辑流之后，阅读实际代码并理解细节应该会变得更加容易。

## 输入 (Inputs)

算子核函数接收一系列参数以供当前线程执行其分配的工作。其中最关键的三个参数是输入指针 `q`、`k_cache` 和 `v_cache`，它们指向全局内存上需要被读取和处理的查询（Query）、键（Key）和值（Value）数据。输出指针 `out` 指向应该写入结果的全局内存。这四个指针实际上引用了多维数组，但每个线程仅访问分配给它的那部分数据。为了简单起见，我在此处省略了所有其他运行时参数。

```cpp
template<typename scalar_t, int HEAD_SIZE, int BLOCK_SIZE, int NUM_THREADS, int PARTITION_SIZE = 0>
__device__ void paged_attention_kernel(
    ... // 其他附带参数。
    const scalar_t* __restrict__ out,       // [num_seqs, num_heads, max_num_partitions, head_size]
    const scalar_t* __restrict__ q,         // [num_seqs, num_heads, head_size]
    const scalar_t* __restrict__ k_cache,   // [num_blocks, num_kv_heads, head_size/x, block_size, x]
    const scalar_t* __restrict__ v_cache,   // [num_blocks, num_kv_heads, head_size, block_size]
    ... // 其他附带参数。
)
```

在函数签名上方还有一系列在编译期确定的模板参数。`scalar_t` 代表查询、键和值数据元素的数据类型（例如 FP16）。`HEAD_SIZE` 表示每个 Head 中的元素数量。`BLOCK_SIZE` 指的是每个块（Block）中的 Token 数量。`NUM_THREADS` 表示每个线程块（Thread Block）中的线程数量。`PARTITION_SIZE` 表示张量并行（Tensor Parallel）GPU 的数量（为了简单起见，我们假设该值为 0，且张量并行已禁用）。

通过这些参数，我们需要执行一系列准备工作。这包括计算当前 Head 索引、块索引以及其他必要的变量。不过现在我们可以先忽略这些准备步骤，直接进入实际的计算过程。一旦我们理解了整个流程，就更容易理解它们了。

## 相关概念 (Concepts)

在我们深入了解计算流程之前，我想先介绍几个后续章节会用到的概念。如果您遇到任何令人困惑的术语，可以跳过本节并在稍后重新阅读。

- **序列 (Sequence)**：一个序列代表一个客户端请求。例如，`q` 所指向的数据形状为 `[num_seqs, num_heads, head_size]`。这代表总共有 `num_seqs` 个由 `q` 指向的查询序列数据。由于该算子核是单查询注意力核（Single query attention kernel），因此每个序列只有一个查询 Token。因此，`num_seqs` 等于当前批处理中处理的 Token 总数。
- **上下文 (Context)**：上下文由序列中已生成的 Token 组成。例如，`["What", "is", "your"]` 是上下文 Token，而输入的查询 Token 是 `"name"`。模型可能会生成 Token `"?"`。
- **向量化 (Vec)**：Vec 是被同时获取和计算的元素列表。对于查询和键数据，向量大小（`VEC_SIZE`）被确定为以便每个线程组每次可以获取并计算 16 字节的数据。对于值数据，向量大小（`V_VEC_SIZE`）被确定为以便每个线程每次可以获取并计算 16 字节的数据。例如，如果 `scalar_t` 是 FP16（2 字节）且 `THREAD_GROUP_SIZE` 是 2，则 `VEC_SIZE` 将为 4，而 `V_VEC_SIZE` 将为 8。
- **线程组 (Thread group)**：线程组是一小组线程（数量为 `THREAD_GROUP_SIZE`），每次获取并计算一个查询 Token 和一个键 Token。每个线程仅处理 Token 数据的一部分。由一个线程组处理的元素总数被称为 `x`。例如，如果线程组包含 2 个线程且 Head 大小为 8，那么线程 0 处理索引为 0, 2, 4, 6 的查询和键元素，而线程 1 处理索引为 1, 3, 5, 7 的元素。
- **块 (Block)**：vLLM 中的键和值缓存数据被划分为块。每个块在每个 Head 上存储固定数量（`BLOCK_SIZE`）的 Token 数据。每个块可能只包含整个上下文 Token 的一部分。例如，如果块大小为 16 且 Head 大小为 128，那么对于一个 Head，一个块可以存储 16 * 128 = 2048 个元素。
- **Warp (线程束)**：Warp 是在流式多处理器（SM）上同时执行的一组 32 个线程（数量为 `WARP_SIZE`）。在此算子核中，每个 Warp 每次处理一个查询 Token 与一个完整块的键 Token 之间的计算（它可能会在多次迭代中处理多个块）。例如，如果一个上下文有 4 个 Warp 和 6 个块，分配方式将是：Warp 0 处理第 0、4 块，Warp 1 处理第 1、5 块，Warp 2 处理第 2 块，Warp 3 处理第 3 块。
- **线程块 (Thread block)**：线程块是一组可以访问相同共享内存的线程（数量为 `NUM_THREADS`）。每个线程块包含多个 Warp（数量为 `NUM_WARPS`），在此算子核中，每个线程块处理一个查询 Token 与整个上下文的键 Token 之间的计算。
- **网格 (Grid)**：网格是线程块的集合，并定义了该集合的形状。在该算子核中，形状为 `(num_heads, num_seqs, max_num_partitions)`。因此，每个线程块仅处理一个 Head、一个序列和一个 Partition 的计算。

## 查询 (Query)

本节将介绍查询数据在内存中是如何存储以及如何被各个线程获取的。如上所述，每个线程组获取一个查询 Token 数据，而每个线程本身仅处理一个查询 Token 数据的一部分。在每个 Warp 内，每个线程组都会获取相同的查询 Token 数据，但会将其与不同的键 Token 数据进行乘法运算。

```cpp
const scalar_t* q_ptr = q + seq_idx * q_stride + head_idx * HEAD_SIZE;
```

![query](../assets/design/paged_attention/query.png)

每个线程定义其自己的 `q_ptr`，它指向全局内存上分配的查询 Token 数据。例如，如果 `VEC_SIZE` 为 4 且 `HEAD_SIZE` 为 128，则 `q_ptr` 指向的数据包含总共 128 个元素，被划分为 128 / 4 = 32 个 Vec。

![q_vecs](../assets/design/paged_attention/q_vecs.png)

```cpp
__shared__ Q_vec q_vecs[THREAD_GROUP_SIZE][NUM_VECS_PER_THREAD];
```

接下来，我们需要将 `q_ptr` 指向的全局内存数据读取到共享内存中，存为 `q_vecs`。需要特别注意的是，每个 Vec 被分配到不同的行。例如，如果 `THREAD_GROUP_SIZE` 是 2，则线程 0 将处理第 0 行的 Vec，而线程 1 处理第 1 行的 Vec。通过以这种方式读取查询数据，相邻的线程（如线程 0 和线程 1）可以读取相邻的内存，从而实现内存合并（Memory Coalescing）以提高性能。

## 键 (Key)

与“查询”一节类似，本节介绍键内存的布局和分配。虽然每个线程组在单次算子核运行中仅处理一个查询 Token，但在多次迭代中它可能会处理多个键 Token。与此同时，每个 Warp 将在多次迭代中处理多个块的键 Token，确保在算子核运行结束时所有的上下文 Token 都已被整个线程组处理。在此上下文中，“处理”指的是执行查询数据与键数据之间的点积运算。

```cpp
const scalar_t* k_ptr = k_cache + physical_block_number * kv_block_stride
                    + kv_head_idx * kv_head_stride
                    + physical_block_offset * x;
```

与 `q_ptr` 不同，每个线程中的 `k_ptr` 在不同迭代中会指向不同的键 Token。如上所示，`k_ptr` 基于分配的块、分配的 Head 和分配的 Token 处的 `k_cache` 指向键 Token 数据。

![key](../assets/design/paged_attention/key.png)

上图展示了键数据的内存布局。它假设 `BLOCK_SIZE` 为 16，`HEAD_SIZE` 为 128，`x` 为 8，`THREAD_GROUP_SIZE` 为 2，且总共有 4 个 Warp。每个矩形代表一个 Head 上的一个键 Token 的所有元素，这将由一个线程组处理。左半部分显示了分配给 Warp 0 的总共 16 块键 Token 数据，而右半部分代表其他 Warp 或迭代的其余键 Token 数据。在每个矩形内部，总共有 32 个 Vec（一个 Token 的 128 个元素），它们将分别由 2 个线程（一个线程组）处理。

![k_vecs](../assets/design/paged_attention/k_vecs.png)

```cpp
K_vec k_vecs[NUM_VECS_PER_THREAD]
```

接下来，我们需要从 `k_ptr` 读取键 Token 数据，并将它们作为 `k_vecs` 存储在寄存器内存中。我们对 `k_vecs` 使用寄存器内存，因为它们仅会被一个线程访问一次，而 `q_vecs` 将被多个线程访问多次。每个 `k_vecs` 将包含多个向量供后续计算使用。每个 Vec 会在每次内层迭代中设置。Vec 的分配方式同样允许 Warp 中的相邻线程一起读取相邻的内存，这再次促进了内存合并。例如，线程 0 将读取 Vec 0，而线程 1 将读取 Vec 1。在下一个内层循环中，线程 0 将读取 Vec 2，而线程 1 将读取 Vec 3，依此类推。

您可能仍对整体流程有些困惑。不用担心，请继续阅读接下来的“QK”一节。它将以更清晰和更高层的方式阐述查询与键的计算流程。

## QK 计算

如以下伪代码所示，在整个循环块之前，我们获取一个 Token 的查询数据并将其存储在 `q_vecs` 中。然后，在外层 for 循环中，我们遍历指向不同 Token 的不同 `k_ptrs`，并在内层 for 循环中准备 `k_vecs`。最后，我们在 `q_vecs` 与每个 `k_vecs` 之间执行点积运算。

```cpp
q_vecs = ...
for ... {
    k_ptr = ...
    for ... {
        k_vecs[i] = ...
    }
    ...
    float qk = scale * Qk_dot<scalar_t, THREAD_GROUP_SIZE>::dot(q_vecs[thread_group_offset], k_vecs);
}
```

如前所述，对于每个线程，它每次仅获取一部分查询和键 Token 数据。然而，在 `Qk_dot<>::dot` 内部会发生跨线程组的规约（Reduction）。因此，此处返回的 `qk` 不仅仅是部分查询与键 Token 点积的结果，而是整个查询与键 Token 数据之间的完整点积结果。

例如，如果 `HEAD_SIZE` 的值为 128 且 `THREAD_GROUP_SIZE` 是 2，则每个线程的 `k_vecs` 将总共包含 64 个元素。然而，返回的 `qk` 实际上是 128 个查询元素与 128 个键元素点积的结果。如果您想了解关于点积和规约的更多细节，可以参考 `Qk_dot<>::dot` 的实现。为简单起见，我不会在本文档中进行详细展开。

## Softmax 计算

接下来，我们需要为所有的 `qk`（其中每个 $x$ 代表一个 `qk`）计算归一化的 Softmax。为此，我们必须获取所有 `qk` 的规约最大值 `qk_max`（即 $m(x)$）和指数和 `exp_sum`（即 $\ell(x)$）。规约应该在整个线程块内进行，包含查询 Token 与所有上下文键 Token 之间的结果。

$$
\begin{gather*}
m(x):=\max _i \quad x_i \\ \quad f(x):=\left[\begin{array}{lll}e^{x_1-m(x)} & \ldots & e^{x_B-m(x)}\end{array}\right]\\ \quad \ell(x):=\sum_i f(x)_i \\
\quad \operatorname{softmax}(x):=\frac{f(x)}{\ell(x)}
\end{gather*}
$$

### `qk_max` 与 `logits`

在获得 `qk` 结果后，我们可以直接用 `qk` 填充临时 `logits` 结果（最终，`logits` 应该存储归一化的 Softmax 结果）。我们还可以比较并收集由当前线程组计算的所有 `qk` 的 `qk_max`。

```cpp
if (thread_group_offset == 0) {
    const bool mask = token_idx >= context_len;
    logits[token_idx - start_token_idx] = mask ? 0.f : qk;
    qk_max = mask ? qk_max : fmaxf(qk_max, qk);
}
```

请注意，此处的 `logits` 存储在共享内存中，因此每个线程组将为其自己分配的上下文 Token 设置相应字段。总体而言，logits 的大小应该是上下文 Token 的数量。

```cpp
for (int mask = WARP_SIZE / 2; mask >= THREAD_GROUP_SIZE; mask /= 2) {
    qk_max = fmaxf(qk_max, VLLM_SHFL_XOR_SYNC(qk_max, mask));
}

if (lane == 0) {
    red_smem[warp_idx] = qk_max;
}
```

然后，我们需要获取每个 Warp 内规约后的 `qk_max`。其核心思想是让 Warp 中的线程相互通信并获得最终的最大 `qk`。

```cpp
for (int mask = NUM_WARPS / 2; mask >= 1; mask /= 2) {
    qk_max = fmaxf(qk_max, VLLM_SHFL_XOR_SYNC(qk_max, mask));
}
qk_max = VLLM_SHFL_SYNC(qk_max, 0);
```

最后，我们可以通过比较该线程块中所有 Warp 的 `qk_max`，获得整个线程块内规约后的 `qk_max`。然后，我们需要将最终结果广播给每个线程。

### `exp_sum`

与 `qk_max` 类似，我们同样需要获取整个线程块内规约后的累加和值。

```cpp
for (int i = thread_idx; i < num_tokens; i += NUM_THREADS) {
    float val = __expf(logits[i] - qk_max);
    logits[i] = val;
    exp_sum += val;
}
...
exp_sum = block_sum<NUM_WARPS>(&red_smem[NUM_WARPS], exp_sum);
```

首先，累加来自每个线程组的所有指数（exp）值，与此同时，将 `logits` 的每个条目从 `qk` 转换为 `exp(qk - qk_max)`。请注意，此处的 `qk_max` 已经是整个线程块内的最大 `qk`。然后，我们可以在整个线程块内对 `exp_sum` 进行规约，就像对 `qk_max` 所做的那样。

```cpp
const float inv_sum = __fdividef(1.f, exp_sum + 1e-6f);
for (int i = thread_idx; i < num_tokens; i += NUM_THREADS) {
    logits[i] *= inv_sum;
}
```

最后，利用规约后的 `qk_max` 和 `exp_sum`，我们可以获得最终归一化的 Softmax 结果存入 `logits`。这个 `logits` 变量将在后面的步骤中用于与值（Value）数据进行点积。现在，它应该存储了所有已分配上下文 Token 的 `qk` 归一化 Softmax 结果。

## 值 (Value)

![value](../assets/design/paged_attention/value.png)

![logits_vec](../assets/design/paged_attention/logits_vec.png)

![v_vec](../assets/design/paged_attention/v_vec.png)

现在我们需要检索值（Value）数据并与 `logits` 执行点积运算。与查询和键不同，值数据没有线程组的概念。如上图所示，与键 Token 的内存布局不同，来自同一列的元素对应于相同的值 Token。对于一个块的值数据，有 `HEAD_SIZE` 行和 `BLOCK_SIZE` 列，它们被划分为多个 `v_vec`。

每个线程每次总是从相同的 `V_VEC_SIZE` 个 Token 中获取 `V_VEC_SIZE` 个元素。因此，单个线程在多次内层迭代中会检索来自不同行但相同列的多个 `v_vec`。对于每个 `v_vec`，它需要与对应的 `logits_vec`（这也是来自 `logits` 的 `V_VEC_SIZE` 个元素）进行点积。总体而言，通过多次内层迭代，每个 Warp 将处理一个块的值 Token。通过多次外层迭代，所有上下文的值 Token 都会被处理完毕。

```cpp
float accs[NUM_ROWS_PER_THREAD];
for ... { // 遍历不同块的迭代。
    logits_vec = ...
    for ... { // 遍历不同行的迭代。
        v_vec = ...
        ...
        accs[i] += dot(logits_vec, v_vec);
    }
}
```

如上述伪代码所示，在外层循环中，与 `k_ptr` 类似，`logits_vec` 遍历不同的块并从 `logits` 读取 `V_VEC_SIZE` 个元素。在内层循环中，每个线程从相同的 Token 读取 `V_VEC_SIZE` 个元素作为 `v_vec` 并执行点积。需要特别注意的是，在每次内层迭代中，线程获取相同 Token 在不同 Head 位置的元素。然后，点积结果被累加到 `accs` 中。因此，`accs` 的每个条目都映射到分配给当前线程的 Head 位置。

例如，如果 `BLOCK_SIZE` 为 16 且 `V_VEC_SIZE` 为 8，则每个线程每次获取 8 个 Token 的 8 个值元素。每个元素来自相同 Head 位置的不同 Token。如果 `HEAD_SIZE` 为 128 且 `WARP_SIZE` 为 32，则在每次内层循环中，一个 Warp 需要获取 `WARP_SIZE * V_VEC_SIZE = 256` 个元素。这意味着一个 Warp 总共需要 128 * 16 / 256 = 8 次内层迭代来处理一整块的值 Token。每个线程中的每个 `accs` 包含在 8 个不同 Head 位置累加的 8 个元素。对于线程 0，`accs` 变量将拥有 8 个元素，它们是值 Head 中从所有分配的 8 个 Token 累加而来的第 0, 32 … 224 个元素。

## LV 规约

现在，我们需要在每个 Warp 内对 `accs` 进行规约。这一过程使每个线程能够累加一个块中所有 Token 对应已分配 Head 位置的 `accs`。

```cpp
for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {
    float acc = accs[i];
    for (int mask = NUM_V_VECS_PER_ROW / 2; mask >= 1; mask /= 2) {
        acc += VLLM_SHFL_XOR_SYNC(acc, mask);
    }
    accs[i] = acc;
}
```

接下来，我们在所有 Warp 之间对 `accs` 进行规约，使每个线程拥有所有上下文 Token 对应已分配 Head 位置的 `accs` 累加和。请注意，每个线程中的每个 `accs` 仅存储所有上下文 Token 整个 Head 中一部分元素的累加结果。不过，总体而言，输出的所有结果都已被计算出来，只是存储在不同线程的寄存器内存中。

??? code

    ```cpp
    float* out_smem = reinterpret_cast<float*>(shared_mem);
    for (int i = NUM_WARPS; i > 1; i /= 2) {
        // 高位 Warp 写入共享内存。
        ...
        float* dst = &out_smem[(warp_idx - mid) * HEAD_SIZE];
        for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {
            ...
            dst[row_idx] = accs[i];
        }

        // 低位 Warp 更新输出。
        const float* src = &out_smem[warp_idx * HEAD_SIZE];
        for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {
            ...
            accs[i] += src[row_idx];
        }

        // 写出 accs。
    }
    ```

## 输出 (Output)

现在我们可以将所有计算出的结果从本地寄存器内存写入最终的输出全局内存中。

```cpp
scalar_t* out_ptr = out + seq_idx * num_heads * max_num_partitions * HEAD_SIZE
                + head_idx * max_num_partitions * HEAD_SIZE
                + partition_idx * HEAD_SIZE;
```

首先，我们需要定义 `out_ptr` 变量，它指向分配的序列和分配的 Head 的起始地址。

```cpp
for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {
    const int row_idx = lane / NUM_V_VECS_PER_ROW + i * NUM_ROWS_PER_ITER;
    if (row_idx < HEAD_SIZE && lane % NUM_V_VECS_PER_ROW == 0) {
        from_float(*(out_ptr + row_idx), accs[i]);
    }
}
```

最后，我们需要遍历不同的已分配 Head 位置，并基于 `out_ptr` 写出对应的累加结果。

## 引用文献 (Citation)

```bibtex
@inproceedings{kwon2023efficient,
  title={Efficient Memory Management for Large Language Model Serving with PagedAttention},
  author={Woosuk Kwon and Zhuohan Li and Siyuan Zhuang and Ying Sheng and Lianmin Zheng and Cody Hao Yu and Joseph E. Gonzalez and Hao Zhang and Ion Stoica},
  booktitle={Proceedings of the ACM SIGOPS 29th Symposium on Operating Systems Principles},
  year={2023}
}
```
