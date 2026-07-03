# 自动前缀缓存 (Automatic Prefix Caching)

## 简介

自动前缀缓存（简称 APC）会缓存现有查询的 KV 缓存（Key-Value Cache），这样当新查询与现有查询共享相同的前缀时，新查询可以直接复用这部分 KV 缓存，从而跳过共享部分的计算。

!!! note
    关于 vLLM 内部如何实现 APC 的技术细节，可以参考[这里](../design/prefix_caching.md)。

## 在 vLLM 中启用 APC

在 vLLM 引擎中设置 `enable_prefix_caching=True` 即可启用 APC。以下是一个示例：

[examples/features/automatic_prefix_caching/automatic_prefix_caching_offline.py](../../examples/features/automatic_prefix_caching/automatic_prefix_caching_offline.py)

## 典型应用场景

以下是 APC 能带来巨大性能收益的两个典型应用场景：

- **长文档查询**：用户使用不同的问题重复查询同一个长文档（例如软件手册或年度报告）。在这种情况下，APC 允许 vLLM *仅处理一次* 该长文档，而无需一次又一次地重新处理它，所有未来的请求都可以通过复用其 KV 缓存来避免重复计算该长文档。这使得 vLLM 能够以更高的吞吐量和更低的延迟来响应后续请求。
- **多轮对话**：用户在同一个聊天会话中多次与应用进行对话。在这种情况下，APC 允许 vLLM 在所有后续对话中复用聊天历史的计算结果，而不是一遍又一遍地处理整个聊天历史，从而使 vLLM 能够以更高的吞吐量和更低的延迟来服务后续请求。

## 局限性

总体而言，APC 不会降低 vLLM 的性能。话虽如此，APC 仅仅减少了处理查询的时间（预填充阶段，prefilling phase），而不会减少生成新 Token 的时间（解码阶段，decoding phase）。因此，当 vLLM 的大部分时间都花在生成查询的回答上（例如回答长度很长时），或者新查询与任何现有查询均不共享相同的前缀时（此时计算无法被复用），APC 不会带来明显的性能提升。
