# 指标度量 (Metrics)

vLLM 暴露了丰富的指标，以支持 V1 引擎的可观测性（Observability）和容量规划（Capacity planning）。

## 目标 (Objectives)

- 提供引擎和请求级别的指标的全面覆盖，以辅助生产监控。
- 优先考虑 Prometheus 集成，因为我们期望这是在生产环境中主要使用的监控方式。
- 为临时测试、调试、开发和探索性使用场景提供日志记录支持（例如，将指标打印到 INFO 日志中）。

## 背景介绍 (Background)

vLLM 中的指标可以分类如下：

1. **服务器级别（Server-level）指标**：跟踪 LLM 引擎状态和性能的全局指标。在 Prometheus 中，它们通常暴露为 Gauge（仪表盘）或 Counter（计数器）。
2. **请求级别（Request-level）指标**：跟踪单个请求特征（如大小和计时）的指标。在 Prometheus 中，它们通常暴露为 Histogram（直方图），并且通常是负责监控 vLLM 的 SRE 团队所跟踪的 SLO（服务水平目标）。

理解它们的直观心智模型是：服务器级别指标有助于解释请求级别指标数值的起伏。

### 指标概述 (Metrics Overview)

### V1 版本的指标

在 V1 中，大量的指标通过 Prometheus 兼容的 `/metrics` 接口以 `vllm:` 为前缀暴露，例如：

- `vllm:num_requests_running` (Gauge) - 当前正在运行的请求数量。
- `vllm:kv_cache_usage_perc` (Gauge) - 已使用的 KV 缓存块所占比例 (0–1)。
- `vllm:prefix_cache_queries` (Counter) - 前缀缓存查询次数。
- `vllm:prefix_cache_hits` (Counter) - 前缀缓存命中次数。
- `vllm:prompt_tokens_total` (Counter) - 已处理的 Prompt Token 总数。
- `vllm:generation_tokens_total` (Counter) - 已生成的 Token 总数。
- `vllm:request_success_total` (Counter) - 结束的请求数（按结束原因分类）。
- `vllm:request_prompt_tokens` (Histogram) - 输入 Prompt Token 数量的直方图。
- `vllm:request_generation_tokens` (Histogram) - 生成 Token 数量的直方图。
- `vllm:time_to_first_token_seconds` (Histogram) - 首字延迟 (TTFT)。
- `vllm:inter_token_latency_seconds` (Histogram) - 逐字生成延迟 (TPOT)。
- `vllm:e2e_request_latency_seconds` (Histogram) - 端到端请求延迟。
- `vllm:request_prefill_time_seconds` (Histogram) - 请求 Prefill 阶段耗时。
- `vllm:request_decode_time_seconds` (Histogram) - 请求 Decode 阶段耗时。

这些指标被记录在 [推理和服务 -> 生产指标 (Production Metrics)](../usage/metrics.md) 中。

### Grafana 仪表盘

vLLM 还提供了一个[参考示例](../../examples/observability/prometheus_grafana/README.md)，用于展示如何使用 Prometheus 收集和存储这些指标，并使用 Grafana 仪表盘对其进行可视化。

在 Grafana 仪表盘中暴露的指标子集向我们指明了哪些指标尤为重要：

- `vllm:e2e_request_latency_seconds_bucket` - 以秒为单位测量的端到端请求延迟。
- `vllm:prompt_tokens` - Prompt Token 数。
- `vllm:generation_tokens` - 生成的 Token 数。
- `vllm:inter_token_latency_seconds` - 以秒为单位的逐字延迟（Time Per Output Token, TPOT）。
- `vllm:time_to_first_token_seconds` - 以秒为单位的首字延迟（Time to First Token, TTFT）。
- `vllm:num_requests_running`（以及 `_swapped` 和 `_waiting`） - 处于 RUNNING、WAITING 和 SWAPPED 状态的请求数。
- `vllm:kv_cache_usage_perc` - vLLM 已使用的缓存块所占百分比。
- `vllm:request_prompt_tokens` - 请求的 Prompt 长度。
- `vllm:request_generation_tokens` - 请求的生成长度。
- `vllm:request_success` - 按结束原因分类的已完成请求数：生成了 EOS Token 或达到了最大序列长度。
- `vllm:request_queue_time_seconds` - 排队时间。
- `vllm:request_prefill_time_seconds` - 请求 Prefill 耗时。
- `vllm:request_decode_time_seconds` - 请求 Decode 耗时。
- `vllm:request_max_num_generation_tokens` - 序列组中的最大生成 Token 数。

更多有趣的关于指标选择的背景，可参见 [添加此仪表盘的 PR](https://github.com/vllm-project/vllm/pull/2316)。

### Prometheus 客户端库 (Prometheus Client Library)

最初是通过 [使用 aioprometheus 库](https://github.com/vllm-project/vllm/pull/1890) 来支持 Prometheus 的，但很快又切换到了 [prometheus_client](https://github.com/vllm-project/vllm/pull/2730)。相关的切换理由在上述两个 PR 中有详细讨论。

在那些迁移过程中，我们曾短暂丢失了用于跟踪 HTTP 指标的 `MetricsMiddleware`，但随后 [使用 prometheus_fastapi_instrumentator](https://github.com/vllm-project/vllm/pull/15657) 重新恢复了该功能：

```bash
$ curl http://0.0.0.0:8000/metrics 2>/dev/null  | grep -P '^http_(?!.*(_bucket|_created|_sum)).*'
http_requests_total{handler="/v1/completions",method="POST",status="2xx"} 201.0
http_request_size_bytes_count{handler="/v1/completions"} 201.0
http_response_size_bytes_count{handler="/v1/completions"} 201.0
http_request_duration_highr_seconds_count 201.0
http_request_duration_seconds_count{handler="/v1/completions",method="POST"} 201.0
```

### 多进程模式 (Multi-process Mode)

在过去，指标是在引擎核心（Core）进程中收集的，并使用多进程模式使其在 API 服务器进程中可用。参见 <https://github.com/vllm-project/vllm/pull/7279>。

而在最近，指标是在 API 服务器进程中收集的，并且仅在 `--api-server-count > 1` 时才使用多进程模式。参见 <https://github.com/vllm-project/vllm/pull/17546> 和 [API 服务器水平扩展 (API server scale-out)](../serving/data_parallel_deployment.md#internal-load-balancing) 中的细节。

### 内置的 Python/进程指标 (Built in Python/Process Metrics)

以下指标由 `prometheus_client` 默认支持，但在使用多进程模式时，它们**不会**被暴露出来：

- `python_gc_objects_collected_total`
- `python_gc_objects_uncollectable_total`
- `python_gc_collections_total`
- `python_info`
- `process_virtual_memory_bytes`
- `process_resident_memory_bytes`
- `process_start_time_seconds`
- `process_cpu_seconds_total`
- `process_open_fds`
- `process_max_fds`

因此，当 `--api-server-count > 1` 时，这些指标不可用。鉴于它们不聚合构成 vLLM 实例的所有进程的统计数据，这些指标的相关性也是存疑的。

## 指标设计 (Metrics Design)

在 ["Even Better Observability"](https://github.com/vllm-project/vllm/issues/3616) 功能提案中，规划了大部分的指标设计。例如，可以查看 [在此 laid out 的详细路线图](https://github.com/vllm-project/vllm/issues/3616#issuecomment-2030858781)。

### 遗留 PR 列表 (Legacy PRs)

为了便于理解指标设计的背景，这里列出了一些添加了原始指标（现在已废弃）的相关 PR：

- <https://github.com/vllm-project/vllm/pull/1890>
- <https://github.com/vllm-project/vllm/pull/2316>
- <https://github.com/vllm-project/vllm/pull/2730>
- <https://github.com/vllm-project/vllm/pull/4464>
- <https://github.com/vllm-project/vllm/pull/7279>

### 指标实现 PR 列表 (Metrics Implementation PRs)

供背景参考，这里是与指标实现相关的 PR：<https://github.com/vllm-project/vllm/issues/10582>：

- <https://github.com/vllm-project/vllm/pull/11962>
- <https://github.com/vllm-project/vllm/pull/11973>
- <https://github.com/vllm-project/vllm/pull/10907>
- <https://github.com/vllm-project/vllm/pull/12416>
- <https://github.com/vllm-project/vllm/pull/12478>
- <https://github.com/vllm-project/vllm/pull/12516>
- <https://github.com/vllm-project/vllm/pull/12530>
- <https://github.com/vllm-project/vllm/pull/12561>
- <https://github.com/vllm-project/vllm/pull/12579>
- <https://github.com/vllm-project/vllm/pull/12592>
- <https://github.com/vllm-project/vllm/pull/12644>

### 指标收集 (Metrics Collection)

在 V1 中，我们希望将计算和开销移出引擎核心进程，以尽量减少每次前向传播之间的时间间隔。

V1 版本的 `EngineCore` 总体设计思路为：

- `EngineCore` 是内层循环。这里的性能是最为关键的。
- `AsyncLLM` 是外层循环。理想情况下，它与 GPU 执行是重叠的（Overlap），因此如果可能的话，任何“开销”都应当放在这里。所以，`AsyncLLM.output_handler_loop` 是进行指标簿记（Bookkeeping）的理想场所。

我们将通过在前端 API 服务器中收集指标来实现这一目标，并使这些指标基于前端能够从引擎核心进程返回的 `EngineCoreOutputs` 中获取的信息。

### 间隔计算 (Interval Calculations)

我们的许多指标都是请求处理过程中各个事件之间的时间间隔。在计算时间间隔时，最佳实践是使用基于“单调时间（Monotonic Time）” (`time.monotonic()`) 的时间戳，而不是“挂钟时间（Wall-clock Time）” (`time.time()`)，因为前者不受系统时钟变更（例如通过 NTP）的影响。

同时需要注意的是，各个进程之间的单调时钟并不一致 —— 每个进程都有它自己的参考基准。因此，比较不同进程的单调时间戳是毫无意义的。

因此，为了计算时间间隔，我们必须比较来自同一个进程的两个单调时间戳。

### 调度器统计 (Scheduler Stats)

引擎核心进程将从调度器中收集一些关键统计数据 —— 例如，在最后一次调度 Pass 之后，正在调度或排队等待的请求数 —— 并将这些统计数据包含在 `EngineCoreOutputs` 中。

### 引擎核心事件 (Engine Core Events)

引擎核心还将记录某些逐请求事件的时间戳，以便前端可以计算这些事件之间的间隔。

这些事件包括：

- `QUEUED` — 引擎核心接收到请求并将其添加到调度器队列时。
- `SCHEDULED` — 请求首次被调度执行时。
- `PREEMPTED` — 请求被放回等待队列中，以腾出空间供其他请求完成。它将在未来重新被调度，并重新开始 Prefill 阶段。
- `NEW_TOKENS` — 生成了包含在 `EngineCoreOutput` 中的输出时。由于这在给定的迭代中是所有请求所共有的，我们使用 `EngineCoreOutputs` 上的单个时间戳来记录此事件。

计算得到的间隔为：

- 排队间隔 (Queue interval) — 在 `QUEUED` 与最近的 `SCHEDULED` 之间。
- Prefill 间隔 (Prefill interval) — 在最近的 `SCHEDULED` 与随后的首次 `NEW_TOKENS` 之间。
- Decode 间隔 (Decode interval) — 在首次（在最近的 `SCHEDULED` 之后）与最后的 `NEW_TOKENS` 之间。
- 推理间隔 (Inference interval) — 在最近的 `SCHEDULED` 与最后的 `NEW_TOKENS` 之间。
- Token 间隔 (Inter-token interval) — 在连续的 `NEW_TOKENS` 之间。

换言之：

![常规情况下的间隔计算](../assets/design/metrics/intervals-1.png)

我们曾探索过让前端使用其能够观测到的事件时间来计算这些间隔的可能性。然而，前端无法观测到 `QUEUED` 和 `SCHEDULED` 事件的发生时间，而且由于我们需要基于来自同一个进程的单调时间戳来计算间隔，我们需要引擎核心来记录所有这些事件的时间戳。

#### 间隔计算与抢占 (Interval Calculations vs Preemptions)

当在 Decode 阶段发生抢占时，由于任何已生成的 Token 都会被重用，我们认为抢占会影响 Token 间隔、Decode 间隔和推理间隔。

![抢占 Decode 阶段的间隔计算](../assets/design/metrics/intervals-2.png)

当在 Prefill 阶段发生抢占时（假设有可能发生此类事件），我们认为抢占会影响首字延迟（TTFT）和 Prefill 间隔。

![抢占 Prefill 阶段的间隔计算](../assets/design/metrics/intervals-3.png)

### 前端统计收集 (Frontend Stats Collection)

前端在处理单个 `EngineCoreOutputs`（即单次引擎核心迭代的输出）时，会收集与该次迭代相关的各种统计数据：

- 在本次迭代中生成的总 Token 数量。
- 在本次迭代中完成的 Prefill 步骤所处理的 Prompt Token 总数。
- 本次迭代中被调度的任何请求的排队间隔。
- 本次迭代中完成 Prefill 阶段的任何请求的 Prefill 间隔。
- 本次迭代中所包含的所有请求的逐字生成延迟（TPOT）。
- 在本次迭代中完成 Prefill 的任何请求的首字延迟（TTFT）。然而，为了将输入处理时间计算在内，我们相对于前端首次接收到请求的时间（`arrival_time`）来计算该间隔。目前 `arrival_time` 是在 Tokenization 开始时记录的。

对于在给定迭代中完成的任何请求，我们还会记录：

- 推理间隔和 Decode 间隔 —— 均相对于已调度和首个 Token 事件进行，如上所述。
- 端到端延迟 —— 前端 `arrival_time` 与前端接收到最后一个 Token 之间的时间间隔。

### KV 缓存驻留度量指标 (KV Cache Residency Metrics)

我们还发射了一组直方图，用于描述采样的 KV 缓存块驻留了多长时间，以及它们被重用的频率。通过采样（`--kv-cache-metrics-sample`）可以保持极小的开销；当选择一个块时，我们记录：

- `lifetime`（生命周期） – 分配 ⟶ 逐出（eviction）
- `idle before eviction`（逐出前闲置） – 最后一次访问 ⟶ 逐出
- `reuse gaps`（重用间隔） – 在重用块时，各次访问之间的停顿时间

这些会直接映射到以下 Prometheus 指标：

- `vllm:kv_block_lifetime_seconds` – 每个采样块存在的时间。
- `vllm:kv_block_idle_before_evict_seconds` – 最后一次访问后的空闲尾部时间。
- `vllm:kv_block_reuse_gap_seconds` – 连续访问之间的时间间隔。

引擎核心只通过 `SchedulerStats` 发送原始的 eviction 事件；前端会消耗它们，将其转换为 Prometheus 观测数据，并在日志记录开启时通过 `LLM.get_metrics()` 暴露相同的数据。在图表上同时查看生命周期和闲置时间，可以轻松发现闲置被困的缓存，或者为了长时间 Decode 而固定提示词的工作负载。

### 指标发布 - 日志 (Metrics Publishing - Logging)

`LoggingStatLogger` 指标发布器每 5 秒输出一条 `INFO` 日志信息，带有几个关键的指标：

- 当前运行/等待的请求数量。
- 当前的 GPU 缓存占用百分比。
- 过去 5 秒内，每秒处理的 Prompt Token 数量。
- 过去 5 秒内，每秒生成的 Token 数量。
- 针对最近 1k 个 KV 缓存块查询的前缀缓存命中率。

### 指标发布 - Prometheus (Metrics Publishing - Prometheus)

`PrometheusStatLogger` 指标发布器通过 Prometheus 兼容的格式，在 `/metrics` HTTP 端点上提供指标。随后，可以配置一个 Prometheus 实例来拉取（Poll）该端点（例如，每秒一次）并在其时间序列数据库中记录这些值。Prometheus 通常与 Grafana 搭配使用，从而允许这些指标随时间进行绘制。

Prometheus 支持以下指标类型：

- **Counter (计数器)**：一个随时间单调递增的值，不会减少，在 vLLM 实例重启时通常会重置为零。例如，在该实例的生命周期内生成的 Token 总数。
- **Gauge (瞬态仪)**：一个会上拉和下滑的值，例如当前被调度执行的请求数量。
- **Histogram (直方图)**：记录在不同桶（Buckets）中的指标样本计数。例如，TTFT 小于 1 毫秒、小于 5 毫秒、小于 10 毫秒、小于 20 毫秒等的请求数量。

Prometheus 指标也可以被标记标签（Labelled），从而允许根据匹配的标签来组合指标。在 vLLM 中，我们在每个指标中都添加了一个 `model_name` 标签，其中包含了该实例服务的模型名称。

输出示例：

```bash
$ curl http://0.0.0.0:8000/metrics
# HELP vllm:num_requests_running Number of requests in model execution batches.
# TYPE vllm:num_requests_running gauge
vllm:num_requests_running{model_name="meta-llama/Llama-3.1-8B-Instruct"} 8.0
...
# HELP vllm:generation_tokens_total Number of generation tokens processed.
# TYPE vllm:generation_tokens_total counter
vllm:generation_tokens_total{model_name="meta-llama/Llama-3.1-8B-Instruct"} 27453.0
...
# HELP vllm:request_success_total Count of successfully processed requests.
# TYPE vllm:request_success_total counter
vllm:request_success_total{finished_reason="stop",model_name="meta-llama/Llama-3.1-8B-Instruct"} 1.0
vllm:request_success_total{finished_reason="length",model_name="meta-llama/Llama-3.1-8B-Instruct"} 131.0
vllm:request_success_total{finished_reason="abort",model_name="meta-llama/Llama-3.1-8B-Instruct"} 0.0
...
# HELP vllm:time_to_first_token_seconds Histogram of time to first token in seconds.
# TYPE vllm:time_to_first_token_seconds histogram
vllm:time_to_first_token_seconds_bucket{le="0.001",model_name="meta-llama/Llama-3.1-8B-Instruct"} 0.0
vllm:time_to_first_token_seconds_bucket{le="0.005",model_name="meta-llama/Llama-3.1-8B-Instruct"} 0.0
vllm:time_to_first_token_seconds_bucket{le="0.01",model_name="meta-llama/Llama-3.1-8B-Instruct"} 0.0
vllm:time_to_first_token_seconds_bucket{le="0.02",model_name="meta-llama/Llama-3.1-8B-Instruct"} 13.0
vllm:time_to_first_token_seconds_bucket{le="0.04",model_name="meta-llama/Llama-3.1-8B-Instruct"} 97.0
vllm:time_to_first_token_seconds_bucket{le="0.06",model_name="meta-llama/Llama-3.1-8B-Instruct"} 123.0
vllm:time_to_first_token_seconds_bucket{le="0.08",model_name="meta-llama/Llama-3.1-8B-Instruct"} 138.0
vllm:time_to_first_token_seconds_bucket{le="0.1",model_name="meta-llama/Llama-3.1-8B-Instruct"} 140.0
vllm:time_to_first_token_seconds_count{model_name="meta-llama/Llama-3.1-8B-Instruct"} 140.0
```

!!! note "注意"
    选择在广泛的用例中对用户最有用处的直方图分桶配置并非显而易见，并且随着时间的推移需要进行不断的微调。

### 缓存配置信息 (Cache Config Info)

`prometheus_client` 提供了对 [Info metrics](https://prometheus.github.io/client_python/instrumenting/info/) 的支持，它们等同于一个值永久设置为 1 的 `Gauge`，但可以通过标签暴露出有趣的键值对信息。这适用于关于实例的不发生变化的信息 —— 因此仅需在启动时观测一次即可 —— 并允许在 Prometheus 中跨实例进行对比。

我们将此概念用于 `vllm:cache_config_info` 指标：

```text
# HELP vllm:cache_config_info Information of the LLMEngine CacheConfig
# TYPE vllm:cache_config_info gauge
vllm:cache_config_info{block_size="16",cache_dtype="auto",calculate_kv_scales="False",cpu_offload_gb="0",enable_prefix_caching="False",gpu_memory_utilization="0.9",...} 1.0
```

然而，出于[不甚明确的原因](gh-pr:7279#discussion_r1710417152)，`prometheus_client` [在多进程模式下从未支持过 Info metrics](https://github.com/prometheus/client_python/pull/300)。我们简单地使用一个设置为 1 且将 `multiprocess_mode` 设置为 `"mostrecent"` 的 `Gauge` 指标来代替。

### LoRA 相关指标

`vllm:lora_requests_info` `Gauge` 与之有些类似，唯一的区别是其值是当前的挂钟时间，并在每次迭代时更新。

所使用的标签名称为：

- `running_lora_adapters`：每个适配器（Adapter）上运行的使用该适配器的请求数量，格式为逗号分隔的字符串。
- `waiting_lora_adapters`：类似于前者，唯独统计的是正在等待被调度的请求数量。
- `max_lora`：静态配置的“单个批次中最大 LoRA 数量”。

将多个适配器的运行中/等待中计数编码在逗号分隔的字符串中似乎有些不太合理 —— 我们本可以使用标签来区分每个适配器的计数。这值得在今后重新审视。

注意，这里使用了 `multiprocess_mode="livemostrecent"` —— 即使用最接近现在的指标，但仅来自于当前运行的进程。

该功能是在 <https://github.com/vllm-project/vllm/pull/9477> 中添加的，并且[至少有一个已知的下游用户](https://github.com/kubernetes-sigs/gateway-api-inference-extension/pull/54)。如果我们重新审视该设计并废弃旧指标，我们应当与下游用户进行协调，以便他们能在该指标移除前完成迁移。

### 前缀缓存指标 (Prefix Cache metrics)

在 <https://github.com/vllm-project/vllm/issues/10582> 中关于添加前缀缓存指标的讨论产生了一些有意思的观点，这可能与我们未来处理指标的方式相关。

每次查询前缀缓存时，我们都会记录查询的 Token 数量以及在缓存中存在的已查询 Token 数量（即命中的 Token 数量）。

然而，我们最感兴趣的指标是命中率 —— 即每次查询的命中次数。

在日志记录方面，我们认为给用户提供在固定的最近查询次数上计算出的命中率是最合理的（目前这一间隔固定为最近的 1k 次查询）。

但是在 Prometheus 方面，我们应当充分利用 Prometheus 的时间序列特性，并允许用户根据他们自己选择的时间间隔来计算命中率。例如，一个 PromQL 查询，用于计算过去 5 分钟内的命中率：

```text
rate(cache_query_hit[5m]) / rate(cache_query_total[5m])
```

为了实现这一目的，我们在 Prometheus 中应当将查询数和命中数记录为 Counter 计数器，而不是将命中率记录为 Gauge。

## 被废弃的指标 (Deprecated Metrics)

### 如何废弃指标

废弃指标绝非一件微不足道的小事。用户可能注意不到一个指标已被废弃，当其在某天突然被移除时可能会感到非常不便，哪怕此时有另一个完全等价的指标可供他们使用。

例如，查看 `vllm:avg_prompt_throughput_toks_per_s` 如何[被废弃](https://github.com/vllm-project/vllm/pull/2764)（在代码中带有一条注释），[被移除](https://github.com/vllm-project/vllm/pull/12383)，随后[被一个用户注意到](https://github.com/vllm-project/vllm/issues/13218)。

总体而言：

1. 我们应当对废弃指标保持谨慎，因为对用户的影响往往很难预估。
2. 我们应当在 `/metrics` 输出中所包含的帮助字符串（Help string）中加入显眼的废弃说明。
3. 我们应当在面向用户的文档和发布说明（Release notes）中列出废弃的指标。
4. 我们应当考虑将废弃的指标隐藏在一个 CLI 参数后，以便为系统管理员在彻底删除指标之前提供[一条临时救急的缓和途径](https://kubernetes.io/docs/concepts/cluster-administration/system-metrics/#show-hidden-metrics)。

请参考整个项目范围内的[废弃政策](../contributing/deprecation_policy.md)。

### 未实现的指标 — `vllm:tokens_total`

由 <https://github.com/vllm-project/vllm/pull/4464> 添加，但似乎从未真正实现过。该指标可以直接移除。

### 重复的指标 — 排队时间 (Queue Time)

`vllm:time_in_queue_requests` 直方图指标由 <https://github.com/vllm-project/vllm/pull/9659> 添加，其计算逻辑如下：

```python
    self.metrics.first_scheduled_time = now
    self.metrics.time_in_queue = now - self.metrics.arrival_time
```

两周后，<https://github.com/vllm-project/vllm/pull/4464> 添加了 `vllm:request_queue_time_seconds`，导致留下：

```python
if seq_group.is_finished():
    if (seq_group.metrics.first_scheduled_time is not None and
            seq_group.metrics.first_token_time is not None):
        time_queue_requests.append(
            seq_group.metrics.first_scheduled_time -
            seq_group.metrics.arrival_time)
    ...
    if seq_group.metrics.time_in_queue is not None:
        time_in_queue_requests.append(
            seq_group.metrics.time_in_queue)
```

这看起来是重复的，其中一个应当被移除。后者已被 Grafana 仪表盘采用，因此我们应当废弃或移除前者。

### 前缀缓存命中率

见上文 —— 我们现在暴露的是 `queries` 查询数和 `hits` 命中数两个 Counter，而不是一个命中率的 Gauge。

### KV 缓存换出 (KV Cache Offloading)

有两个遗留的指标与一个在 V1 中已不再适用的“Swapped”抢占模式相关：

- `vllm:num_requests_swapped`
- `vllm:cpu_cache_usage_perc`

在过去，当一个请求被抢占（例如，为了在 KV 缓存中腾出空间以完成其他请求）时，KV 缓存块会被换出（Swap out）到 CPU 内存中。由于在 V1 中不再使用此功能，`--swap-space` 标志已被移除。

历史上，[vLLM 长期以来支持束搜索 (Beam Search)](https://github.com/vllm-project/vllm/issues/6226)。`SequenceGroup` 封装了共享同一个 Prompt KV 缓存块的 N 个序列这一概念。这实现了请求之间的 KV 缓存块共享，并通过写时复制（Copy-on-write）来进行分支。CPU 换出本就是为这些类似于束搜索的场景设计的。

此后，引入了前缀缓存的概念，它允许 KV 缓存块隐式地被共享。这被证明是一个比 CPU 换出更好的选择，因为块可以根据需求缓慢被淘汰，而被淘汰的提示词部分则可以被重新计算。

在 V1 中 `SequenceGroup` 被移除了，但在添加“并行采样” (`n>1`) 的支持时需要一个替代方案。[束搜索从核心中移出了](https://github.com/vllm-project/vllm/issues/8306)，那里曾经为了一个非常罕见的特性而存在了许多复杂的代码。

在 V1 中，由于前缀缓存更胜一筹（零额外开销）因而被默认启用，抢占和重新计算策略应该能更好地运作。

## 未来规划 (Future Work)

### 并行采样 (Parallel Sampling)

某些遗留指标只有在“并行采样”的背景下才具有相关性。在此情况下，一个请求中的 `n` 参数被用于请求来自同一个 Prompt 的多个补全。

作为在 <https://github.com/vllm-project/vllm/pull/10980> 中添加并行采样支持的一部分，我们也应当添加这些指标：

- `vllm:request_params_n` (Histogram) — 观测每个已完成请求的 `n` 参数的值。
- `vllm:request_max_num_generation_tokens` (Histogram) — 观测每个已完成序列组中所有序列的最大输出长度。在没有并行采样时，这等同于 `vllm:request_generation_tokens`。

### 推测解码 (Speculative Decoding)

某些遗留指标对于“推测解码”是特有的。在此情况下，我们使用更快、更近似的方法或模型生成候选 Token，随后使用大模型去验证这些 Token。

- `vllm:spec_decode_draft_acceptance_rate` (Gauge)
- `vllm:spec_decode_efficiency` (Gauge)
- `vllm:spec_decode_num_accepted_tokens` (Counter)
- `vllm:spec_decode_num_draft_tokens` (Counter)
- `vllm:spec_decode_num_emitted_tokens` (Counter)

目前有一个处于 Review 中的 PR (<https://github.com/vllm-project/vllm/pull/12193>) 旨在向 V1 添加基于“Prompt 查找 (Ngram)”的推测解码，后续还将支持其他技术。我们应当在此背景下重新审视这些指标。

!!! note "注意"
    我们可能应当将接受率（Acceptance rate）暴露为独立的 Accepted 和 Draft 两个 Counter，就像我们对前缀缓存命中率所做的那样。Efficiency 也需要类似的对待。

### 弹性伸缩和负载均衡 (Autoscaling and Load-balancing)

我们的指标最常见的一个使用场景就是支持 vLLM 实例的自动弹性伸缩（Autoscaling）。

有关 [Kubernetes Serving 工作组](https://github.com/kubernetes/community/tree/master/wg-serving) 的相关讨论，可参见：

- [在 Kubernetes 中标准化大模型服务器指标 (Standardizing Large Model Server Metrics in Kubernetes)](https://docs.google.com/document/d/1SpSp1E6moa4HSrJnS4x3NpLuj88sMXr2tbofKlzTZpk)
- [在 Kubernetes 中用于性能评估和自动伸缩的 LLM 工作负载基准测试 (Benchmarking LLM Workloads for Performance Evaluation and Autoscaling in Kubernetes)](https://docs.google.com/document/d/1k4Q4X14hW4vftElIuYGDu5KDe2LtV1XammoG-Xi3bbQ)
- [推理性能 (Inference Perf)](https://github.com/kubernetes-sigs/wg-serving/tree/main/proposals/013-inference-perf)
- <https://github.com/vllm-project/vllm/issues/5041> 与 <https://github.com/vllm-project/vllm/pull/12726>。

这是一个相当复杂的话题。考虑 Rob 的这一条评价：

> 我认为该指标应当聚焦于估算能够导致“平均请求长度 > 每秒查询数”的最大并发量……因为这正是真正能够使服务器达到“饱和”的原因。

一个明确的目标是，我们应当暴露检测此饱和点所需的指标，以便系统管理员能够在此基础上实施自动伸缩规则。然而为了做到这一点，我们需要对系统管理员（和自动化监控系统）应如何判断一个实例正趋近于饱和有一个清晰的视点：

> 如何确定模型服务器计算的饱和点（即在这一临界点上，我们无法通过更高的请求率获取更高的吞吐量，反而开始引入额外的延迟），以便我们进行有效的自动伸缩？

### 指标命名 (Metric Naming)

我们对指标的命名方法可能也值得被重新审视：

1. 在指标名称中使用冒号似乎与[“冒号被保留用于用户定义的记录规则”](https://prometheus.io/docs/concepts/data_model/#metric-names-and-labels)相违背。
2. 我们的绝大多数指标都遵循了以单位结尾的约定，但也并非完全如此。
3. 我们的某些指标名称以 `_total` 结尾：
   如果指标名称的后缀为 `_total`，它会被自动移除。而当为 Counter 暴露时间序列时，一个 `_total` 后缀又会被自动添加。这是为了 OpenMetrics 和 Prometheus 文本格式之间的兼容性，因为 OpenMetrics 要求有 `_total` 后缀。

### 添加更多指标

对于新指标的设想我们从不缺乏：

- 其他项目（例如 [TGI](https://github.com/IBM/text-generation-inference?tab=readme-ov-file#metrics)）中的例子。
- 产生于特定使用场景下的提案，例如上述 Kubernetes 弹性伸缩话题。
- 产生于标准化努力（例如 [OpenTelemetry 对 Gen AI 的语义公约](https://github.com/open-telemetry/semantic-conventions/tree/main/docs/gen-ai)）的提案。

在添加新指标时我们应当采取谨慎的态度。虽然指标在添加时通常非常直接：

1. 它们可能会很难被移除 —— 参见上文的废弃说明。
2. 在启用时，它们可能会带来明显的性能影响。并且，指标通常在能于生产中被默认启用时才具有真正的价值。
3. 它们对项目的开发和维护有负面影响。随着时间的推移，每个新增的指标都使这项工作更加耗费时间，也许并非所有的指标都配得上这一针对其维护的持续投入。

## 追踪监控 - OpenTelemetry (Tracing - OpenTelemetry)

指标提供了关于系统性能和健康状态在一段时间内的聚合视图。而追踪（Tracing）则跟踪单个请求在不同服务和组件中穿梭时的足迹。两者都属于更广泛的“可观测性”范畴。

vLLM 提供了对 OpenTelemetry 追踪的支持：

- 由 <https://github.com/vllm-project/vllm/pull/4687> 添加并由 <https://github.com/vllm-project/vllm/pull/20372> 恢复。
- 使用 `--oltp-traces-endpoint` 和 `--collect-detailed-traces` 进行配置。
- [OpenTelemetry 相关博客](https://opentelemetry.io/blog/2024/llm-observability/)。
- [用户文档](../../examples/observability/opentelemetry/README.md)。
- [相关 Medium 文章](https://medium.com/@ronen.schaffer/follow-the-trail-supercharging-vllm-with-opentelemetry-distributed-tracing-aa655229b46f)。
- [IBM 产品文档](https://www.ibm.com/docs/en/instana-observability/current?topic=mgaa-monitoring-large-language-models-llms-vllm-public-preview)。

OpenTelemetry 成立了 [Gen AI 工作组](https://github.com/open-telemetry/community/blob/main/projects/gen-ai.md)。

由于指标自身已是一个足够庞大的话题，我们认为追踪这个话题是与指标截然分开的。

### OpenTelemetry 模型前向传播 vs 执行时间

当前的实现暴露了以下两个指标：

- `vllm:model_forward_time_milliseconds` (Histogram) — 当此请求处于批次中时，花费在模型前向传播中的时间。
- `vllm:model_execute_time_milliseconds` (Histogram) — 花费在模型执行函数（Model Execute）中的时间。这将包括模型前向传播、跨 Worker 的 Block/Sync（块/同步）、CPU-GPU 同步时间以及采样时间。

这些指标仅在启用了 OpenTelemetry 追踪且使用了 `--collect-detailed-traces=all/model/worker` 时被激活。关于此选项的文档指出：

> 为指定的模块收集详细的 Traces。这涉及到使用可能会消耗大量资源或导致阻塞的操作，因此可能会对性能产生影响。

这些指标是由 <https://github.com/vllm-project/vllm/pull/7089> 添加的，在 OpenTelemetry 追踪中显示如下：

```text
-> gen_ai.latency.time_in_scheduler: Double(0.017550230026245117)
-> gen_ai.latency.time_in_model_forward: Double(3.151565277099609)
-> gen_ai.latency.time_in_model_execute: Double(3.6468167304992676)
```

鉴于我们已经有了 `inference_time` 和 `decode_time` 指标，问题在于高分辨率耗时计算是否存在足够普遍的使用场景，以支持引入此开销的合理性。

由于我们将单独对待 OpenTelemetry 支持这一议题，我们会将这些特定的指标归入该主题下讨论。
