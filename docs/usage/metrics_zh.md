# 生产环境指标 (Production Metrics)

vLLM 暴露了许多可用于监控系统健康状况的指标。这些指标通过 vLLM OpenAI 兼容 API 服务器上的 `/metrics` 端点进行展示。

您可以使用 Python 或使用 [Docker](../deployment/docker.md) 来启动服务器：

```bash
vllm serve unsloth/Llama-3.2-1B-Instruct
```

然后查询该端点以从服务器获取最新的指标：

??? console "输出示例"

    ```console
    $ curl http://0.0.0.0:8000/metrics

    # HELP vllm:iteration_tokens_total Histogram of number of tokens per engine_step.
    # TYPE vllm:iteration_tokens_total histogram
    vllm:iteration_tokens_total_sum{model_name="unsloth/Llama-3.2-1B-Instruct"} 0.0
    vllm:iteration_tokens_total_bucket{le="1.0",model_name="unsloth/Llama-3.2-1B-Instruct"} 3.0
    vllm:iteration_tokens_total_bucket{le="8.0",model_name="unsloth/Llama-3.2-1B-Instruct"} 3.0
    vllm:iteration_tokens_total_bucket{le="16.0",model_name="unsloth/Llama-3.2-1B-Instruct"} 3.0
    vllm:iteration_tokens_total_bucket{le="32.0",model_name="unsloth/Llama-3.2-1B-Instruct"} 3.0
    vllm:iteration_tokens_total_bucket{le="64.0",model_name="unsloth/Llama-3.2-1B-Instruct"} 3.0
    vllm:iteration_tokens_total_bucket{le="128.0",model_name="unsloth/Llama-3.2-1B-Instruct"} 3.0
    vllm:iteration_tokens_total_bucket{le="256.0",model_name="unsloth/Llama-3.2-1B-Instruct"} 3.0
    vllm:iteration_tokens_total_bucket{le="512.0",model_name="unsloth/Llama-3.2-1B-Instruct"} 3.0
    ...
    ```

以下是暴露的具体指标：

## 通用指标 (General Metrics)

--8<-- "docs/generated/metrics/general.inc.md"

## 投机解码指标 (Speculative Decoding Metrics)

--8<-- "docs/generated/metrics/spec_decode.inc.md"

## NIXL KV 连接器指标 (NIXL KV Connector Metrics)

--8<-- "docs/generated/metrics/nixl_connector.inc.md"

## 模型 FLOPS 利用率 (MFU) 性能指标

这些指标在通过 `--enable-mfu-metrics` 启用后可用：

--8<-- "docs/generated/metrics/perf.inc.md"

## 弃用政策 (Deprecation Policy)

注意：当指标在版本 `X.Y` 中被弃用时，它们将在版本 `X.Y+1` 中被隐藏，但可以通过使用 `--show-hidden-metrics-for-version=X.Y` 参数来重新启用，并最终在版本 `X.Y+2` 中被彻底移除。
