# vLLM 学习与本地服务器笔记

本笔记专为本机定制：Apple Silicon macOS。Python 3.13 可在 `/opt/homebrew/bin/python3.13` 路径下获取；vLLM-Metal 目前需要原生 arm64 的 Python 3.12，本机目前尚未安装。

## 什么是 vLLM

vLLM 是一个用于大语言模型的推理与服务引擎。它的实用价值在于通过以下技术高效地服务大量请求：

- 用于 KV 缓存内存管理的 PagedAttention。
- 用于在线服务吞吐量的连续批处理（Continuous batching）。
- 前缀缓存（Prefix caching）、分块预填充（Chunked prefill）、投机解码（Speculative decoding）和量化（Quantization）。
- 通过 `vllm serve` 提供的兼容 OpenAI 接口的 HTTP 服务器。

本地重要文件：

- `README.md`：项目概述与主要功能。
- `docs/getting_started/quickstart.md`：首个离线与在线示例。
- `docs/serving/online_serving/openai_compatible_server.md`：HTTP API 使用方法。
- `docs/configuration/serve_args.md`：服务器配置参考。
- `vllm/entrypoints/cli/main.py`：CLI 注册。
- `vllm/entrypoints/cli/serve.py`：`vllm serve` 调度路径。
- `vllm/entrypoints/openai/api_server.py`：兼容 OpenAI 的服务器。

## 推荐学习路径

1. 阅读服务心智模型。

   `vllm serve <model>` 会启动一个本地 HTTP 服务器，通常位于 `http://localhost:8000`，并且一次只服务一个模型。

2. 了解兼容 OpenAI 的 API。

   常用的端点包括：

   - `GET /v1/models`
   - `POST /v1/chat/completions`
   - `POST /v1/completions`
   - `GET /health`
   - `GET /metrics`

3. 使用小模型进行实践。

   在配备 NVIDIA GPU 的 Linux 上，从以下命令开始：

   ```bash
   vllm serve Qwen/Qwen2.5-1.5B-Instruct
   ```

   在这台 Mac 上，推荐使用 vLLM-Metal：

   ```bash
   source ~/.venv-vllm-metal/bin/activate
   vllm serve mlx-community/Qwen2.5-0.5B-Instruct-4bit
   ```

4. 使用 OpenAI SDK 语义调用服务器。

   服务器运行后，使用 `local_notes/vllm_chat_client.py`。

5. 仅在第一个请求成功工作后再进行调优。

   常用参数：

   - `--host 127.0.0.1`
   - `--port 8000`
   - `--api-key token-abc123`
   - `--served-model-name local-qwen`
   - `--max-model-len 4096`
   - `--generation-config vllm`
   - `--dtype auto`

## Apple Silicon 的选择

### 最佳本地路径：vLLM-Metal

vLLM 主仓库通过独立的社区包 `vllm-metal`（使用 MLX 和 Metal）来支持 Apple GPU 加速。请使用来自 `mlx-community` 的经过 MLX 优化的模型。

当前 vLLM-Metal 的要求：

- 搭载 Apple Silicon 芯片的 macOS。
- 原生 arm64 Python 3.12。
- Xcode 命令行工具。

建议尝试的首个模型：

```text
mlx-community/Qwen2.5-0.5B-Instruct-4bit
```

如果需要，先安装 Python 3.12：

```bash
brew install python@3.12
```

然后安装 vLLM-Metal：

```bash
curl -fsSL https://raw.githubusercontent.com/vllm-project/vllm-metal/main/install.sh | bash
```

### CPU 源码构建路径

主仓库包含实验性的 macOS CPU 支持。目前它需要从源码构建，且运行速度比 Metal 慢。请使用 Python 3.13，如果系统的 `python3` 指向 Python 3.14，请勿使用系统默认的。

```bash
/opt/homebrew/bin/python3.13 -m venv .venv-cpu
source .venv-cpu/bin/activate
python -m pip install --upgrade pip uv
uv pip install -r requirements/cpu.txt --index-strategy unsafe-best-match
uv pip install -e .
```

然后运行：

```bash
export VLLM_CPU_KVCACHE_SPACE=4
export VLLM_CPU_NUM_OF_RESERVED_CPU=1
vllm serve facebook/opt-125m --dtype=float32 --host 127.0.0.1 --port 8000
```

## 服务器启动方案

### Mac Metal 服务器

```bash
local_notes/start_vllm_metal_server.sh
```

### 通用 CUDA 服务器

```bash
vllm serve Qwen/Qwen2.5-1.5B-Instruct \
  --host 127.0.0.1 \
  --port 8000 \
  --api-key token-abc123 \
  --generation-config vllm
```

### CPU 学习服务器

```bash
export VLLM_CPU_KVCACHE_SPACE=4
export VLLM_CPU_NUM_OF_RESERVED_CPU=1
vllm serve facebook/opt-125m --dtype=float32 --host 127.0.0.1 --port 8000
```

## 客户端调用

列出模型：

```bash
curl http://localhost:8000/v1/models
```

健康检查：

```bash
curl http://localhost:8000/health
```

聊天请求：

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-community/Qwen2.5-0.5B-Instruct-4bit",
    "messages": [{"role": "user", "content": "用三句话解释 vLLM。"}],
    "max_tokens": 128,
    "temperature": 0.2
  }'
```

如果服务器启动时带了 `--api-key token-abc123` 参数，请添加：

```bash
-H "Authorization: Bearer token-abc123"
```

## 下一步学习内容

- `docs/design/paged_attention_zh.md`：为什么 vLLM 以不同的方式管理 KV 缓存。
- `docs/design/arch_overview_zh.md`：核心架构。
- `docs/features/automatic_prefix_caching_zh.md`：重复提示词优化（Prefix Caching）。
- `docs/features/quantization/README_zh.md`：内存与吞吐量权衡（量化）。
- `docs/features/lora_zh.md`：适配器服务（LoRA）。
- `docs/usage/metrics_zh.md`：运维指标与可见性。
