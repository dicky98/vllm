#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:-mlx-community/Qwen2.5-0.5B-Instruct-4bit}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
VLLM_HOST_IP="${VLLM_HOST_IP:-127.0.0.1}"
export VLLM_HOST_IP

if [[ -f "${HOME}/.venv-vllm-metal/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "${HOME}/.venv-vllm-metal/bin/activate"
else
  echo "Missing ~/.venv-vllm-metal. Install vLLM-Metal first." >&2
  echo "See: https://github.com/vllm-project/vllm-metal#installation" >&2
  exit 1
fi

exec vllm serve "${MODEL}" --host "${HOST}" --port "${PORT}"
