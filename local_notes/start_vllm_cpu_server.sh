#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:-facebook/opt-125m}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
DTYPE="${DTYPE:-float32}"

export VLLM_CPU_KVCACHE_SPACE="${VLLM_CPU_KVCACHE_SPACE:-4}"
export VLLM_CPU_NUM_OF_RESERVED_CPU="${VLLM_CPU_NUM_OF_RESERVED_CPU:-1}"

exec vllm serve "${MODEL}" --dtype "${DTYPE}" --host "${HOST}" --port "${PORT}"

