#!/usr/bin/env bash
set -euo pipefail

PORT="${PORT:-8000}"

if ! command -v lsof >/dev/null 2>&1; then
  echo "lsof is required to find the vLLM server listening on port ${PORT}." >&2
  exit 1
fi

PIDS="$(lsof -nP -tiTCP:"${PORT}" -sTCP:LISTEN || true)"

if [[ -z "${PIDS}" ]]; then
  echo "No process is listening on TCP port ${PORT}."
  exit 0
fi

MATCHED=()
while IFS= read -r pid; do
  [[ -z "${pid}" ]] && continue
  command_line="$(ps -p "${pid}" -o command= || true)"
  if [[ "${command_line}" == *"vllm serve"* ]]; then
    MATCHED+=("${pid}")
  else
    echo "Refusing to stop PID ${pid}; it is not a vLLM server:"
    echo "  ${command_line}"
  fi
done <<< "${PIDS}"

if [[ "${#MATCHED[@]}" -eq 0 ]]; then
  echo "No vLLM server process found on TCP port ${PORT}."
  exit 1
fi

echo "Stopping vLLM server on TCP port ${PORT}: ${MATCHED[*]}"
kill "${MATCHED[@]}"

for _ in {1..20}; do
  if ! lsof -nP -tiTCP:"${PORT}" -sTCP:LISTEN >/dev/null 2>&1; then
    echo "vLLM server stopped."
    exit 0
  fi
  sleep 0.5
done

echo "vLLM server did not stop after 10 seconds; sending SIGKILL."
kill -9 "${MATCHED[@]}" 2>/dev/null || true
echo "vLLM server stopped."

