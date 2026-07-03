#!/usr/bin/env python3
"""Small OpenAI-compatible client for a local vLLM server."""

import argparse

import httpx
from openai import OpenAI


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--model", default="mlx-community/Qwen2.5-0.5B-Instruct-4bit")
    parser.add_argument("--prompt", default="用三句话解释 vLLM 适合解决什么问题。")
    parser.add_argument("--max-tokens", type=int, default=160)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--stream", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    client = OpenAI(
        base_url=args.base_url,
        api_key=args.api_key,
        http_client=httpx.Client(trust_env=False),
    )

    model = args.model
    if not model:
        models = client.models.list()
        if not models.data:
            raise RuntimeError("The server did not return any served models.")
        model = models.data[0].id

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": args.prompt}],
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        stream=args.stream,
    )

    if args.stream:
        for chunk in response:
            delta = chunk.choices[0].delta.content
            if delta:
                print(delta, end="", flush=True)
        print()
    else:
        print(response.choices[0].message.content)


if __name__ == "__main__":
    main()
