#!/usr/bin/env python3
"""Fail-closed C1 correctness, code-decode, and cold-prefill acceptance."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
import time
import urllib.request
from typing import Any


CODE_PROMPTS = {
    "python": (
        "Implement a production-quality Python LRU cache with type hints, tests, "
        "complexity notes, and thread-safety discussion. Keep writing useful code "
        "and tests until the completion budget is exhausted."
    ),
    "rust": (
        "Implement a production-quality bounded work-stealing queue in Rust with "
        "tests, error handling, and safety commentary. Keep writing useful code "
        "and tests until the completion budget is exhausted."
    ),
    "typescript": (
        "Implement a production-quality TypeScript JSON-RPC client with retries, "
        "cancellation, schema validation, and tests. Keep writing useful code and "
        "tests until the completion budget is exhausted."
    ),
    "cuda_cpp": (
        "Implement and explain a production-quality CUDA C++ tiled reduction "
        "kernel with a host harness, validation, and benchmark. Keep writing useful "
        "code until the completion budget is exhausted."
    ),
    "go": (
        "Implement a production-quality bounded concurrent worker pool in Go with "
        "context cancellation, backpressure, and tests. Keep writing useful code "
        "and tests until the completion budget is exhausted."
    ),
}


def post_json(url: str, payload: dict[str, Any], timeout: int = 1800) -> Any:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def stream_chat(
    base_url: str,
    model: str,
    prompt: str,
    completion_budget: int,
    response_format: dict[str, Any] | None = None,
) -> tuple[str, dict[str, Any]]:
    payload: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
        "max_completion_tokens": completion_budget,
        "chat_template_kwargs": {"thinking": False},
    }
    if response_format is not None:
        payload["response_format"] = response_format

    started = time.perf_counter()
    first_content_at: float | None = None
    last_content_at: float | None = None
    usage: dict[str, Any] = {}
    finish_reason: str | None = None
    content_parts: list[str] = []
    request = urllib.request.Request(
        f"{base_url}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=1800) as response:
        for raw_line in response:
            line = raw_line.decode("utf-8").strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            event = json.loads(line[6:])
            if event.get("usage"):
                usage = event["usage"]
            choices = event.get("choices") or []
            if not choices:
                continue
            choice = choices[0]
            finish_reason = choice.get("finish_reason") or finish_reason
            delta = choice.get("delta") or {}
            piece = delta.get("content") or ""
            if piece:
                now = time.perf_counter()
                first_content_at = first_content_at or now
                last_content_at = now
                content_parts.append(piece)
    ended = time.perf_counter()
    if first_content_at is None or last_content_at is None:
        raise RuntimeError("stream ended without a content token")

    prompt_tokens = int(usage["prompt_tokens"])
    completion_tokens = int(usage["completion_tokens"])
    decode_seconds = max(last_content_at - first_content_at, 1e-9)
    details = usage.get("prompt_tokens_details") or {}
    metrics = {
        "prompt_tokens": prompt_tokens,
        "cached_tokens": int(details.get("cached_tokens") or 0),
        "completion_tokens": completion_tokens,
        "ttft_s": first_content_at - started,
        "prefill_tok_s": prompt_tokens / (first_content_at - started),
        "decode_tok_s": max(completion_tokens - 1, 0) / decode_seconds,
        "elapsed_s": ended - started,
        "finish_reason": finish_reason,
    }
    return "".join(content_parts), metrics


def token_count(base_url: str, model: str, content: str) -> int:
    response = post_json(
        f"{base_url}/tokenize",
        {
            "model": model,
            "messages": [{"role": "user", "content": content}],
            "add_special_tokens": True,
        },
    )
    return int(response["count"])


def build_exact_prefill_prompt(base_url: str, model: str, target: int) -> str:
    nonce = time.time_ns()
    prefix = (
        f"Cold-prefill acceptance nonce {nonce}. Read all filler tokens. "
        "The final instruction is authoritative.\n"
    )
    suffix = "\nReply with exactly: PREFILL OK."
    low, high = 0, target
    while low <= high:
        middle = (low + high) // 2
        count = token_count(base_url, model, prefix + (" x" * middle) + suffix)
        if count < target:
            low = middle + 1
        elif count > target:
            high = middle - 1
        else:
            return prefix + (" x" * middle) + suffix
    prompt = prefix + (" x" * high) + suffix
    count = token_count(base_url, model, prompt)
    if count != target:
        raise RuntimeError(f"could not construct {target}-token prompt; got {count}")
    return prompt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="deepseek-v4-flash-0731-spark")
    parser.add_argument("--code-completion-tokens", type=int, default=512)
    parser.add_argument("--cold-prefill-tokens", type=int, default=252047)
    parser.add_argument("--skip-correctness", action="store_true")
    parser.add_argument("--skip-code", action="store_true")
    parser.add_argument("--skip-prefill", action="store_true")
    args = parser.parse_args()

    result: dict[str, Any] = {
        "date_unix_ns": time.time_ns(),
        "base_url": args.base_url,
        "model": args.model,
        "thinking": False,
    }

    failures: list[str] = []
    if not args.skip_correctness:
        semantic_text, semantic_metrics = stream_chat(
            args.base_url,
            args.model,
            "Solve precisely. Return only the decimal integer equal to 17 multiplied by 19.",
            32,
        )
        semantic_normalized = semantic_text.strip()
        if re.search(r"(?:^|\D)323\.?$", semantic_normalized) is None:
            failures.append(f"semantic gate failed: {semantic_text!r}")
        result["semantic"] = {
            "status": "pass" if not failures else "fail",
            "text": semantic_text,
            **semantic_metrics,
        }

        schema = {
            "type": "json_schema",
            "json_schema": {
                "name": "structured_acceptance",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "language": {"type": "string", "const": "Python"},
                        "answer": {"type": "integer", "const": 323},
                        "valid": {"type": "boolean", "const": True},
                    },
                    "required": ["language", "answer", "valid"],
                    "additionalProperties": False,
                },
            },
        }
        structured_text, structured_metrics = stream_chat(
            args.base_url,
            args.model,
            "Return the required object for Python, with answer 17 * 19 and valid true.",
            64,
            response_format=schema,
        )
        structured = json.loads(structured_text)
        expected = {"language": "Python", "answer": 323, "valid": True}
        if structured != expected:
            failures.append(f"structured gate failed: {structured!r}")
        result["structured_json_schema"] = {
            "status": "pass" if structured == expected else "fail",
            "parsed": structured,
            **structured_metrics,
        }

    if not args.skip_code:
        trials = []
        for language, prompt in CODE_PROMPTS.items():
            content, metrics = stream_chat(
                args.base_url,
                args.model,
                prompt,
                args.code_completion_tokens,
            )
            trials.append(
                {
                    "language": language,
                    "content_sha256": hashlib.sha256(content.encode()).hexdigest(),
                    "content_prefix": content[:120],
                    **metrics,
                }
            )
        speeds = [float(trial["decode_tok_s"]) for trial in trials]
        result["code_c1_decode"] = {
            "trials": trials,
            "minimum_tok_s": min(speeds),
            "median_tok_s": statistics.median(speeds),
            "mean_tok_s": statistics.fmean(speeds),
            "gate_tok_s": 35.0,
            "gate": "pass" if min(speeds) >= 35.0 else "fail",
        }
        if min(speeds) < 35.0:
            failures.append(f"decode gate failed: {min(speeds):.3f} tok/s")

    if not args.skip_prefill:
        prompt = build_exact_prefill_prompt(
            args.base_url, args.model, args.cold_prefill_tokens
        )
        content, metrics = stream_chat(args.base_url, args.model, prompt, 16)
        if content.strip() != "PREFILL OK.":
            failures.append(f"cold-prefill correctness failed: {content!r}")
        if metrics["prompt_tokens"] != args.cold_prefill_tokens:
            failures.append(f"cold-prefill token count drifted: {metrics!r}")
        if metrics["cached_tokens"] != 0:
            failures.append(f"cold-prefill was cached: {metrics!r}")
        result["cold_c1_prefill"] = {
            "response": content,
            **metrics,
            "gate_tok_s": 1000.0,
            "gate": "pass" if metrics["prefill_tok_s"] >= 1000.0 else "fail",
        }
        if metrics["prefill_tok_s"] < 1000.0:
            failures.append(
                f"prefill gate failed: {metrics['prefill_tok_s']:.3f} tok/s"
            )

    print(json.dumps(result, indent=2))
    if failures:
        raise SystemExit("; ".join(failures))


if __name__ == "__main__":
    main()
