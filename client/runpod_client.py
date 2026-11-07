#!/usr/bin/env python3
"""Minimal client for the RunPod endpoint. Standard library only.

    export RUNPOD_API_KEY=...  RUNPOD_ENDPOINT_ID=...
    python client/runpod_client.py "Explain Tailscale in one line"          # /runsync
    python client/runpod_client.py --async "Write a haiku about NVMe"       # /run + poll
    python client/runpod_client.py --openai "Hello"                         # /openai/v1/chat/completions
    python client/runpod_client.py --health                                 # workers / queue

As a library:
    from runpod_client import chat
    print(chat([{"role": "user", "content": "hi"}]))
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request

API = "https://api.runpod.ai/v2"


def _cfg() -> tuple[str, str]:
    key, endpoint = os.getenv("RUNPOD_API_KEY"), os.getenv("RUNPOD_ENDPOINT_ID")
    if not key or not endpoint:
        sys.exit("Set RUNPOD_API_KEY and RUNPOD_ENDPOINT_ID")
    return key, endpoint


def _request(method: str, path: str, body: dict | None = None, timeout: float = 120) -> dict:
    key, endpoint = _cfg()
    req = urllib.request.Request(
        f"{API}/{endpoint}{path}",
        data=json.dumps(body).encode() if body is not None else None,
        method=method,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"HTTP {e.code}: {e.read().decode(errors='replace')}") from None


def _unwrap(output):
    """The worker yields once for non-streaming calls; aggregate mode wraps it in a list."""
    if isinstance(output, list) and len(output) == 1:
        output = output[0]
    if isinstance(output, dict) and "error" in output:
        raise RuntimeError(output["error"].get("message", output["error"]))
    return output


def _text(resp: dict) -> str:
    choice = resp["choices"][0]
    return choice.get("message", {}).get("content") or choice.get("text", "")


def run_sync(job_input: dict, timeout: float = 300) -> dict:
    """POST /runsync. Waits up to ~90 s server-side, then falls back to polling."""
    res = _request("POST", "/runsync", {"input": job_input}, timeout=timeout)
    if res.get("status") in ("IN_QUEUE", "IN_PROGRESS"):  # cold start outlasted runsync
        return poll(res["id"], timeout=timeout)
    return res


def run_async(job_input: dict) -> str:
    """POST /run; returns the job id."""
    return _request("POST", "/run", {"input": job_input})["id"]


def poll(job_id: str, timeout: float = 600, interval: float = 2) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        res = _request("GET", f"/status/{job_id}")
        if res["status"] in ("COMPLETED", "FAILED", "CANCELLED", "TIMED_OUT"):
            return res
        time.sleep(interval)
    raise TimeoutError(f"job {job_id} still running after {timeout}s")


def chat(messages: list[dict], **params) -> str:
    res = run_sync({"messages": messages, **params})
    if res.get("status") != "COMPLETED":
        raise RuntimeError(f"job {res.get('status')}: {res.get('error') or res}")
    return _text(_unwrap(res["output"]))


def openai_chat(messages: list[dict], model: str = "runpod-qwen3-14b", **params) -> str:
    """Same call through RunPod's OpenAI-compatible route (what Open WebUI uses)."""
    body = {"model": model, "messages": messages, **params}
    return _text(_request("POST", "/openai/v1/chat/completions", body, timeout=600))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("prompt", nargs="?", default="Say hello from RunPod in five words.")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--async", dest="use_async", action="store_true", help="use /run + /status polling")
    mode.add_argument("--openai", action="store_true", help="use /openai/v1/chat/completions")
    mode.add_argument("--health", action="store_true", help="show worker and queue counts")
    ap.add_argument("--model", default=os.getenv("RUNPOD_MODEL", "runpod-qwen3-14b"))
    ap.add_argument("--max-tokens", type=int, default=256)
    args = ap.parse_args()

    if args.health:
        print(json.dumps(_request("GET", "/health"), indent=2))
        return

    messages = [{"role": "user", "content": args.prompt}]
    t0 = time.monotonic()
    if args.openai:
        out = openai_chat(messages, model=args.model, max_tokens=args.max_tokens)
    elif args.use_async:
        job_id = run_async({"messages": messages, "max_tokens": args.max_tokens})
        print(f"queued {job_id}", file=sys.stderr)
        res = poll(job_id)
        if res["status"] != "COMPLETED":
            sys.exit(f"job {res['status']}: {res.get('error')}")
        out = _text(_unwrap(res["output"]))
    else:
        out = chat(messages, max_tokens=args.max_tokens)
    print(out)
    print(f"({time.monotonic() - t0:.1f}s)", file=sys.stderr)


if __name__ == "__main__":
    main()
