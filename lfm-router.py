#!/usr/bin/env python3
"""
LFM Smart Router — lightweight local-first LLM proxy with latency sensing.

Strategy:
  Routes EVERYTHING to LFM (Ollama localhost:11434) first — LFM handles
  both chat AND tool calling.
  If LFM is slower than LFM_TIMEOUT_SEC, falls through to OpenRouter.
  Tracks a sliding window of LFM latencies. If the rolling average stays
  above SLOW_THRESHOLD_SEC, skips LFM and routes straight to OpenRouter
  until a periodic probe detects recovery.

  Zero third-party deps — pure Python stdlib.

Usage:
  python3 lfm-router.py [--port 20130]

Environment:
  OPENROUTER_API_KEY   (required for fallback)
  LFM_ROUTER_PORT      (default 20130)
  LFM_TIMEOUT_SEC      (default 90 — per-request timeout for LFM)
  SLOW_THRESHOLD_SEC   (default 60 — rolling avg above this = skip LFM)
  PROBE_INTERVAL       (default 300 — seconds between recovery probes)
  OPENROUTER_MODEL     (default openrouter/auto)
  NVIDIA_API_KEY       (required for NVIDIA NIM free tier)
  NVIDIA_MODEL         (default nvidia/llama-3.3-nemotron-super-49b-v1)
  NVIDIA_BASE          (default https://integrate.api.nvidia.com/v1)
  GROQ_API_KEY         (required for Groq free tier)
  GROQ_MODEL           (default llama-3.3-70b-versatile)
  GROQ_BASE            (default https://api.groq.com/openai/v1)
  GOOGLE_API_KEY       (required for Google Gemini)
  GOOGLE_MODEL         (default gemini-3-flash-preview)
  GOOGLE_BASE          (default https://generativelanguage.googleapis.com/v1beta/openai)
"""

import http.server
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request
from collections import deque


# ── config ────────────────────────────────────────────────────────────────

PORT = int(os.environ.get("LFM_ROUTER_PORT", "20130"))
LFM_TIMEOUT_SEC = float(os.environ.get("LFM_TIMEOUT_SEC", "90"))
SLOW_THRESHOLD_SEC = float(os.environ.get("SLOW_THRESHOLD_SEC", "60"))
HEALTH_WINDOW = int(os.environ.get("HEALTH_WINDOW", "6"))
PROBE_INTERVAL_SEC = float(os.environ.get("PROBE_INTERVAL", "300"))
OPENROUTER_BASE = "https://openrouter.ai/api/v1"
OPENROUTER_MODEL = os.environ.get("OPENROUTER_MODEL", "openrouter/auto")
NVIDIA_BASE = os.environ.get("NVIDIA_BASE", "https://integrate.api.nvidia.com/v1")
NVIDIA_MODEL = os.environ.get("NVIDIA_MODEL", "nvidia/llama-3.3-nemotron-super-49b-v1")
GROQ_BASE = os.environ.get("GROQ_BASE", "https://api.groq.com/openai/v1")
GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
GOOGLE_BASE = os.environ.get("GOOGLE_BASE", "https://generativelanguage.googleapis.com/v1beta/openai")
GOOGLE_MODEL = os.environ.get("GOOGLE_MODEL", "gemini-3-flash-preview")
OLLAMA_BASE = os.environ.get("OLLAMA_BASE", "http://localhost:11434/v1")
LFM_MODEL = os.environ.get("LFM_MODEL", "lfm2.5-2.6b-hermes:latest")
OLLAMA_TIMEOUT_SEC = float(os.environ.get("OLLAMA_TIMEOUT_SEC", "90"))

# ── state ─────────────────────────────────────────────────────────────────

_lfm_latencies = deque(maxlen=HEALTH_WINDOW)
_last_probe_at = 0.0
_lock = threading.Lock()


def rolling_avg() -> float:
    with _lock:
        if not _lfm_latencies:
            return 0.0
        return sum(_lfm_latencies) / len(_lfm_latencies)


def record_latency(seconds: float):
    with _lock:
        _lfm_latencies.append(seconds)


def should_skip_lfm() -> bool:
    """Return True when rolling average says LFM is slow."""
    avg = rolling_avg()
    return avg > SLOW_THRESHOLD_SEC


# ── provider calls ────────────────────────────────────────────────────────

def call_ollama(body: dict, model: str | None = None) -> tuple[int, dict, dict]:
    """POST to LFM (or specified model) on Ollama.

    Uses the NATIVE /api/chat endpoint — the OpenAI-compatible endpoint
    loses the real answer for thinking models (LFM2.5 puts the final
    response in `content` + reasoning in `thinking`, but /v1/chat/completions
    only exposes the reasoning). We translate the native response back
    into OpenAI chat-completions shape.
    """
    target_model = model or LFM_MODEL
    timeout = OLLAMA_TIMEOUT_SEC
    native_body = {
        "model": target_model,
        "messages": body.get("messages", []),
        "stream": False,
        "options": {"num_predict": body.get("max_tokens", 4096)},
    }
    # forward tools so LFM can do tool calling
    if body.get("tools"):
        native_body["tools"] = body["tools"]
    if body.get("temperature") is not None:
        native_body["options"]["temperature"] = body["temperature"]
    data = json.dumps(native_body).encode()
    req = urllib.request.Request(
        "http://localhost:11434/api/chat",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        t0 = time.monotonic()
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            elapsed = time.monotonic() - t0
            raw = resp.read()
        native = json.loads(raw)
        record_latency(elapsed)
        # Translate native response to OpenAI chat-completions shape
        msg = native.get("message", {})
        parsed = {
            "id": f"chatcmpl-lfm-{int(time.time())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": native.get("model", target_model),
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": msg.get("content") or "",
                },
                "finish_reason": native.get("done_reason", "stop"),
            }],
            "usage": {
                "prompt_tokens": native.get("prompt_eval_count", 0),
                "completion_tokens": native.get("eval_count", 0),
                "total_tokens": (native.get("prompt_eval_count", 0)
                                 + native.get("eval_count", 0)),
            },
        }
        # keep thinking as reasoning if present (some consumers like it)
        if msg.get("thinking"):
            parsed["choices"][0]["message"]["reasoning"] = msg["thinking"]
        # Thinking model may burn all tokens on reasoning — if content is
        # empty and reasoning exists, promote reasoning → content so the
        # response isn't empty.
        if not parsed["choices"][0]["message"].get("content", "") and msg.get("thinking"):
            parsed["choices"][0]["message"]["content"] = msg["thinking"]
        # translate tool_calls to OpenAI format
        if msg.get("tool_calls"):
            tc = []
            for t in msg["tool_calls"]:
                fn = t.get("function", {})
                # native API returns arguments as object, OpenAI expects string
                args = fn.get("arguments", {})
                if isinstance(args, dict):
                    args = json.dumps(args)
                tc.append({
                    "id": t.get("id", f"call_{int(time.time())}"),
                    "type": "function",
                    "function": {
                        "name": fn.get("name", ""),
                        "arguments": args,
                    },
                })
            parsed["choices"][0]["message"]["tool_calls"] = tc
        return (200, parsed, {})
    except urllib.error.HTTPError as e:
        elapsed = time.monotonic() - t0
        record_latency(elapsed)
        body_text = e.read()
        try:
            parsed = json.loads(body_text)
        except Exception:
            parsed = {"error": {"message": body_text.decode(), "code": e.code}}
        return (e.code, parsed, dict(e.headers))
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        record_latency(LFM_TIMEOUT_SEC)
        return (503, {"error": {"message": f"LFM error: {e}", "code": 503}}, {})


def call_openrouter(body: dict) -> tuple[int, dict, dict]:
    """POST to OpenRouter fallback."""
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        return (502, {"error": {"message": "OPENROUTER_API_KEY not set", "code": 502}}, {})
    body["model"] = OPENROUTER_MODEL
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        f"{OPENROUTER_BASE}/chat/completions",
        data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            raw = resp.read()
        parsed = json.loads(raw)
        return (resp.status, parsed, dict(resp.headers))
    except urllib.error.HTTPError as e:
        body_text = e.read()
        try:
            parsed = json.loads(body_text)
        except Exception:
            parsed = {"error": {"message": body_text.decode(), "code": e.code}}
        return (e.code, parsed, dict(e.headers))
    except Exception as e:
        return (503, {"error": {"message": f"OpenRouter error: {e}", "code": 503}}, {})


def call_openai_compat(base: str, model: str, api_key: str, body: dict, name: str) -> tuple[int, dict, dict]:
    """POST to any OpenAI-compatible endpoint."""
    if not api_key:
        return (502, {"error": {"message": f"{name} key not set", "code": 502}}, {})
    body["model"] = model
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        f"{base}/chat/completions",
        data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            raw = resp.read()
        parsed = json.loads(raw)
        return (resp.status, parsed, dict(resp.headers))
    except urllib.error.HTTPError as e:
        body_text = e.read()
        try:
            parsed = json.loads(body_text)
        except Exception:
            parsed = {"error": {"message": body_text.decode(), "code": e.code}}
        return (e.code, parsed, dict(e.headers))
    except Exception as e:
        return (503, {"error": {"message": f"{name} error: {e}", "code": 503}}, {})


def call_nvidia(body: dict) -> tuple[int, dict, dict]:
    """POST to NVIDIA NIM free tier."""
    return call_openai_compat(NVIDIA_BASE, NVIDIA_MODEL, os.environ.get("NVIDIA_API_KEY", ""), body, "NVIDIA")


def call_groq(body: dict) -> tuple[int, dict, dict]:
    """POST to Groq free tier."""
    return call_openai_compat(GROQ_BASE, GROQ_MODEL, os.environ.get("GROQ_API_KEY", ""), body, "GROQ")


def call_google(body: dict) -> tuple[int, dict, dict]:
    """POST to Google Gemini (OpenAI-compatible endpoint)."""
    return call_openai_compat(GOOGLE_BASE, GOOGLE_MODEL, os.environ.get("GOOGLE_API_KEY", ""), body, "GOOGLE")


# ── probe ─────────────────────────────────────────────────────────────────

def probe_lfm():
    """Quick heartbeat via native API — if LFM responds fast, good; otherwise penalise window."""
    global _last_probe_at
    t0 = time.monotonic()
    try:
        probe_body = json.dumps({
            "model": LFM_MODEL,
            "messages": [{"role": "user", "content": "ping"}],
            "stream": False,
            "options": {"num_predict": 1},
        }).encode()
        req = urllib.request.Request(
            "http://localhost:11434/api/chat",
            data=probe_body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            _ = resp.read()
        elapsed = time.monotonic() - t0
        record_latency(elapsed)
    except Exception:
        record_latency(LFM_TIMEOUT_SEC)
    _last_probe_at = time.monotonic()


# ── HTTP handler ───────────────────────────────────────────────────────────

class RouterHandler(http.server.BaseHTTPRequestHandler):

    def log_message(self, fmt, *args):
        sys.stderr.write(f"[lfm-router] {args[0]}\n")

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.end_headers()

    def do_GET(self):
        if self.path in ("/health", "/v1/health"):
            avg = rolling_avg()
            status = json.dumps({
                "status": "ok",
                "lfm_rolling_avg_s": round(avg, 1),
                "lfm_samples": len(_lfm_latencies),
                "last_probe_at": _last_probe_at,
                "config": {
                    "lfm_model": LFM_MODEL,
                    "lfm_timeout_s": LFM_TIMEOUT_SEC,
                    "slow_threshold_s": SLOW_THRESHOLD_SEC,
                    "nvidia_model": NVIDIA_MODEL,
                    "groq_model": GROQ_MODEL,
                    "google_model": GOOGLE_MODEL,
                    "openrouter_model": OPENROUTER_MODEL,
                    "health_window": HEALTH_WINDOW,
                    "probe_interval_s": PROBE_INTERVAL_SEC,
                },
            })
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(status.encode())
            return
        self.send_response(404)
        self.end_headers()

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length)
        try:
            body = json.loads(raw)
        except json.JSONDecodeError:
            self._respond(400, {"error": {"message": "invalid JSON", "code": 400}})
            return

        t0 = time.monotonic()
        route = "LFM"

        if should_skip_lfm():
            # LFM rolling average too slow — skip straight to the free
            # fallback chain (NEVER straight to OpenRouter)
            route = "NVIDIA (LFM slow)"
            status, data, headers = call_nvidia(body)
            if status >= 400 or status == 503:
                route = "NVIDIA→Groq (LFM slow)"
                status, data, headers = call_groq(body)
                if status >= 400 or status == 503:
                    route = "NVIDIA→Groq→Google (LFM slow)"
                    status, data, headers = call_google(body)
                    if status >= 400 or status == 503:
                        route = "NVIDIA→Groq→Google→OpenRouter (LFM slow)"
                        status, data, headers = call_openrouter(body)
        else:
            # Tier 1: LFM (local, fast, handles chat + tools)
            status, data, headers = call_ollama(body)
            if status >= 400 or status == 503:
                # Tier 2: NVIDIA NIM free
                route = "LFM→NVIDIA"
                status, data, headers = call_nvidia(body)
                if status >= 400 or status == 503:
                    # Tier 3: Groq free
                    route = "LFM→NVIDIA→Groq"
                    status, data, headers = call_groq(body)
                    if status >= 400 or status == 503:
                        # Tier 4: Google Gemini
                        route = "LFM→NVIDIA→Groq→Google"
                        status, data, headers = call_google(body)
                        if status >= 400 or status == 503:
                            # Tier 5: OpenRouter (final escape hatch)
                            route = "LFM→NVIDIA→Groq→Google→OpenRouter"
                            status, data, headers = call_openrouter(body)

        elapsed = time.monotonic() - t0

        # Inject routing metadata
        if isinstance(data, dict) and "usage" in data:
            data["_routing"] = {
                "route": route,
                "total_elapsed_s": round(elapsed, 1),
                "lfm_rolling_avg_s": round(rolling_avg(), 1),
                "lfm_samples": len(_lfm_latencies),
            }

        self._respond(status, data)

    def _respond(self, status: int, data):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())


# ── periodic probe thread ──────────────────────────────────────────────────

def probe_loop():
    urllib.request.install_opener(urllib.request.build_opener(urllib.request.ProxyHandler({})))
    while True:
        time.sleep(PROBE_INTERVAL_SEC)
        try:
            probe_lfm()
            avg = rolling_avg()
            print(f"[lfm-router] probe — avg {avg:.1f}s", file=sys.stderr, flush=True)
        except Exception as e:
            print(f"[lfm-router] probe error: {e}", file=sys.stderr, flush=True)


# ── main ───────────────────────────────────────────────────────────────────

def main():
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    urllib.request.install_opener(opener)

    try:
        probe_lfm()
    except Exception:
        pass

    t = threading.Thread(target=probe_loop, daemon=True)
    t.start()

    server = http.server.HTTPServer(("127.0.0.1", PORT), RouterHandler)
    print(f"[lfm-router] listening on 127.0.0.1:{PORT}", file=sys.stderr, flush=True)
    print(f"[lfm-router]   Primary (T1):  {LFM_MODEL} @ {OLLAMA_BASE}", file=sys.stderr, flush=True)
    print(f"[lfm-router]   Fallback (T2): {NVIDIA_MODEL} @ NVIDIA NIM", file=sys.stderr, flush=True)
    print(f"[lfm-router]   Fallback (T3): {GROQ_MODEL} @ Groq", file=sys.stderr, flush=True)
    print(f"[lfm-router]   Fallback (T4): {GOOGLE_MODEL} @ Google Gemini", file=sys.stderr, flush=True)
    print(f"[lfm-router]   Fallback (T5): {OPENROUTER_MODEL} @ OpenRouter", file=sys.stderr, flush=True)
    print(f"[lfm-router]   Timeout: {OLLAMA_TIMEOUT_SEC}s", file=sys.stderr, flush=True)
    print(f"[lfm-router]   Slow threshold: {SLOW_THRESHOLD_SEC}s ({HEALTH_WINDOW} samples)", file=sys.stderr, flush=True)
    print(f"[lfm-router]   Probe every {PROBE_INTERVAL_SEC}s", file=sys.stderr, flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[lfm-router] shutting down", file=sys.stderr, flush=True)
        server.server_close()


if __name__ == "__main__":
    main()