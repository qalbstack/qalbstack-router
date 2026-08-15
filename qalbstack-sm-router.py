#!/usr/bin/env python3
"""
Qalbstack Smart Router 2.1

Stable OpenAI-compatible endpoint for Hermes + all Qalbstack cron jobs.

Selectable local models:
  T1a  LFM2.5-2.6B
  T1b  Qwen3.8-27B

Internal fallback cascade:
  NVIDIA -> Groq -> Google -> OpenRouter

Important design:
  - The public /v1/models catalog contains ONLY selectable local models.
  - Cloud providers are internal fallback tiers, not selectable "models".
  - Tool definitions are preserved on every tier.
  - Local Ollama requests receive num_ctx.
  - OpenAI-compatible cloud providers do NOT receive Ollama-specific
    num_ctx/context_length fields.
  - Hermes can select the preferred local model by sending its model ID.
  - Unknown/legacy model IDs fall back to the configured default local model,
    preserving compatibility with existing cron jobs.
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
from urllib.parse import urlparse


# ── Configuration ──────────────────────────────────────────────────────────

PORT = int(os.environ.get("SM_ROUTER_PORT", "20130"))

LFM_TIMEOUT_SEC = float(os.environ.get("LFM_TIMEOUT_SEC", "120"))
SLOW_THRESHOLD_SEC = float(os.environ.get("SLOW_THRESHOLD_SEC", "60"))
HEALTH_WINDOW = int(os.environ.get("HEALTH_WINDOW", "6"))
PROBE_INTERVAL_SEC = float(os.environ.get("PROBE_INTERVAL", "300"))
PROBE_TIMEOUT_SEC = float(os.environ.get("PROBE_TIMEOUT", "30"))

DEFAULT_CONTEXT = int(os.environ.get("SM_ROUTER_CONTEXT", "65536"))
MAX_OUTPUT = int(os.environ.get("SM_ROUTER_MAX_OUTPUT", "16384"))

OLLAMA_BASE = os.environ.get(
    "OLLAMA_BASE",
    "http://localhost:11434",
)

LFM_MODEL = os.environ.get(
    "LFM_MODEL",
    "lfm2.5-2.6b",
)

QWEN_MODEL = os.environ.get(
    "QWEN_MODEL",
    "hf.co/unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_XL",
)

# This is the local model used when an old/legacy client sends an
# unrecognised model ID (for example openrouter/auto from the old config).
AUTO_MODEL = "api"

DEFAULT_LOCAL_MODEL = os.environ.get(
    "SM_ROUTER_LOCAL_MODEL",
    LFM_MODEL,
)

OPENROUTER_BASE = "https://openrouter.ai/api/v1"
OPENROUTER_MODEL = os.environ.get(
    "OPENROUTER_MODEL",
    "openrouter/auto",
)

NVIDIA_BASE = os.environ.get(
    "NVIDIA_BASE",
    "https://integrate.api.nvidia.com/v1",
)
NVIDIA_MODEL = os.environ.get(
    "NVIDIA_MODEL",
    "nvidia/llama-3.3-nemotron-super-49b-v1",
)

GROQ_BASE = os.environ.get(
    "GROQ_BASE",
    "https://api.groq.com/openai/v1",
)
GROQ_MODEL = os.environ.get(
    "GROQ_MODEL",
    "llama-3.3-70b-versatile",
)

GOOGLE_BASE = os.environ.get(
    "GOOGLE_BASE",
    "https://generativelanguage.googleapis.com/v1beta/openai",
)
GOOGLE_MODEL = os.environ.get(
    "GOOGLE_MODEL",
    "gemini-3-flash-preview",
)

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
NVIDIA_API_KEY = os.environ.get("NVIDIA_API_KEY", "")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")


# ── State ──────────────────────────────────────────────────────────────────

_lfm_latencies = deque(maxlen=HEALTH_WINDOW)
_last_probe_at = 0.0
_lock = threading.Lock()
_log_lock = threading.Lock()


# ── Logging ─────────────────────────────────────────────────────────────────

def log(msg: str):
    with _log_lock:
        ts = time.strftime("%H:%M:%S")
        print(
            f"[{ts}] sm-router: {msg}",
            file=sys.stderr,
            flush=True,
        )


# ── Latency helpers ────────────────────────────────────────────────────────

def rolling_avg() -> float:
    with _lock:
        if not _lfm_latencies:
            return 0.0
        return sum(_lfm_latencies) / len(_lfm_latencies)


def record_latency(seconds: float):
    with _lock:
        _lfm_latencies.append(seconds)


def should_skip_lfm() -> bool:
    return rolling_avg() > SLOW_THRESHOLD_SEC


# ── Model helpers ──────────────────────────────────────────────────────────

LOCAL_MODELS = (
    AUTO_MODEL,
    LFM_MODEL,
    QWEN_MODEL,
)


def auto_select_model(body: dict) -> str:
    """
    Deterministic local Auto policy.

    LFM is the fast/default model.
    Qwen is selected for requests that strongly suggest deeper reasoning,
    substantial analysis, larger code/design work, or unusually large tool
    payloads.

    This is deliberately transparent rather than using an extra LLM as a
    hidden classifier.
    """
    score = 0

    messages = body.get("messages") or []
    user_text = ""

    for message in reversed(messages):
        if message.get("role") == "user":
            content = message.get("content", "")
            if isinstance(content, str):
                user_text = content
            break

    text = user_text.lower()

    if len(user_text) >= 2000:
        score += 2
    elif len(user_text) >= 800:
        score += 1

    complex_terms = (
        "reason through",
        "deep reasoning",
        "reasoning",
        "analyze",
        "analysis",
        "architecture",
        "architect",
        "debug",
        "debugging",
        "compare",
        "comparison",
        "evaluate",
        "evaluation",
        "design",
        "derive",
        "prove",
        "logic puzzle",
        "multi-step",
        "step by step",
        "research",
        "investigate",
        "trade-off",
        "tradeoff",
        "root cause",
        "code review",
        "refactor",
    )

    score += sum(2 for term in complex_terms if term in text)

    if "```" in user_text:
        score += 2

    tools = body.get("tools") or []
    if len(tools) >= 8:
        score += 2

    # Qwen for genuinely complex requests; LFM otherwise.
    if score >= 4:
        return QWEN_MODEL

    return LFM_MODEL


def resolve_local_model(requested_model: str) -> tuple[str, str]:
    """
    Resolve a requested model into one of the selectable local models.

    Returns:
        (actual_model, display_label)
    """
    requested_model = str(requested_model or "").strip()

    if requested_model == QWEN_MODEL:
        return QWEN_MODEL, "Qwen"

    if requested_model == LFM_MODEL:
        return LFM_MODEL, "LFM"

    # Legacy compatibility:
    # Existing Hermes/cron configurations may still send openrouter/auto,
    # provider aliases, or other old IDs. Those should not bypass the router.
    return DEFAULT_LOCAL_MODEL, (
        "Qwen" if DEFAULT_LOCAL_MODEL == QWEN_MODEL else "LFM"
    )


# ── Context helpers ────────────────────────────────────────────────────────

def requested_context(body: dict) -> int:
    """
    Get the client's requested context.

    Hermes can represent this as num_ctx, context_length, or
    context_window. If absent, use the router's 65K default.
    """
    for key in (
        "num_ctx",
        "context_length",
        "context_window",
    ):
        value = body.get(key)

        if isinstance(value, int) and value > 0:
            return value

    return DEFAULT_CONTEXT


def requested_max_tokens(body: dict) -> int:
    value = body.get(
        "max_tokens",
        MAX_OUTPUT,
    )

    try:
        value = int(value)
    except (TypeError, ValueError):
        value = MAX_OUTPUT

    return max(
        1,
        min(value, MAX_OUTPUT),
    )


# ── HTTP helpers ────────────────────────────────────────────────────────────

def _post(
    url: str,
    body: dict,
    api_key: str = "",
    timeout: float = 120,
) -> tuple[int, dict, float]:

    data = json.dumps(body).encode()

    headers = {
        "Content-Type": "application/json",
        "User-Agent": "qalbstack-sm-router/2.1",
    }

    if api_key:
        headers["Authorization"] = (
            f"Bearer {api_key}"
        )

    req = urllib.request.Request(
        url,
        data=data,
        headers=headers,
        method="POST",
    )

    t0 = time.monotonic()

    try:
        with urllib.request.urlopen(
            req,
            timeout=timeout,
        ) as resp:
            raw = resp.read()

        elapsed = time.monotonic() - t0

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            parsed = {
                "error": {
                    "message": raw.decode(
                        errors="replace"
                    ),
                    "code": resp.status,
                }
            }

        return (
            resp.status,
            parsed,
            elapsed,
        )

    except urllib.error.HTTPError as e:
        elapsed = time.monotonic() - t0
        raw = e.read()

        try:
            parsed = json.loads(raw)
        except Exception:
            parsed = {
                "error": {
                    "message": raw.decode(
                        errors="replace"
                    ),
                    "code": e.code,
                }
            }

        return (
            e.code,
            parsed,
            elapsed,
        )

    except urllib.error.URLError as e:
        elapsed = time.monotonic() - t0

        return (
            503,
            {
                "error": {
                    "message": (
                        f"Connection failed: {e.reason}"
                    ),
                    "code": 503,
                }
            },
            elapsed,
        )

    except TimeoutError:
        elapsed = time.monotonic() - t0

        return (
            504,
            {
                "error": {
                    "message": "Upstream timeout",
                    "code": 504,
                }
            },
            elapsed,
        )

    except OSError as e:
        elapsed = time.monotonic() - t0

        return (
            503,
            {
                "error": {
                    "message": f"OS error: {e}",
                    "code": 503,
                }
            },
            elapsed,
        )


# ── T1: Local Ollama ───────────────────────────────────────────────────────

def call_local(body: dict) -> tuple[int, dict, float, str]:
    requested_model = str(
        body.get("model") or DEFAULT_LOCAL_MODEL
    ).strip()

    model, route_label = resolve_local_model(
        requested_model
    )

    context = requested_context(body)

    native = {
        "model": model,
        "messages": body.get(
            "messages",
            [],
        ),
        "stream": False,
        "options": {
            "num_predict": requested_max_tokens(
                body
            ),
            "num_ctx": context,
        },
    }

    # Preserve tools exactly as Hermes provided them.
    if body.get("tools"):
        native["tools"] = body["tools"]

    if body.get("temperature") is not None:
        native["options"]["temperature"] = body[
            "temperature"
        ]

    if body.get("top_p") is not None:
        native["options"]["top_p"] = body[
            "top_p"
        ]

    data = json.dumps(native).encode()

    req = urllib.request.Request(
        f"{OLLAMA_BASE}/api/chat",
        data=data,
        headers={
            "Content-Type": "application/json"
        },
        method="POST",
    )

    t0 = time.monotonic()

    try:
        with urllib.request.urlopen(
            req,
            timeout=LFM_TIMEOUT_SEC,
        ) as resp:
            raw = resp.read()

        elapsed = time.monotonic() - t0

        # Keep the historical latency window for LFM.
        # Qwen is not used by the LFM latency skip logic.
        if model == LFM_MODEL:
            record_latency(elapsed)

        native_resp = json.loads(raw)

        msg = native_resp.get(
            "message",
            {},
        )

        parsed = {
            "id": (
                f"chatcmpl-sm-{int(time.time())}"
            ),
            "object": "chat.completion",
            "created": int(time.time()),
            "model": native_resp.get(
                "model",
                model,
            ),
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": (
                            msg.get("content")
                            or ""
                        ),
                    },
                    "finish_reason": (
                        native_resp.get(
                            "done_reason",
                            "stop",
                        )
                    ),
                }
            ],
            "usage": {
                "prompt_tokens": native_resp.get(
                    "prompt_eval_count",
                    0,
                ),
                "completion_tokens": native_resp.get(
                    "eval_count",
                    0,
                ),
                "total_tokens": (
                    native_resp.get(
                        "prompt_eval_count",
                        0,
                    )
                    + native_resp.get(
                        "eval_count",
                        0,
                    )
                ),
            },
        }

        # Preserve model reasoning/thinking.
        if msg.get("thinking"):
            parsed["choices"][0]["message"][
                "reasoning"
            ] = msg["thinking"]

            if not msg.get("content"):
                parsed["choices"][0]["message"][
                    "content"
                ] = msg["thinking"]

        # Translate Ollama native tool calls to OpenAI.
        if msg.get("tool_calls"):
            tool_calls = []

            for tool_call in msg[
                "tool_calls"
            ]:
                function = tool_call.get(
                    "function",
                    {},
                )

                arguments = function.get(
                    "arguments",
                    {},
                )

                if isinstance(arguments, dict):
                    arguments = json.dumps(
                        arguments
                    )

                tool_calls.append(
                    {
                        "id": tool_call.get(
                            "id",
                            f"call_{int(time.time())}",
                        ),
                        "type": "function",
                        "function": {
                            "name": function.get(
                                "name",
                                "",
                            ),
                            "arguments": arguments,
                        },
                    }
                )

            parsed["choices"][0]["message"][
                "tool_calls"
            ] = tool_calls

        return (
            200,
            parsed,
            elapsed,
            route_label,
        )

    except urllib.error.HTTPError as e:
        elapsed = time.monotonic() - t0

        if model == LFM_MODEL:
            record_latency(elapsed)

        raw = e.read()

        try:
            parsed = json.loads(raw)
        except Exception:
            parsed = {
                "error": {
                    "message": raw.decode(
                        errors="replace"
                    ),
                    "code": e.code,
                }
            }

        error_message = ""
        if isinstance(parsed, dict):
            err = parsed.get("error")
            if isinstance(err, dict):
                error_message = str(
                    err.get("message") or ""
                ).replace("\\n", " ")[:300]
            elif parsed.get("message"):
                error_message = str(
                    parsed["message"]
                ).replace("\\n", " ")[:300]

        if error_message:
            log(
                f"✗ {route_label} → HTTP {e.code} "
                f"({elapsed:.1f}s) "
                f"error={error_message!r}"
            )

        return (
            e.code,
            parsed,
            elapsed,
            route_label,
        )

    except (
        urllib.error.URLError,
        TimeoutError,
        OSError,
    ) as e:
        elapsed = time.monotonic() - t0

        if model == LFM_MODEL:
            record_latency(
                max(
                    elapsed,
                    LFM_TIMEOUT_SEC,
                )
            )

        return (
            503,
            {
                "error": {
                    "message": (
                        f"{route_label} error: {e}"
                    ),
                    "code": 503,
                }
            },
            elapsed,
            route_label,
        )

    except json.JSONDecodeError:
        elapsed = time.monotonic() - t0

        if model == LFM_MODEL:
            record_latency(elapsed)

        return (
            502,
            {
                "error": {
                    "message": (
                        f"{route_label} returned "
                        "unparseable response"
                    ),
                    "code": 502,
                }
            },
            elapsed,
            route_label,
        )


# ── Tiers 2–5: OpenAI-compatible fallbacks ─────────────────────────────────

def call_openai(
    url: str,
    model: str,
    api_key: str,
    body: dict,
    tier_name: str,
    timeout: float = 120,
) -> tuple[int, dict, float, str]:

    if not api_key:
        return (
            502,
            {
                "error": {
                    "message": (
                        f"{tier_name} API key not set"
                    ),
                    "code": 502,
                }
            },
            0,
            tier_name,
        )

    # Preserve the OpenAI-compatible request fields such as messages,
    # tools, tool_choice, response_format, etc.
    #
    # Remove Ollama/router-specific fields that cloud providers reject.
    # The router handles streaming itself, so upstream tiers always receive
    # one completed non-streaming request.
    outbound = dict(body)

    for key in (
        "num_ctx",
        "context_length",
        "context_window",
        "options",
        "stream_options",
    ):
        outbound.pop(key, None)

    outbound["stream"] = False
    outbound["model"] = model

    code, data, elapsed = _post(
        f"{url}/chat/completions",
        outbound,
        api_key,
        timeout,
    )

    return (
        code,
        data,
        elapsed,
        tier_name,
    )


def call_nvidia(body: dict):
    return call_openai(
        NVIDIA_BASE,
        NVIDIA_MODEL,
        NVIDIA_API_KEY,
        body,
        "NVIDIA",
    )


def call_groq(body: dict):
    return call_openai(
        GROQ_BASE,
        GROQ_MODEL,
        GROQ_API_KEY,
        body,
        "Groq",
    )


def call_google(body: dict):
    return call_openai(
        GOOGLE_BASE,
        GOOGLE_MODEL,
        GOOGLE_API_KEY,
        body,
        "Google",
    )


def call_openrouter(body: dict):
    return call_openai(
        OPENROUTER_BASE,
        OPENROUTER_MODEL,
        OPENROUTER_API_KEY,
        body,
        "OpenRouter",
        timeout=180,
    )


# ── Internal cascade ───────────────────────────────────────────────────────

FALLBACK_TIERS = [
    ("NVIDIA", call_nvidia),
    ("Groq", call_groq),
    ("Google", call_google),
    ("OpenRouter", call_openrouter),
]


def route_request(body: dict):

    t_start = time.monotonic()

    requested_model = str(
        body.get("model") or AUTO_MODEL
    ).strip()

    requested_lower = requested_model.lower()

    if requested_lower in {
        AUTO_MODEL,
        "openrouter/auto",
        "qalbstack-auto",
        "auto",
    }:
        preferred_model = auto_select_model(body)
        preferred_label = (
            "Qwen-Auto"
            if preferred_model == QWEN_MODEL
            else "LFM-Auto"
        )
    else:
        preferred_model, preferred_label = resolve_local_model(
            requested_model
        )

    working_body = dict(body)
    working_body["model"] = preferred_model

    log_history = []
    last_error_data = None
    last_error_code = 503

    # LFM gets latency-based skipping because its latency is tracked.
    # Qwen always gets an actual attempt first when selected.
    try_local = True

    if (
        preferred_model == LFM_MODEL
        and should_skip_lfm()
    ):
        try_local = False
        log(
            f"LFM slow (avg {rolling_avg():.1f}s) "
            "— skipping local tier to NVIDIA"
        )

    if try_local:
        t_try = time.monotonic()

        try:
            (
                code,
                data,
                elapsed,
                label,
            ) = call_local(working_body)

        except Exception as e:
            code = 503
            data = {
                "error": {
                    "message": f"Unhandled: {e}",
                    "code": 503,
                }
            }
            elapsed = time.monotonic() - t_try
            label = preferred_label

        log_history.append(
            {
                "tier": label,
                "status": code,
                "elapsed_s": round(
                    elapsed,
                    1,
                ),
            }
        )

        if code < 400:
            total_elapsed = (
                time.monotonic() - t_start
            )

            if isinstance(data, dict):
                data["_routing"] = {
                    "route": label,
                    "preferred_model": preferred_model,
                    "total_elapsed_s": round(
                        total_elapsed,
                        1,
                    ),
                    "tiers": log_history,
                    "lfm_rolling_avg_s": round(
                        rolling_avg(),
                        1,
                    ),
                    "context_requested": (
                        requested_context(body)
                    ),
                    "tools_present": bool(
                        body.get("tools")
                    ),
                }

            log(
                f"✓ {label} "
                f"({total_elapsed:.1f}s)"
            )

            return (
                code,
                data,
                log_history,
            )

        error_message = ""
        if isinstance(data, dict):
            err = data.get("error")
            if isinstance(err, dict):
                error_message = str(
                    err.get("message") or ""
                ).replace("\n", " ")[:300]

        if error_message:
            log(
                f"✗ {label} → HTTP {code} "
                f"({elapsed:.1f}s) "
                f"error={error_message!r}"
            )
        else:
            log(
                f"✗ {label} → HTTP {code} "
                f"({elapsed:.1f}s)"
            )

        last_error_data = data
        last_error_code = code

    # Internal fallback cascade.
    for tier_name, tier_fn in FALLBACK_TIERS:
        t_try = time.monotonic()

        try:
            code, data, elapsed, label = tier_fn(
                body
            )

        except Exception as e:
            code = 503
            data = {
                "error": {
                    "message": f"Unhandled: {e}",
                    "code": 503,
                }
            }
            elapsed = time.monotonic() - t_try
            label = tier_name

        log_history.append(
            {
                "tier": tier_name,
                "status": code,
                "elapsed_s": round(
                    elapsed,
                    1,
                ),
            }
        )

        if code >= 400:
            error_message = ""
            if isinstance(data, dict):
                err = data.get("error")
                if isinstance(err, dict):
                    error_message = str(
                        err.get("message") or ""
                    ).replace("\\n", " ")[:300]
                elif data.get("message"):
                    error_message = str(
                        data["message"]
                    ).replace("\\n", " ")[:300]

            if error_message:
                log(
                    f"✗ {tier_name} → HTTP {code} "
                    f"({elapsed:.1f}s) "
                    f"error={error_message!r}"
                )
            else:
                log(
                    f"✗ {tier_name} → HTTP {code} "
                    f"({elapsed:.1f}s)"
                )

        if code < 400:
            total_elapsed = (
                time.monotonic() - t_start
            )

            route_label = " → ".join(
                h["tier"]
                for h in log_history
            )

            log(
                f"✓ {route_label} "
                f"({total_elapsed:.1f}s)"
            )

            if isinstance(data, dict):
                data["_routing"] = {
                    "route": route_label,
                    "preferred_model": preferred_model,
                    "total_elapsed_s": round(
                        total_elapsed,
                        1,
                    ),
                    "tiers": log_history,
                    "lfm_rolling_avg_s": round(
                        rolling_avg(),
                        1,
                    ),
                    "context_requested": (
                        requested_context(body)
                    ),
                    "tools_present": bool(
                        body.get("tools")
                    ),
                }

            return (
                code,
                data,
                log_history,
            )

        log(
            f"✗ {tier_name} → HTTP {code} "
            f"({elapsed:.1f}s)"
        )

        last_error_data = data
        last_error_code = code

    total_elapsed = (
        time.monotonic() - t_start
    )

    log(
        "FAILED all tiers "
        f"({total_elapsed:.1f}s)"
    )

    fallback = last_error_data or {
        "error": {
            "message": "All tiers failed",
            "code": 503,
        }
    }

    if isinstance(fallback, dict):
        fallback["_routing"] = {
            "route": "ALL_TIERS_FAILED",
            "preferred_model": preferred_model,
            "total_elapsed_s": round(
                total_elapsed,
                1,
            ),
            "tiers": log_history,
        }

    return (
        last_error_code,
        fallback,
        log_history,
    )


# ── Probe ──────────────────────────────────────────────────────────────────

def probe_lfm():

    global _last_probe_at

    t0 = time.monotonic()

    try:
        probe = json.dumps(
            {
                "model": LFM_MODEL,
                "messages": [
                    {
                        "role": "user",
                        "content": "hi",
                    }
                ],
                "stream": False,
                "options": {
                    "num_predict": 1,
                    "num_ctx": 2048,
                },
            }
        ).encode()

        req = urllib.request.Request(
            f"{OLLAMA_BASE}/api/chat",
            data=probe,
            headers={
                "Content-Type": "application/json"
            },
            method="POST",
        )

        with urllib.request.urlopen(
            req,
            timeout=PROBE_TIMEOUT_SEC,
        ) as resp:
            resp.read()

        elapsed = time.monotonic() - t0
        record_latency(elapsed)

        log(
            f"probe — LFM healthy "
            f"({elapsed:.1f}s)"
        )

    except Exception as e:
        record_latency(
            LFM_TIMEOUT_SEC
        )
        log(
            f"probe — LFM failed: {e}"
        )

    _last_probe_at = time.monotonic()


# ── HTTP Handler ────────────────────────────────────────────────────────────

class RouterHandler(
    http.server.BaseHTTPRequestHandler
):

    def log_message(self, fmt, *args):
        if (
            args[0].startswith("GET /health")
            or args[0].startswith("GET /v1/health")
        ):
            return

        log(args[0])

    def _send_json(
        self,
        status: int,
        data: dict,
    ):
        try:
            self.send_response(status)
            self.send_header(
                "Content-Type",
                "application/json",
            )
            self.send_header(
                "Access-Control-Allow-Origin",
                "*",
            )
            self.end_headers()

            self.wfile.write(
                json.dumps(data).encode()
            )

        except (
            BrokenPipeError,
            ConnectionResetError,
        ):
            log(
                "client disconnected "
                "(broken pipe)"
            )

    def _send_error(
        self,
        status: int,
        message: str,
    ):
        self._send_json(
            status,
            {
                "error": {
                    "message": message,
                    "code": status,
                }
            },
        )

    def _send_streaming_completion(self, status: int, data: dict):
        """Convert a completed response into valid OpenAI SSE."""
        if status >= 400:
            self._send_json(status, data)
            return

        self.send_response(200)
        self.send_header(
            "Content-Type",
            "text/event-stream; charset=utf-8",
        )
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()

        try:
            choice = (data.get("choices") or [{}])[0]
            message = choice.get("message") or {}
            chunk_id = data.get(
                "id",
                f"chatcmpl-sm-{int(time.time())}",
            )
            created = data.get(
                "created",
                int(time.time()),
            )
            model = data.get(
                "model",
                DEFAULT_LOCAL_MODEL,
            )

            def send_event(payload):
                wire = (
                    "data: "
                    + json.dumps(
                        payload,
                        ensure_ascii=False,
                    )
                    + "\n\n"
                )
                self.wfile.write(
                    wire.encode("utf-8")
                )
                self.wfile.flush()

            send_event({
                "id": chunk_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [{
                    "index": 0,
                    "delta": {
                        "role": "assistant"
                    },
                    "finish_reason": None,
                }],
            })

            content = message.get("content") or ""
            if content:
                send_event({
                    "id": chunk_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "choices": [{
                        "index": 0,
                        "delta": {
                            "content": content
                        },
                        "finish_reason": None,
                    }],
                })

            reasoning = message.get("reasoning")
            if reasoning:
                send_event({
                    "id": chunk_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "choices": [{
                        "index": 0,
                        "delta": {
                            "reasoning": reasoning
                        },
                        "finish_reason": None,
                    }],
                })

            tool_calls = message.get("tool_calls") or []
            if tool_calls:
                send_event({
                    "id": chunk_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "choices": [{
                        "index": 0,
                        "delta": {
                            "tool_calls": tool_calls
                        },
                        "finish_reason": None,
                    }],
                })

            send_event({
                "id": chunk_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [{
                    "index": 0,
                    "delta": {},
                    "finish_reason": (
                        choice.get(
                            "finish_reason",
                            "stop",
                        )
                    ),
                }],
            })

            if data.get("usage"):
                send_event({
                    "id": chunk_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "choices": [],
                    "usage": data["usage"],
                })

            self.wfile.write(
                b"data: [DONE]\n\n"
            )
            self.wfile.flush()

        except (
            BrokenPipeError,
            ConnectionResetError,
        ):
            log(
                "client disconnected during SSE stream"
            )

    def do_OPTIONS(self):

        self.send_response(200)
        self.send_header(
            "Access-Control-Allow-Origin",
            "*",
        )
        self.send_header(
            "Access-Control-Allow-Methods",
            "POST, GET, OPTIONS",
        )
        self.send_header(
            "Access-Control-Allow-Headers",
            "Content-Type, Authorization",
        )
        self.end_headers()

    def do_GET(self):

        path = urlparse(
            self.path
        ).path

        if path in (
            "/health",
            "/v1/health",
        ):

            data = {
                "status": "ok",
                "lfm_rolling_avg_s": round(
                    rolling_avg(),
                    1,
                ),
                "lfm_samples": len(
                    _lfm_latencies
                ),
                "last_probe_at": (
                    _last_probe_at
                ),
                "config": {
                    "default_local_model": (
                        DEFAULT_LOCAL_MODEL
                    ),
                    "lfm_model": LFM_MODEL,
                    "qwen_model": QWEN_MODEL,
                    "default_context": (
                        DEFAULT_CONTEXT
                    ),
                    "lfm_timeout_s": (
                        LFM_TIMEOUT_SEC
                    ),
                    "slow_threshold_s": (
                        SLOW_THRESHOLD_SEC
                    ),
                    "nvidia_model": NVIDIA_MODEL,
                    "groq_model": GROQ_MODEL,
                    "google_model": GOOGLE_MODEL,
                    "openrouter_model": (
                        OPENROUTER_MODEL
                    ),
                    "health_window": (
                        HEALTH_WINDOW
                    ),
                    "probe_interval_s": (
                        PROBE_INTERVAL_SEC
                    ),
                },
            }

            self._send_json(
                200,
                data,
            )
            return

        # IMPORTANT:
        # Only selectable local models are exposed here.
        # Cloud fallback tiers remain internal routing policy.
        if path in (
            "/v1/models",
            "/api/tags",
        ):

            models = [
                {
                    "id": AUTO_MODEL,
                    "object": "model",
                    "created": int(
                        time.time()
                    ),
                    "owned_by": "qalbstack-router",
                },
                {
                    "id": LFM_MODEL,
                    "object": "model",
                    "created": int(
                        time.time()
                    ),
                    "owned_by": "ollama",
                },
                {
                    "id": QWEN_MODEL,
                    "object": "model",
                    "created": int(
                        time.time()
                    ),
                    "owned_by": "ollama",
                },
            ]

            self._send_json(
                200,
                {
                    "object": "list",
                    "data": models,
                },
            )
            return

        if (
            path.startswith("/v1/models/")
            or path.startswith("/api/models/")
        ):

            requested = (
                path.rsplit(
                    "/",
                    1,
                )[-1]
            )

            model = (
                requested
                if requested in LOCAL_MODELS
                else DEFAULT_LOCAL_MODEL
            )

            self._send_json(
                200,
                {
                    "id": model,
                    "object": "model",
                    "owned_by": "ollama",
                },
            )
            return

        if path == "/api/show":

            self._send_json(
                200,
                {
                    "modelfile": (
                        f"FROM {DEFAULT_LOCAL_MODEL}\n"
                    ),
                    "template": "{{ .Prompt }}",
                    "details": {
                        "parent_model": "",
                        "format": "gguf",
                        "family": "local",
                    },
                    "model_info": {},
                },
            )
            return

        if path in (
            "/version",
            "/props",
            "/v1/props",
        ):

            self._send_json(
                200,
                {
                    "version": (
                        "qalbstack-sm-router/2.1"
                    ),
                    "props": {},
                },
            )
            return

        self._send_error(
            404,
            f"Not found: {path}",
        )

    def do_POST(self):

        path = urlparse(
            self.path
        ).path

        if path in (
            "/v1/chat/completions",
            "/api/chat",
        ):

            length = int(
                self.headers.get(
                    "Content-Length",
                    0,
                )
            )

            raw = self.rfile.read(
                length
            )

            try:
                body = json.loads(raw)
            except json.JSONDecodeError:
                self._send_error(
                    400,
                    "Invalid JSON",
                )
                return

            wants_stream = bool(
                body.get("stream")
            )

            if wants_stream:
                body = dict(body)
                body["stream"] = False

            status, data, _ = route_request(
                body
            )

            if wants_stream:
                self._send_streaming_completion(
                    status,
                    data,
                )
            else:
                self._send_json(
                    status,
                    data,
                )

            return

        if path == "/api/generate":

            length = int(
                self.headers.get(
                    "Content-Length",
                    0,
                )
            )

            raw = self.rfile.read(
                length
            )

            try:
                body = json.loads(raw)
            except json.JSONDecodeError:
                self._send_error(
                    400,
                    "Invalid JSON",
                )
                return

            prompt = body.get(
                "prompt",
                "",
            )

            options = body.get(
                "options",
                {},
            )

            chat_body = {
                "model": body.get(
                    "model",
                    DEFAULT_LOCAL_MODEL,
                ),
                "messages": [
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                "max_tokens": options.get(
                    "num_predict",
                    MAX_OUTPUT,
                ),
                "temperature": options.get(
                    "temperature",
                    0.7,
                ),
                "stream": False,
            }

            # Translate Ollama options for the
            # local tier.
            if options.get("num_ctx"):
                chat_body["num_ctx"] = (
                    options["num_ctx"]
                )

            status, data, _ = route_request(
                chat_body
            )

            if (
                status == 200
                and isinstance(data, dict)
            ):

                choice = data.get(
                    "choices",
                    [{}],
                )[0]

                msg = choice.get(
                    "message",
                    {},
                )

                gen_resp = {
                    "model": data.get(
                        "model",
                        DEFAULT_LOCAL_MODEL,
                    ),
                    "created_at": data.get(
                        "created",
                        "",
                    ),
                    "response": msg.get(
                        "content",
                        "",
                    ),
                    "done": True,
                    "done_reason": choice.get(
                        "finish_reason",
                        "stop",
                    ),
                    "context": [],
                    "total_duration": 0,
                    "prompt_eval_count": (
                        data.get(
                            "usage",
                            {},
                        ).get(
                            "prompt_tokens",
                            0,
                        )
                    ),
                    "eval_count": (
                        data.get(
                            "usage",
                            {},
                        ).get(
                            "completion_tokens",
                            0,
                        )
                    ),
                }

                self._send_json(
                    status,
                    gen_resp,
                )

            else:
                self._send_json(
                    status,
                    data,
                )

            return

        self._send_error(
            404,
            f"Not found: {path}",
        )


# ── Probe thread ────────────────────────────────────────────────────────────

def probe_loop():

    urllib.request.install_opener(
        urllib.request.build_opener(
            urllib.request.ProxyHandler({})
        )
    )

    while True:

        time.sleep(
            PROBE_INTERVAL_SEC
        )

        try:
            probe_lfm()
        except Exception as e:
            log(
                f"probe error: {e}"
            )


# ── Main ───────────────────────────────────────────────────────────────────

def main():

    urllib.request.install_opener(
        urllib.request.build_opener(
            urllib.request.ProxyHandler({})
        )
    )

    # Initial cheap LFM health probe.
    try:
        probe_lfm()
    except Exception:
        pass

    thread = threading.Thread(
        target=probe_loop,
        daemon=True,
    )

    thread.start()

    server = http.server.ThreadingHTTPServer(
        ("127.0.0.1", PORT),
        RouterHandler,
    )

    server.timeout = 0.5

    log(
        f"listening on 127.0.0.1:{PORT}"
    )

    log(
        "Local models: "
        f"{LFM_MODEL} / {QWEN_MODEL}"
    )

    log(
        "Auto policy: LFM for ordinary/fast work; "
        "Qwen for complex reasoning/analysis"
    )

    log(
        f"Default local: "
        f"{DEFAULT_LOCAL_MODEL}"
    )

    log(
        f"Internal fallback: "
        f"NVIDIA → Groq → Google → OpenRouter"
    )

    log(
        f"Context: {DEFAULT_CONTEXT}"
    )

    log(
        f"LFM timeout: "
        f"{LFM_TIMEOUT_SEC}s | "
        f"Slow threshold: "
        f"{SLOW_THRESHOLD_SEC}s"
    )

    log(
        "Threaded: YES | "
        f"Probes every "
        f"{PROBE_INTERVAL_SEC}s"
    )

    try:
        server.serve_forever()

    except KeyboardInterrupt:
        log("shutting down")

        server.server_close()


if __name__ == "__main__":
    main()
