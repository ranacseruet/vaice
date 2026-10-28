"""Settings, overridable through environment variables."""
import os
from pathlib import Path
from urllib.parse import urlsplit


def _env(*names: str, default: str) -> str:
    """The first of `names` that is set, else `default`. The current name comes first, so it wins
    over the older name it replaced; both keep working (the old one is not removed)."""
    for name in names:
        raw = os.getenv(name)
        if raw is not None:
            return raw
    return default


def _number(*names: str, default: str, kind: type[int] | type[float]) -> int | float:
    for name in names:  # same alias rule as _env, but the error names the variable that was set
        raw = os.getenv(name)
        if raw is not None:
            try:
                return kind(raw)
            except ValueError:
                raise SystemExit(f"vaice: {name} must be a number ({kind.__name__}), got {raw!r}") from None
    return kind(default)


def server_base(url: str) -> str:
    """scheme://host:port of `url`: the path is dropped, so an old VAICE_LLM_URL that ended in
    /v1/chat/completions still points at the server. Raises ValueError if there is no host."""
    parts = urlsplit(url if "://" in url else f"http://{url}")
    if parts.scheme not in ("http", "https") or not parts.netloc:
        raise ValueError("expected an http(s) URL like http://localhost:8001")
    return f"{parts.scheme}://{parts.netloc}"


def _server_url(*names: str, default: str) -> str:
    for name in names:  # same alias rule as _env, but the error names the variable that was set
        raw = os.getenv(name)
        if raw is not None:
            try:
                return server_base(raw)
            except ValueError as exc:
                raise SystemExit(f"vaice: {name} must be a URL ({exc}), got {raw!r}") from None
    return server_base(default)


# auto: the llama.cpp backend (llama-server's own speech) if the server is up and serves audio output,
# else the native speech-to-speech model on MLX. `cascade` is the old name of `llamacpp`.
BACKEND = os.getenv("VAICE_BACKEND", "auto")

# Native backend (mlx-audio speech-to-speech). Empty VAICE_NATIVE_MODEL disables it.
NATIVE_MODEL = os.getenv("VAICE_NATIVE_MODEL", "mlx-community/LFM2.5-Audio-1.5B-8bit")
NATIVE_PROMPT = os.getenv(
    "VAICE_NATIVE_PROMPT",
    "Respond with interleaved text and audio. Reply naturally and concisely in 1 to 2 spoken sentences.",
)

# llama.cpp backend: llama-server's /v1/realtime. The "Respond with interleaved text and audio." mode line
# is added in code (responders/llamacpp.py), so a prompt set for the older cascade keeps its meaning.
LLAMACPP_URL = _server_url("VAICE_LLAMACPP_URL", "VAICE_LLM_URL", default="http://localhost:8001")
LLAMACPP_PROMPT = _env(
    "VAICE_LLAMACPP_PROMPT",
    "VAICE_LLM_PROMPT",
    "VAICE_PROMPT",
    default="Reply naturally and concisely in 1 to 2 spoken sentences.",
)
LLAMACPP_TIMEOUT = _number("VAICE_LLAMACPP_TIMEOUT", "VAICE_LLM_TIMEOUT", default="120", kind=float)

PORT = _number("VAICE_PORT", default="7860", kind=int)

# The conversation is kept here between runs; the last HISTORY_TURNS exchanges go into each prompt
DATA_DIR = Path(os.getenv("VAICE_DATA_DIR", "~/.vaice")).expanduser()
CONVERSATION_FILE = DATA_DIR / "conversation.json"
HISTORY_TURNS = _number("VAICE_HISTORY_TURNS", default="10", kind=int)
