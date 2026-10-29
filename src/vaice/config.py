"""Settings, overridable through environment variables."""
import os
from pathlib import Path
from urllib.parse import urlsplit


def _env(name: str, *, default: str) -> str:
    raw = os.getenv(name)
    return default if raw is None else raw


def _number(name: str, *, default: str, kind: type[int] | type[float]) -> int | float:
    raw = os.getenv(name)
    if raw is None:
        return kind(default)
    try:
        return kind(raw)
    except ValueError:
        raise SystemExit(f"vaice: {name} must be a number ({kind.__name__}), got {raw!r}") from None


def server_base(url: str) -> str:
    """scheme://host:port of `url`: any path suffix is dropped. Raises ValueError if there is no host."""
    parts = urlsplit(url if "://" in url else f"http://{url}")
    if parts.scheme not in ("http", "https") or not parts.netloc:
        raise ValueError("expected an http(s) URL like http://localhost:8001")
    return f"{parts.scheme}://{parts.netloc}"


def _server_url(name: str, *, default: str) -> str:
    raw = os.getenv(name)
    if raw is None:
        return server_base(default)
    try:
        return server_base(raw)
    except ValueError as exc:
        raise SystemExit(f"vaice: {name} must be a URL ({exc}), got {raw!r}") from None


# llama-server's /v1/realtime. The "Respond with interleaved text and audio." mode line
# is added in code (responders/llamacpp.py), so VAICE_LLAMACPP_PROMPT only has to say how to answer.
LLAMACPP_URL = _server_url("VAICE_LLAMACPP_URL", default="http://localhost:8001")
LLAMACPP_PROMPT = _env(
    "VAICE_LLAMACPP_PROMPT",
    default="Reply naturally and concisely in 1 to 2 spoken sentences.",
)
LLAMACPP_TIMEOUT = _number("VAICE_LLAMACPP_TIMEOUT", default="120", kind=float)

PORT = _number("VAICE_PORT", default="7860", kind=int)

# The conversation is kept here between runs; the last HISTORY_TURNS exchanges go into each prompt
DATA_DIR = Path(os.getenv("VAICE_DATA_DIR", "~/.vaice")).expanduser()
CONVERSATION_FILE = DATA_DIR / "conversation.json"
HISTORY_TURNS = _number("VAICE_HISTORY_TURNS", default="10", kind=int)
