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


def _flag(name: str, *, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    if raw.lower() in ("1", "true", "yes", "on"):
        return True
    if raw.lower() in ("0", "false", "no", "off"):
        return False
    raise SystemExit(f"vaice: {name} must be a boolean (1/true/yes/on or 0/false/no/off), got {raw!r}")


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

# Reply cap sent as the realtime session's max_output_tokens; text tokens and audio frames count
# together (the LFM3-Audio codec emits 12 tokens per 80 ms of speech). The model card's own
# voice-chat example uses 1024; llama-server's session default would be "inf".
LLAMACPP_MAX_OUTPUT_TOKENS = _number("VAICE_LLAMACPP_MAX_OUTPUT_TOKENS", default="1024", kind=int)

PORT = _number("VAICE_PORT", default="7860", kind=int)

# The conversation is kept here between runs; the last HISTORY_TURNS exchanges go into each prompt
DATA_DIR = Path(os.getenv("VAICE_DATA_DIR", "~/.vaice")).expanduser()
CONVERSATION_FILE = DATA_DIR / "conversation.json"
HISTORY_TURNS = _number("VAICE_HISTORY_TURNS", default="10", kind=int)

# fastrtc's Silero VAD: what counts as speech, and when an utterance ends. A barge-in fires when
# the interrupting sound ends, so the threshold decides what can interrupt a playing reply —
# raise it if room noise or speaker feedback cuts replies short. The min-silence gap delays both
# the reply start and interrupts; the speech pad pads detected speech on both sides.
VAD_THRESHOLD = _number("VAICE_VAD_THRESHOLD", default="0.6", kind=float)
VAD_MIN_SILENCE_MS = _number("VAICE_VAD_MIN_SILENCE_MS", default="150", kind=int)
VAD_SPEECH_PAD_MS = _number("VAICE_VAD_SPEECH_PAD_MS", default="30", kind=int)
if not 0 < VAD_THRESHOLD < 1:
    raise SystemExit(f"vaice: VAICE_VAD_THRESHOLD must be between 0 and 1 (exclusive), got {VAD_THRESHOLD}")
if VAD_MIN_SILENCE_MS < 0 or VAD_SPEECH_PAD_MS < 0:
    raise SystemExit("vaice: VAICE_VAD_MIN_SILENCE_MS and VAICE_VAD_SPEECH_PAD_MS must be >= 0")

# Whether speech can interrupt a playing reply at all. Off = a reply always runs to the end.
CAN_INTERRUPT = _flag("VAICE_CAN_INTERRUPT", default=True)
