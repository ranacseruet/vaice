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


def _choice(name: str, *, default: str, choices: set[str]) -> str:
    raw = os.getenv(name, default)
    if raw not in choices:
        formatted = ", ".join(sorted(choices))
        raise SystemExit(f"vaice: {name} must be one of {formatted}, got {raw!r}")
    return raw


def _path(*names: str, default: str | Path) -> Path:
    return Path(_env(*names, default=str(default))).expanduser()


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


def _text_api_url() -> str:
    """OpenAI-compatible API base, preserving path prefixes; defaults to the llama-server `/v1` base."""
    raw = os.getenv("VAICE_TEXT_URL")
    if raw is None:
        return f"{LLAMACPP_URL}/v1"
    parts = urlsplit(raw if "://" in raw else f"http://{raw}")
    if parts.scheme not in ("http", "https") or not parts.netloc or parts.query or parts.fragment:
        raise SystemExit(
            "vaice: VAICE_TEXT_URL must be an http(s) API base like http://localhost:8001/v1, "
            f"got {raw!r}"
        )
    return f"{parts.scheme}://{parts.netloc}{parts.path.rstrip('/')}"


# llama-server's /v1/realtime. The "Respond with interleaved text and audio." mode line
# is added in code (responders/llamacpp.py), so a prompt set with the older VAICE_LLM_PROMPT name keeps its meaning.
LLAMACPP_URL = _server_url("VAICE_LLAMACPP_URL", "VAICE_LLM_URL", default="http://localhost:8001")
LLAMACPP_PROMPT = _env(
    "VAICE_LLAMACPP_PROMPT",
    "VAICE_LLM_PROMPT",
    "VAICE_PROMPT",
    default="Reply naturally and concisely in 1 to 2 spoken sentences.",
)
LLAMACPP_TIMEOUT = _number("VAICE_LLAMACPP_TIMEOUT", "VAICE_LLM_TIMEOUT", default="120", kind=float)
TEXT_URL = _text_api_url()

# The historical path stays the default. Auto is opt-in because text endpoint compatibility cannot
# be established from model modality metadata alone.
PIPELINE = _choice("VAICE_PIPELINE", default="realtime", choices={"realtime", "text", "auto"})

# The text path uses the same prompt aliases as realtime, but the removed VAICE_LLM_MODEL setting stays ignored.
TEXT_MODEL = _env("VAICE_TEXT_MODEL", default="")
TEXT_PROMPT = _env(
    "VAICE_TEXT_PROMPT",
    "VAICE_LLM_PROMPT",
    "VAICE_PROMPT",
    default="Reply naturally and concisely in 1 to 2 spoken sentences.",
)

WHISPER_MODEL_DIR = _path("VAICE_WHISPER_MODEL_DIR", default=Path("~/.vaice/models/sherpa-onnx-whisper-tiny"))
WHISPER_ENCODER = _path(
    "VAICE_WHISPER_ENCODER",
    default=WHISPER_MODEL_DIR / "tiny-encoder.int8.onnx",
)
WHISPER_DECODER = _path(
    "VAICE_WHISPER_DECODER",
    default=WHISPER_MODEL_DIR / "tiny-decoder.int8.onnx",
)
WHISPER_TOKENS = _path("VAICE_WHISPER_TOKENS", default=WHISPER_MODEL_DIR / "tiny-tokens.txt")
WHISPER_LANGUAGE = _env("VAICE_WHISPER_LANGUAGE", default="")

KOKORO_MODEL_DIR = _path("VAICE_KOKORO_MODEL_DIR", default=Path("~/.vaice/models/kokoro-en-v0_19"))
KOKORO_MODEL = _path("VAICE_KOKORO_MODEL", default=KOKORO_MODEL_DIR / "model.onnx")
KOKORO_VOICES = _path("VAICE_KOKORO_VOICES", default=KOKORO_MODEL_DIR / "voices.bin")
KOKORO_TOKENS = _path("VAICE_KOKORO_TOKENS", default=KOKORO_MODEL_DIR / "tokens.txt")
KOKORO_LEXICON = _path("VAICE_KOKORO_LEXICON", default=KOKORO_MODEL_DIR / "lexicon-en-us.txt")
KOKORO_DATA_DIR = _path("VAICE_KOKORO_DATA_DIR", default=KOKORO_MODEL_DIR / "espeak-ng-data")
KOKORO_SPEAKER = _env("VAICE_KOKORO_SPEAKER", default="0")
KOKORO_SPEED = _number("VAICE_KOKORO_SPEED", default="1", kind=float)

PORT = _number("VAICE_PORT", default="7860", kind=int)

# The conversation is kept here between runs; the last HISTORY_TURNS exchanges go into each prompt
DATA_DIR = Path(os.getenv("VAICE_DATA_DIR", "~/.vaice")).expanduser()
CONVERSATION_FILE = DATA_DIR / "conversation.json"
HISTORY_TURNS = _number("VAICE_HISTORY_TURNS", default="10", kind=int)
