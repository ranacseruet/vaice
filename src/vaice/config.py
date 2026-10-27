"""Settings, overridable through environment variables."""
import os
from pathlib import Path


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


# auto: native speech-to-speech model if it loads, else the cascade (llama.cpp + Kokoro)
BACKEND = os.getenv("VAICE_BACKEND", "auto")

# Native backend (mlx-audio speech-to-speech). Empty VAICE_NATIVE_MODEL disables it.
NATIVE_MODEL = os.getenv("VAICE_NATIVE_MODEL", "mlx-community/LFM2.5-Audio-1.5B-8bit")
NATIVE_PROMPT = os.getenv(
    "VAICE_NATIVE_PROMPT",
    "Respond with interleaved text and audio. Reply naturally and concisely in 1 to 2 spoken sentences.",
)

# Cascade backend
LLM_URL = os.getenv("VAICE_LLM_URL", "http://localhost:8001/v1/chat/completions")
LLM_MODEL = os.getenv("VAICE_LLM_MODEL", "gemma-4-12b")  # llama-server serves one model; name is informational
LLM_PROMPT = _env(
    "VAICE_LLM_PROMPT", "VAICE_PROMPT", default="Reply naturally and concisely in 1 to 2 spoken sentences."
)
LLM_TIMEOUT = _number("VAICE_LLM_TIMEOUT", default="120", kind=float)

TTS_MODEL = os.getenv("VAICE_TTS_MODEL", "mlx-community/Kokoro-82M-bf16")
TTS_VOICE = _env("VAICE_TTS_VOICE", "VAICE_VOICE", default="af_heart")
TTS_SPEED = _number("VAICE_TTS_SPEED", "VAICE_SPEED", default="1.1", kind=float)

PORT = _number("VAICE_PORT", default="7860", kind=int)

# The conversation is kept here between runs; the last HISTORY_TURNS exchanges go into each prompt
DATA_DIR = Path(os.getenv("VAICE_DATA_DIR", "~/.vaice")).expanduser()
CONVERSATION_FILE = DATA_DIR / "conversation.json"
HISTORY_TURNS = _number("VAICE_HISTORY_TURNS", default="10", kind=int)
