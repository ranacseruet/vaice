"""Settings, overridable through environment variables."""
import os
from pathlib import Path


def _number(name: str, default: str, kind: type[int] | type[float]) -> int | float:
    raw = os.getenv(name, default)
    try:
        return kind(raw)
    except ValueError:
        raise SystemExit(f"vaice: {name} must be a number ({kind.__name__}), got {raw!r}") from None


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
LLM_PROMPT = os.getenv("VAICE_PROMPT", "Reply naturally and concisely in 1 to 2 spoken sentences.")
LLM_TIMEOUT = _number("VAICE_LLM_TIMEOUT", "120", float)

TTS_MODEL = os.getenv("VAICE_TTS_MODEL", "mlx-community/Kokoro-82M-bf16")
TTS_VOICE = os.getenv("VAICE_VOICE", "af_heart")
TTS_SPEED = _number("VAICE_SPEED", "1.1", float)

PORT = _number("VAICE_PORT", "7860", int)

# The conversation is kept here between runs; the last HISTORY_TURNS exchanges go into each prompt
DATA_DIR = Path(os.getenv("VAICE_DATA_DIR", "~/.vaice")).expanduser()
CONVERSATION_FILE = DATA_DIR / "conversation.json"
HISTORY_TURNS = _number("VAICE_HISTORY_TURNS", "10", int)
