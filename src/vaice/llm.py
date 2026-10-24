"""Streaming client for llama.cpp's OpenAI-compatible chat endpoint."""
import json
from collections.abc import Iterator

import requests

from vaice import config


def delta_text(line: bytes) -> str:
    """Text carried by one SSE line; "" for keep-alives, [DONE] and non-text chunks."""
    if not line.startswith(b"data: "):
        return ""
    data = line[6:].decode("utf-8")
    if data == "[DONE]":
        return ""
    choices = json.loads(data).get("choices") or []  # final usage chunk may have none
    if not choices:
        return ""
    # llama.cpp sends "content": null on the first (role-only) chunk
    return choices[0].get("delta", {}).get("content") or ""


def stream_reply(audio_b64: str) -> Iterator[str]:
    """Send mic audio straight to the model and yield reply text as it streams."""
    payload = {
        "model": config.LLM_MODEL,
        "stream": True,
        "messages": [{
            "role": "user",
            "content": [
                # Gemma expects the text prompt before the audio
                {"type": "text", "text": config.LLM_PROMPT},
                {"type": "input_audio", "input_audio": {"data": audio_b64, "format": "wav"}},
            ],
        }],
    }
    with requests.post(config.LLM_URL, json=payload, stream=True, timeout=config.LLM_TIMEOUT) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if line == b"data: [DONE]":
                break
            yield delta_text(line)
