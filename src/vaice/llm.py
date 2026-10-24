"""Streaming client for llama.cpp's OpenAI-compatible chat endpoint."""
import json
import logging
from collections.abc import Iterator
from urllib.parse import urljoin

import requests

from vaice import config

log = logging.getLogger(__name__)


def delta_text(line: bytes) -> str:
    """Text carried by one SSE line; "" for keep-alives, [DONE] and non-text chunks."""
    if not line.startswith(b"data: "):
        return ""
    data = line[6:].decode("utf-8")
    if data == "[DONE]":
        return ""
    chunk = json.loads(data)
    if "error" in chunk:  # llama-server reports mid-stream failures as a data line too
        raise RuntimeError(f"llama-server error: {chunk['error']}")
    choices = chunk.get("choices") or []  # final usage chunk may have none
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
        if not r.ok:  # raise_for_status() would drop the body, which carries llama-server's reason
            raise requests.HTTPError(f"{r.status_code} from {config.LLM_URL}: {r.text[:500]}", response=r)
        for line in r.iter_lines():
            if line == b"data: [DONE]":
                break
            yield delta_text(line)


def check_server() -> None:
    """Warn at startup if llama-server isn't reachable, instead of failing on the first question."""
    health = urljoin(config.LLM_URL, "/health")
    try:
        requests.get(health, timeout=2).raise_for_status()
    except requests.RequestException as exc:
        log.warning("llama-server not reachable at %s (%s); cascade replies will fail until it is up", health, exc)
