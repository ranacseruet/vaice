"""Streaming client for llama.cpp's OpenAI-compatible chat endpoint."""
import json
import logging
from collections.abc import Iterator, Sequence
from urllib.parse import urljoin

import requests

from vaice import config
from vaice.conversation import Message

log = logging.getLogger(__name__)

TRANSCRIBE_PROMPT = "Transcribe this audio verbatim. Output only the transcript."


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


def _check(r: requests.Response) -> None:
    if not r.ok:  # raise_for_status() would drop the body, which carries llama-server's reason
        raise requests.HTTPError(f"{r.status_code} from {config.LLM_URL}: {r.text[:500]}", response=r)


def stream_reply(question: str, history: Sequence[Message]) -> Iterator[str]:
    """Yield the reply to the transcribed question as it streams, with earlier turns as context.

    History is trimmed in blocks (see conversation.window_start), so between trims the prompt
    prefix is unchanged and llama-server reuses its cache.
    """
    payload = {
        "model": config.LLM_MODEL,
        "stream": True,
        "messages": [{"role": "system", "content": config.LLM_PROMPT}]
        + [{"role": role, "content": text} for role, text in history]
        + [{"role": "user", "content": question}],
    }
    with requests.post(config.LLM_URL, json=payload, stream=True, timeout=config.LLM_TIMEOUT) as r:
        _check(r)
        for line in r.iter_lines():
            if line == b"data: [DONE]":
                break
            yield delta_text(line)


def transcribe(audio_b64: str) -> str:
    """What the audio says, verbatim."""
    payload = {
        "model": config.LLM_MODEL,
        "stream": False,
        "max_tokens": 200,
        "temperature": 0,  # the server's default sampling mishears words now and then
        "messages": [{
            "role": "user",
            "content": [
                # Gemma expects the text prompt before the audio
                {"type": "text", "text": TRANSCRIBE_PROMPT},
                {"type": "input_audio", "input_audio": {"data": audio_b64, "format": "wav"}},
            ],
        }],
    }
    r = requests.post(config.LLM_URL, json=payload, timeout=config.LLM_TIMEOUT)
    _check(r)
    return (r.json()["choices"][0]["message"]["content"] or "").strip()


def check_server() -> None:
    """Warn at startup if llama-server isn't reachable, instead of failing on the first question."""
    health = urljoin(config.LLM_URL, "/health")
    try:
        r = requests.get(health, timeout=2)
    except requests.RequestException as exc:
        log.warning("llama-server not reachable at %s (%s); cascade replies will fail until it is up", health, exc)
        return
    if r.status_code == 503:  # llama-server answers 503 "Loading model" until its weights are in
        log.info("llama-server at %s is still loading its model", health)
    elif not r.ok:
        log.warning("llama-server health check at %s returned %d: %s", health, r.status_code, r.text[:200])
