import json
import logging

import pytest
import requests

from vaice import llm
from vaice.llm import delta_text


def sse(obj) -> bytes:
    return b"data: " + json.dumps(obj).encode()


def test_null_content_first_chunk_is_empty_string():
    # Regression: llama.cpp's role-only first chunk has "content": null
    assert delta_text(sse({"choices": [{"delta": {"role": "assistant", "content": None}}]})) == ""


def test_content_chunk():
    assert delta_text(sse({"choices": [{"delta": {"content": " Paris"}}]})) == " Paris"


def test_empty_choices_chunk():
    assert delta_text(sse({"choices": []})) == ""


def test_keepalive_and_done():
    assert delta_text(b"") == ""
    assert delta_text(b"data: [DONE]") == ""


def test_error_payload_mid_stream_raises():
    with pytest.raises(RuntimeError, match="context full"):
        delta_text(sse({"error": {"code": 500, "message": "context full"}}))


class FakeResponse:
    def __init__(self, status, body="", lines=()):
        self.status_code, self.text, self._lines = status, body, lines
        self.ok = status < 400

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def iter_lines(self):
        return iter(self._lines)

    def raise_for_status(self):
        if not self.ok:
            raise requests.HTTPError(f"{self.status_code} Client Error")

    def json(self):
        return json.loads(self.text)


def test_http_error_keeps_llama_servers_reason(monkeypatch):
    body = '{"error":{"code":400,"message":"Failed to load image or audio file"}}'
    monkeypatch.setattr(llm.requests, "post", lambda *a, **kw: FakeResponse(400, body))
    with pytest.raises(requests.HTTPError, match="Failed to load image or audio file"):
        list(llm.stream_reply("hi", []))


def test_stream_reply_yields_text_until_done(monkeypatch):
    lines = [
        sse({"choices": [{"delta": {"content": "Hi"}}]}),
        b"",
        b"data: [DONE]",
        sse({"choices": [{"delta": {"content": "late"}}]}),  # must not be read
    ]
    monkeypatch.setattr(llm.requests, "post", lambda *a, **kw: FakeResponse(200, lines=lines))
    assert "".join(llm.stream_reply("hi", [])) == "Hi"


def capture_post(monkeypatch, response):
    sent = {}

    def post(url, json=None, **kwargs):
        sent["payload"] = json
        return response

    monkeypatch.setattr(llm.requests, "post", post)
    return sent


def test_the_question_follows_the_system_prompt_and_history_as_plain_text(monkeypatch):
    sent = capture_post(monkeypatch, FakeResponse(200, lines=[b"data: [DONE]"]))
    history = [("user", "capital of France?"), ("assistant", "Paris.")]

    list(llm.stream_reply("And Germany?", history))

    assert sent["payload"]["messages"] == [
        {"role": "system", "content": llm.config.LLM_PROMPT},
        {"role": "user", "content": "capital of France?"},
        {"role": "assistant", "content": "Paris."},
        {"role": "user", "content": "And Germany?"},
    ]
    assert sent["payload"]["stream"] is True


def test_transcribe_asks_for_a_verbatim_transcript_and_returns_it_stripped(monkeypatch):
    body = json.dumps({"choices": [{"message": {"role": "assistant", "content": " What is the capital of France?\n"}}]})
    sent = capture_post(monkeypatch, FakeResponse(200, body))

    assert llm.transcribe("AAAA") == "What is the capital of France?"

    payload = sent["payload"]
    assert payload["stream"] is False and payload["max_tokens"] == 200
    assert payload["temperature"] == 0  # greedy: sampling mishears words now and then
    (message,) = payload["messages"]
    # the text prompt comes before the audio, as Gemma needs
    assert message["content"][0] == {"type": "text", "text": llm.TRANSCRIBE_PROMPT}
    assert message["content"][1]["input_audio"] == {"data": "AAAA", "format": "wav"}


def test_transcribe_of_null_content_is_empty(monkeypatch):
    capture_post(monkeypatch, FakeResponse(200, json.dumps({"choices": [{"message": {"content": None}}]})))
    assert llm.transcribe("AAAA") == ""


def test_transcribe_http_error_keeps_llama_servers_reason(monkeypatch):
    body = '{"error":{"code":400,"message":"audio too long"}}'
    capture_post(monkeypatch, FakeResponse(400, body))
    with pytest.raises(requests.HTTPError, match="audio too long"):
        llm.transcribe("AAAA")


def test_unreachable_server_is_a_startup_warning(monkeypatch, caplog):
    def refused(*a, **kw):
        raise requests.ConnectionError("connection refused")

    monkeypatch.setattr(llm.requests, "get", refused)
    monkeypatch.setattr(llm.config, "LLM_URL", "http://localhost:8001/v1/chat/completions")
    with caplog.at_level(logging.WARNING, logger=llm.__name__):
        llm.check_server()
    assert "not reachable at http://localhost:8001/health" in caplog.text


def test_server_still_loading_is_not_a_warning(monkeypatch, caplog):
    loading = FakeResponse(503, '{"error":{"message":"Loading model"}}')
    monkeypatch.setattr(llm.requests, "get", lambda *a, **kw: loading)
    with caplog.at_level(logging.INFO, logger=llm.__name__):
        llm.check_server()
    assert "still loading" in caplog.text
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]
