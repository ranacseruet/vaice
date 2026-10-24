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


def test_http_error_keeps_llama_servers_reason(monkeypatch):
    body = '{"error":{"code":400,"message":"Failed to load image or audio file"}}'
    monkeypatch.setattr(llm.requests, "post", lambda *a, **kw: FakeResponse(400, body))
    with pytest.raises(requests.HTTPError, match="Failed to load image or audio file"):
        list(llm.stream_reply("AAAA"))


def test_stream_reply_yields_text_until_done(monkeypatch):
    lines = [
        sse({"choices": [{"delta": {"content": "Hi"}}]}),
        b"",
        b"data: [DONE]",
        sse({"choices": [{"delta": {"content": "late"}}]}),  # must not be read
    ]
    monkeypatch.setattr(llm.requests, "post", lambda *a, **kw: FakeResponse(200, lines=lines))
    assert "".join(llm.stream_reply("AAAA")) == "Hi"


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
