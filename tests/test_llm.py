import json

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
