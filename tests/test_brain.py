import json

import pytest

from vaice import brain as brain_module
from vaice.brain import BrainReply, ChatBrain, ToolCall

MESSAGES = [{"role": "user", "content": "hi"}]
TOOLS = [{"type": "function", "function": {"name": "disk_space", "description": "d", "parameters": {}}}]


class FakeResponse:
    def __init__(self, payload=None, status=200, text=None):
        self.status_code = status
        self.ok = status < 400
        self._payload = payload
        self.text = text if text is not None else json.dumps(payload)

    def json(self):
        return json.loads(self.text)


@pytest.fixture
def post(monkeypatch):
    """Replaces requests.post: `post.reply` is the response; `post.sent` records (url, body, headers, timeout)."""

    class Post:
        reply = FakeResponse({"choices": [{"message": {"role": "assistant", "content": "Hello."}}]})

        def __init__(self):
            self.sent = []

        def __call__(self, url, json=None, headers=None, timeout=None):
            self.sent.append((url, json, headers, timeout))
            return self.reply

    fake = Post()
    monkeypatch.setattr(brain_module.requests, "post", fake)
    return fake


def completion(message):
    return FakeResponse({"choices": [{"message": message}]})


def test_a_plain_request_has_no_model_no_auth_and_no_tools(post):
    reply = ChatBrain("http://server:8001/", 7.5).complete(MESSAGES, None)

    url, body, headers, timeout = post.sent[0]
    assert url == "http://server:8001/v1/chat/completions"
    assert body == {"messages": MESSAGES, "temperature": 0, "max_tokens": 300}
    assert headers == {}
    assert timeout == 7.5
    assert reply == BrainReply("Hello.", [], {"role": "assistant", "content": "Hello."})


def test_the_model_the_key_the_tools_and_the_token_cap_are_sent_when_set(post):
    ChatBrain("http://server", 5, model="my-model", api_key="s3cret", max_tokens=50).complete(MESSAGES, TOOLS)

    _, body, headers, _ = post.sent[0]
    assert body == {"messages": MESSAGES, "temperature": 0, "max_tokens": 50, "model": "my-model", "tools": TOOLS}
    assert headers == {"Authorization": "Bearer s3cret"}


def test_empty_tools_are_left_out(post):
    ChatBrain("http://server", 5).complete(MESSAGES, [])
    assert "tools" not in post.sent[0][1]


def test_tool_calls_are_parsed_and_the_message_is_kept_for_the_conversation(post):
    post.reply = completion({
        "role": "assistant", "content": None,
        "tool_calls": [{"id": "abc", "type": "function", "function": {"name": "disk_space", "arguments": "{}"}},
                       {"id": "def", "type": "function",
                        "function": {"name": "find_files", "arguments": '{"name": "report"}'}}]})

    reply = ChatBrain("http://server", 5).complete(MESSAGES, TOOLS)

    assert reply.text == ""
    assert reply.calls == [ToolCall("abc", "disk_space", "{}"), ToolCall("def", "find_files", '{"name": "report"}')]
    assert reply.message["role"] == "assistant"
    assert reply.message["content"] == ""  # None becomes text, so it can be sent back
    assert [c["id"] for c in reply.message["tool_calls"]] == ["abc", "def"]


def test_arguments_that_arrive_as_an_object_become_a_json_string(post):
    post.reply = completion({"role": "assistant", "content": "", "tool_calls": [
        {"id": "abc", "function": {"name": "find_files", "arguments": {"name": "report", "folder": "downloads"}}}]})

    reply = ChatBrain("http://server", 5).complete(MESSAGES, TOOLS)

    assert json.loads(reply.calls[0].arguments) == {"name": "report", "folder": "downloads"}
    sent_back = reply.message["tool_calls"][0]
    assert sent_back == {"id": "abc", "type": "function",
                         "function": {"name": "find_files", "arguments": reply.calls[0].arguments}}


def test_a_call_without_an_id_or_arguments_still_parses(post):
    post.reply = completion({"role": "assistant", "content": "", "tool_calls": [
        {"function": {"name": "disk_space"}}]})

    reply = ChatBrain("http://server", 5).complete(MESSAGES, TOOLS)

    assert reply.calls == [ToolCall("call_0", "disk_space", "")]
    assert reply.message["tool_calls"][0]["id"] == "call_0"


def test_the_text_is_stripped_and_a_message_without_calls_has_no_tool_calls_key(post):
    post.reply = completion({"role": "assistant", "content": "  Fine.\n", "tool_calls": []})

    reply = ChatBrain("http://server", 5).complete(MESSAGES, TOOLS)

    assert reply.text == "Fine."
    assert reply.calls == []
    assert "tool_calls" not in reply.message


def test_a_non_2xx_answer_raises_with_the_status_and_the_start_of_the_body(post):
    post.reply = FakeResponse(status=503, text="x" * 500)

    with pytest.raises(RuntimeError, match=r"HTTP 503: x{200}$"):
        ChatBrain("http://server", 5).complete(MESSAGES, None)


@pytest.mark.parametrize("text", [
    "not json at all",
    "{}",
    '{"choices": []}',
    '{"choices": [{}]}',
    '{"choices": ["text"]}',
    '{"choices": [{"message": null}]}',
])
def test_a_response_without_a_message_raises(post, text):
    post.reply = FakeResponse(text=text)

    with pytest.raises(RuntimeError, match="sent no message"):
        ChatBrain("http://server", 5).complete(MESSAGES, None)


@pytest.mark.parametrize("message", [
    {"role": "assistant", "content": ["parts"]},
    {"role": "assistant", "content": "", "tool_calls": ["nope"]},
    {"role": "assistant", "content": "", "tool_calls": [{"id": "x", "function": {}}]},
    {"role": "assistant", "content": "", "tool_calls": [{"id": "x"}]},
])
def test_unexpected_content_or_a_malformed_call_raises(post, message):
    post.reply = completion(message)

    with pytest.raises(RuntimeError, match=r"unexpected content|malformed tool call"):
        ChatBrain("http://server", 5).complete(MESSAGES, None)
