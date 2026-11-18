import json
import logging
import traceback

import pytest

from vaice import brain as brain_module
from vaice.brain import CANT_REACH, UNUSABLE, BrainError, BrainReply, ChatBrain, ToolCall

ENDPOINT = "http://server/v1/chat/completions"
MESSAGES = [{"role": "user", "content": "hi"}]
TOOLS = [{"type": "function", "function": {"name": "disk_space", "description": "d", "parameters": {}}}]
KEY = "fake-key-ABCDEF123456"


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


def raising(error):
    def post(*_args, **_kwargs):
        raise error

    return post


@pytest.mark.parametrize("sentence", [CANT_REACH, brain_module.KEY_REFUSED, brain_module.BUSY, UNUSABLE])
def test_the_spoken_sentences_come_out_of_the_speech_path_intact(sentence):
    from vaice import voice

    spoken = voice.sentences(voice.speakable(sentence))

    assert " ".join(spoken) == sentence  # nothing is cleaned away (the colon, the apostrophes)


# -- the endpoint and the request body


def test_the_local_brain_posts_to_v1_chat_completions_with_temperature_0_and_a_token_cap(post):
    reply = ChatBrain.local("http://server:8001/", 7.5).complete(MESSAGES, None)

    url, body, headers, timeout = post.sent[0]
    assert url == "http://server:8001/v1/chat/completions"
    assert body == {"messages": MESSAGES, "temperature": 0, "max_tokens": 300}
    assert headers == {}
    assert timeout == 7.5
    assert reply == BrainReply("Hello.", [], {"role": "assistant", "content": "Hello."})


@pytest.mark.parametrize("base", [
    "http://localhost:1234/v1",
    "http://localhost:1234/v1/",
    "https://api.openai.com/v1",
])
def test_the_helper_posts_to_its_base_plus_chat_completions_without_doubling_v1(post, base):
    ChatBrain.helper(base, 5).complete(MESSAGES, None)

    assert post.sent[0][0] == f"{base.rstrip('/')}/chat/completions"
    assert "/v1/v1" not in post.sent[0][0]


def test_the_helper_sends_no_token_cap_unless_given_one(post):
    ChatBrain.helper("http://h/v1", 5).complete(MESSAGES, None)
    ChatBrain.helper("http://h/v1", 5, max_tokens=200).complete(MESSAGES, None)

    assert "max_tokens" not in post.sent[0][1]
    assert post.sent[1][1]["max_tokens"] == 200


def test_a_helper_base_without_v1_is_used_as_given(post):
    ChatBrain.helper("http://localhost:11434", 5).complete(MESSAGES, None)
    assert post.sent[0][0] == "http://localhost:11434/chat/completions"


def test_the_helper_sends_neither_temperature_nor_a_token_cap(post):
    ChatBrain.helper("http://localhost:1234/v1", 5).complete(MESSAGES, TOOLS)

    _, body, headers, timeout = post.sent[0]
    assert body == {"messages": MESSAGES, "tools": TOOLS}
    assert headers == {}  # no key set, no Authorization
    assert timeout == 5


def test_the_helper_sends_the_model_and_the_bearer_key_only_when_set(post):
    ChatBrain.helper("http://localhost:1234/v1", 5, model="m-1", api_key=KEY).complete(MESSAGES, None)
    ChatBrain.helper("http://localhost:1234/v1", 5, model="", api_key="").complete(MESSAGES, None)

    assert post.sent[0][1] == {"messages": MESSAGES, "model": "m-1"}
    assert post.sent[0][2] == {"Authorization": f"Bearer {KEY}"}
    assert post.sent[1][1] == {"messages": MESSAGES}
    assert post.sent[1][2] == {}


def test_the_constructor_choice_decides_temperature_and_cap(post):
    ChatBrain("http://x/chat", 5, model="my-model", api_key="s3cret", temperature=0.2, max_tokens=50).complete(
        MESSAGES, TOOLS)

    _, body, headers, _ = post.sent[0]
    assert body == {"messages": MESSAGES, "temperature": 0.2, "max_tokens": 50, "model": "my-model", "tools": TOOLS}
    assert headers == {"Authorization": "Bearer s3cret"}


def test_empty_tools_are_left_out(post):
    ChatBrain(ENDPOINT, 5).complete(MESSAGES, [])
    assert "tools" not in post.sent[0][1]


# -- the reply


def test_tool_calls_are_parsed_and_the_message_is_kept_for_the_conversation(post):
    post.reply = completion({
        "role": "assistant", "content": None,
        "tool_calls": [{"id": "abc", "type": "function", "function": {"name": "disk_space", "arguments": "{}"}},
                       {"id": "def", "type": "function",
                        "function": {"name": "find_files", "arguments": '{"name": "report"}'}}]})

    reply = ChatBrain(ENDPOINT, 5).complete(MESSAGES, TOOLS)

    assert reply.text == ""
    assert reply.calls == [ToolCall("abc", "disk_space", "{}"), ToolCall("def", "find_files", '{"name": "report"}')]
    assert reply.message["role"] == "assistant"
    assert reply.message["content"] == ""  # None becomes text, so it can be sent back
    assert [c["id"] for c in reply.message["tool_calls"]] == ["abc", "def"]


def test_arguments_that_arrive_as_an_object_become_a_json_string(post):
    post.reply = completion({"role": "assistant", "content": "", "tool_calls": [
        {"id": "abc", "function": {"name": "find_files", "arguments": {"name": "report", "folder": "downloads"}}}]})

    reply = ChatBrain(ENDPOINT, 5).complete(MESSAGES, TOOLS)

    assert json.loads(reply.calls[0].arguments) == {"name": "report", "folder": "downloads"}
    sent_back = reply.message["tool_calls"][0]
    assert sent_back == {"id": "abc", "type": "function",
                         "function": {"name": "find_files", "arguments": reply.calls[0].arguments}}


def test_a_call_without_an_id_or_arguments_still_parses(post):
    post.reply = completion({"role": "assistant", "content": "", "tool_calls": [
        {"function": {"name": "disk_space"}}]})

    reply = ChatBrain(ENDPOINT, 5).complete(MESSAGES, TOOLS)

    assert reply.calls == [ToolCall("call_0", "disk_space", "")]
    assert reply.message["tool_calls"][0]["id"] == "call_0"


def test_the_text_is_stripped_and_a_message_without_calls_has_no_tool_calls_key(post):
    post.reply = completion({"role": "assistant", "content": "  Fine.\n", "tool_calls": []})

    reply = ChatBrain(ENDPOINT, 5).complete(MESSAGES, TOOLS)

    assert reply.text == "Fine."
    assert reply.calls == []
    assert "tool_calls" not in reply.message


def test_the_message_sent_back_holds_only_role_content_and_tool_calls(post):
    post.reply = completion({
        "role": "assistant", "content": "Done.", "reasoning_content": "thinking about it", "reasoning": "more",
        "refusal": None, "annotations": [], "audio": None,
        "tool_calls": [{"id": "a", "type": "function", "index": 0,
                        "function": {"name": "disk_space", "arguments": "{}"}}]})

    reply = ChatBrain.helper("http://h/v1", 5).complete(MESSAGES, TOOLS)

    assert reply.text == "Done."  # content only, never the reasoning
    assert reply.message == {"role": "assistant", "content": "Done.", "tool_calls": [
        {"id": "a", "type": "function", "function": {"name": "disk_space", "arguments": "{}"}}]}


def test_a_message_without_calls_holds_only_role_and_content(post):
    post.reply = completion({"role": "assistant", "content": "Hi.", "reasoning_content": "x", "tool_calls": []})

    reply = ChatBrain.helper("http://h/v1", 5).complete(MESSAGES, None)

    assert reply.message == {"role": "assistant", "content": "Hi."}


def test_a_reply_cut_off_at_the_token_limit_with_nothing_said_warns_and_is_an_empty_answer(post, caplog):
    post.reply = FakeResponse({"choices": [{"finish_reason": "length", "message": {
        "role": "assistant", "content": "", "reasoning_content": "hmm" * 100}}]})

    with caplog.at_level(logging.WARNING, logger="vaice.brain"):
        reply = ChatBrain.helper("http://h/v1", 5).complete(MESSAGES, None)

    assert reply.text == "" and reply.calls == []  # the caller falls back as for any empty answer
    assert "token limit" in caplog.text and "hmm" not in caplog.text


@pytest.mark.parametrize("choice", [
    {"finish_reason": "length", "message": {"role": "assistant", "content": "A part"}},
    {"finish_reason": "stop", "message": {"role": "assistant", "content": ""}},
])
def test_no_token_limit_warning_when_there_is_text_or_the_stop_was_normal(post, caplog, choice):
    post.reply = FakeResponse({"choices": [choice]})

    with caplog.at_level(logging.WARNING, logger="vaice.brain"):
        ChatBrain.helper("http://h/v1", 5).complete(MESSAGES, None)

    assert caplog.text == ""


# -- failures


def test_a_non_2xx_answer_raises_with_the_status_and_the_start_of_the_body(post):
    post.reply = FakeResponse(status=503, text="x" * 500)

    with pytest.raises(BrainError, match=r"HTTP 503: x{200}$") as caught:
        ChatBrain(ENDPOINT, 5).complete(MESSAGES, None)

    assert isinstance(caught.value, RuntimeError)  # callers that caught RuntimeError still do
    assert caught.value.spoken == CANT_REACH


@pytest.mark.parametrize("status", [401, 403])
def test_a_refused_key_is_reported_without_the_response_body(post, status, caplog):
    post.reply = FakeResponse(status=status, text=f'{{"error": "Incorrect API key provided: {KEY[:12]}"}} {KEY}')

    with caplog.at_level(logging.DEBUG), pytest.raises(BrainError) as caught:
        ChatBrain.helper("https://h/v1", 5, "m", KEY).complete(MESSAGES, None)

    assert str(caught.value) == f"the brain endpoint refused the API key (HTTP {status})"
    assert caught.value.spoken == "I can't reach my language model: the key was refused."
    for text in (str(caught.value), repr(caught.value), caplog.text):
        assert KEY[:12] not in text and "Incorrect" not in text


def test_a_rate_limit_is_busy_not_broken(post):
    post.reply = FakeResponse(status=429, text="slow down")

    with pytest.raises(BrainError) as caught:
        ChatBrain.helper("https://h/v1", 5).complete(MESSAGES, None)

    assert str(caught.value) == "the brain endpoint is rate limited (HTTP 429)"
    assert caught.value.spoken == "My language model is busy right now. Try again in a moment."


def test_a_key_that_another_error_body_echoes_is_blanked_out_of_the_message(post):
    post.reply = FakeResponse(status=400, text=f"bad request for key {KEY}")

    with pytest.raises(BrainError) as caught:
        ChatBrain.helper("https://h/v1", 5, "m", KEY).complete(MESSAGES, None)

    assert KEY not in str(caught.value)
    assert "HTTP 400: bad request for key [key]" in str(caught.value)


@pytest.mark.parametrize("reply", [
    completion({"role": "assistant", "content": [f"here is {KEY}"]}),  # content that is not text
    completion({"role": "assistant", "tool_calls": [{"function": {"arguments": KEY}}]}),  # a call without a name
    FakeResponse(text=f"not json, but it holds {KEY}"),
    FakeResponse({"choices": [{"message": f"a string, with {KEY}"}]}),
])
def test_a_key_that_an_unusable_reply_echoes_never_reaches_the_message_the_chain_or_the_log(post, reply, caplog):
    import traceback

    post.reply = reply

    with pytest.raises(BrainError) as caught:
        ChatBrain.helper("https://h/v1", 5, "m", KEY).complete(MESSAGES, None)

    shown = "".join(traceback.format_exception(caught.value)) + repr(caught.value) + caplog.text
    assert KEY not in shown
    assert caught.value.__cause__ is None and caught.value.spoken == UNUSABLE


def test_a_timeout_is_wrapped_and_names_no_url(monkeypatch):
    monkeypatch.setattr(brain_module.requests, "post", raising(
        brain_module.requests.Timeout("HTTPSConnectionPool(host='h', port=443): Read timed out")))

    with pytest.raises(BrainError) as caught:
        ChatBrain.helper("https://h/v1", 7.5).complete(MESSAGES, None)

    assert str(caught.value) == "the brain endpoint did not answer within 7.5 s"
    assert caught.value.spoken == CANT_REACH


@pytest.mark.parametrize("error", [
    brain_module.requests.ConnectionError("Max retries exceeded with url: /v1/chat/completions"),
    brain_module.requests.exceptions.SSLError("certificate verify failed"),
])
def test_a_connection_error_is_wrapped(monkeypatch, error):
    monkeypatch.setattr(brain_module.requests, "post", raising(error))

    with pytest.raises(BrainError) as caught:
        ChatBrain.helper("https://h/v1", 5).complete(MESSAGES, None)

    assert str(caught.value) == f"the brain endpoint could not be reached ({type(error).__name__})"
    assert caught.value.spoken == CANT_REACH


def test_a_requests_error_that_quotes_the_header_never_reaches_the_message_the_chain_or_the_log(monkeypatch, caplog):
    # some requests versions quote the offending header value in the exception
    error = brain_module.requests.exceptions.InvalidHeader(f"Invalid header value: 'Bearer {KEY}'")
    monkeypatch.setattr(brain_module.requests, "post", raising(error))

    with caplog.at_level(logging.DEBUG), pytest.raises(BrainError) as caught:
        ChatBrain.helper("https://h/v1", 5, "m", KEY).complete(MESSAGES, None)

    exc = caught.value
    assert exc.__cause__ is None and exc.__suppress_context__  # not chained: a traceback would print the original
    for text in (str(exc), repr(exc), caplog.text, "".join(traceback.format_exception(exc))):
        assert KEY not in text


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

    with pytest.raises(BrainError, match="sent no message") as caught:
        ChatBrain(ENDPOINT, 5).complete(MESSAGES, None)

    assert caught.value.spoken == UNUSABLE


@pytest.mark.parametrize("message", [
    {"role": "assistant", "content": ["parts"]},
    {"role": "assistant", "content": "", "tool_calls": ["nope"]},
    {"role": "assistant", "content": "", "tool_calls": [{"id": "x", "function": {}}]},
    {"role": "assistant", "content": "", "tool_calls": [{"id": "x"}]},
])
def test_unexpected_content_or_a_malformed_call_raises(post, message):
    post.reply = completion(message)

    with pytest.raises(BrainError, match=r"unexpected content|malformed tool call") as caught:
        ChatBrain(ENDPOINT, 5).complete(MESSAGES, None)

    assert caught.value.spoken == UNUSABLE
