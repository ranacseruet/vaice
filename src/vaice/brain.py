"""The brain: one chat-completions request that answers or asks for tools.

Generic on purpose: any OpenAI-compatible `POST <base>/chat/completions` endpoint works (the local llama-server
at `<url>/v1`, LM Studio, a hosted API) and nothing here knows which one it is talking to. `ChatBrain` takes the
complete endpoint URL; `ChatBrain.local` and `ChatBrain.helper` build it from a base the way each is configured.

Every failure of the request is a `BrainError` carrying a fixed sentence the assistant can speak. The API key is
never put into an error, a log line or a repr: the only place it goes is the Authorization header.
"""
import json
import logging
from dataclasses import dataclass

import requests

log = logging.getLogger(__name__)

ERROR_BODY_CHARS = 200  # characters of a failed response put into the error message
CANT_REACH = "I can't reach my language model right now."
KEY_REFUSED = "I can't reach my language model: the key was refused."
BUSY = "My language model is busy right now. Try again in a moment."
UNUSABLE = "My language model sent an answer I couldn't use."


class BrainError(RuntimeError):
    """The brain could not answer. `str(exc)` is for the log and the UI; `spoken` is the short sentence to say."""

    def __init__(self, message: str, spoken: str = CANT_REACH):
        super().__init__(message)
        self.spoken = spoken


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: str  # the model's JSON string


@dataclass(frozen=True)
class BrainReply:
    text: str
    calls: list[ToolCall]
    message: dict  # the assistant message, to append to the conversation before the tool results


def chat_endpoint(base: str) -> str:
    """`<base>/chat/completions`, `base` being what OpenAI clients take as their base URL (`/v1` included)."""
    return f"{base.rstrip('/')}/chat/completions"


class ChatBrain:
    """`temperature` and `max_tokens` are sent only when not None: hosted models differ in which of them they
    accept, and a token cap cuts a reasoning model's answer short, so the helper sends neither."""

    def __init__(self, endpoint: str, timeout: float, model: str | None = None, api_key: str | None = None,
                 temperature: float | None = 0, max_tokens: int | None = 300):
        self._endpoint = endpoint
        self._timeout = timeout
        self._model = model
        self._api_key = api_key
        self._temperature = temperature
        self._max_tokens = max_tokens

    @classmethod
    def local(cls, url: str, timeout: float) -> "ChatBrain":
        """The local llama-server at `url` (scheme://host:port): temperature 0 and a 300-token cap, as measured."""
        return cls(chat_endpoint(f"{url.rstrip('/')}/v1"), timeout)

    @classmethod
    def helper(cls, base_url: str, timeout: float, model: str = "", api_key: str = "") -> "ChatBrain":
        """Any OpenAI-compatible endpoint at `base_url` (`/v1` included): only messages, tools and the model."""
        return cls(chat_endpoint(base_url), timeout, model or None, api_key or None, temperature=None,
                   max_tokens=None)

    def complete(self, messages: list[dict], tools: list[dict] | None) -> BrainReply:
        """One completion. `tools` are chat-completions definitions; none = a plain answer."""
        body: dict = {"messages": messages}
        if self._temperature is not None:
            body["temperature"] = self._temperature
        if self._max_tokens is not None:
            body["max_tokens"] = self._max_tokens
        if self._model:
            body["model"] = self._model
        if tools:
            body["tools"] = tools
        headers = {"Authorization": f"Bearer {self._api_key}"} if self._api_key else {}
        try:
            r = requests.post(self._endpoint, json=body, headers=headers, timeout=self._timeout)
        except requests.Timeout:
            raise BrainError(f"the brain endpoint did not answer within {self._timeout:g} s") from None
        except requests.RequestException as exc:
            # Not chained and without the exception's text: requests' InvalidHeader quotes the header value
            # (`utils._validate_header_part`, requests 2.34), i.e. "Bearer <key>"
            raise BrainError(f"the brain endpoint could not be reached ({type(exc).__name__})") from None
        if r.status_code in (401, 403):  # never the body: it can echo part of the key
            raise BrainError(f"the brain endpoint refused the API key (HTTP {r.status_code})", KEY_REFUSED)
        if r.status_code == 429:
            raise BrainError(f"the brain endpoint is rate limited (HTTP {r.status_code})", BUSY)
        if not r.ok:
            raise BrainError(f"chat completions answered HTTP {r.status_code}: {self._clip(r.text)}")
        try:
            choice = r.json()["choices"][0]
            message = choice["message"]
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise BrainError(f"chat completions sent no message: {self._clip(r.text)!r}", UNUSABLE) from exc
        if not isinstance(message, dict):
            raise BrainError(f"chat completions sent no message: {self._clip(r.text)!r}", UNUSABLE)
        reply = _reply(message)
        if choice.get("finish_reason") == "length" and not reply.text and not reply.calls:
            log.warning("the brain stopped at its token limit before it answered (finish_reason length)")
        return reply

    def _clip(self, text: str) -> str:
        """The start of a response body for an error message, with the key blanked out if the body echoes it."""
        if self._api_key:
            text = text.replace(self._api_key, "[key]")
        return text[:ERROR_BODY_CHARS]


def _reply(message: dict) -> BrainReply:
    content = message.get("content")
    if content is not None and not isinstance(content, str):
        raise BrainError(f"chat completions sent unexpected content: {str(content)[:ERROR_BODY_CHARS]!r}", UNUSABLE)
    calls = []
    for index, raw in enumerate(message.get("tool_calls") or []):
        function = raw.get("function") if isinstance(raw, dict) else None
        if not isinstance(function, dict) or not function.get("name"):
            raise BrainError(f"chat completions sent a malformed tool call: {str(raw)[:ERROR_BODY_CHARS]!r}", UNUSABLE)
        arguments = function.get("arguments")
        if arguments is None:
            arguments = ""
        elif not isinstance(arguments, str):  # some servers send the arguments as an object
            arguments = json.dumps(arguments)
        calls.append(ToolCall(str(raw.get("id") or f"call_{index}"), str(function["name"]), arguments))
    # What is sent back in later rounds holds only these keys: hosted endpoints reject unknown message fields,
    # and the reasoning (reasoning_content, reasoning), refusal and annotations must never be replayed
    normalized: dict = {"role": "assistant", "content": content or ""}
    if calls:  # as sent back, every call has an id and string arguments
        normalized["tool_calls"] = [
            {"id": call.id, "type": "function", "function": {"name": call.name, "arguments": call.arguments}}
            for call in calls]
    return BrainReply((content or "").strip(), calls, normalized)
