"""The brain: one chat-completions request that answers or asks for tools.

Generic on purpose: any OpenAI-compatible `POST {url}/v1/chat/completions` endpoint works, and nothing here
knows about llama-server.
"""
import json
from dataclasses import dataclass

import requests

ERROR_BODY_CHARS = 200  # characters of a failed response put into the error message


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


class ChatBrain:
    def __init__(self, url: str, timeout: float, model: str | None = None, api_key: str | None = None,
                 max_tokens: int = 300):
        self._endpoint = f"{url.rstrip('/')}/v1/chat/completions"
        self._timeout = timeout
        self._model = model
        self._api_key = api_key
        self._max_tokens = max_tokens

    def complete(self, messages: list[dict], tools: list[dict] | None) -> BrainReply:
        """One completion at temperature 0. `tools` are chat-completions definitions; none = a plain answer."""
        body: dict = {"messages": messages, "temperature": 0, "max_tokens": self._max_tokens}
        if self._model:
            body["model"] = self._model
        if tools:
            body["tools"] = tools
        headers = {"Authorization": f"Bearer {self._api_key}"} if self._api_key else {}
        r = requests.post(self._endpoint, json=body, headers=headers, timeout=self._timeout)
        if not r.ok:
            raise RuntimeError(f"chat completions answered HTTP {r.status_code}: {r.text[:ERROR_BODY_CHARS]}")
        try:
            message = r.json()["choices"][0]["message"]
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"chat completions sent no message: {r.text[:ERROR_BODY_CHARS]!r}") from exc
        if not isinstance(message, dict):
            raise RuntimeError(f"chat completions sent no message: {r.text[:ERROR_BODY_CHARS]!r}")
        return _reply(message)


def _reply(message: dict) -> BrainReply:
    content = message.get("content")
    if content is not None and not isinstance(content, str):
        raise RuntimeError(f"chat completions sent unexpected content: {str(content)[:ERROR_BODY_CHARS]!r}")
    calls = []
    for index, raw in enumerate(message.get("tool_calls") or []):
        function = raw.get("function") if isinstance(raw, dict) else None
        if not isinstance(function, dict) or not function.get("name"):
            raise RuntimeError(f"chat completions sent a malformed tool call: {str(raw)[:ERROR_BODY_CHARS]!r}")
        arguments = function.get("arguments")
        if arguments is None:
            arguments = ""
        elif not isinstance(arguments, str):  # some servers send the arguments as an object
            arguments = json.dumps(arguments)
        calls.append(ToolCall(str(raw.get("id") or f"call_{index}"), str(function["name"]), arguments))
    normalized = {**message, "role": "assistant", "content": content or ""}
    if calls:  # as sent back, every call has an id and string arguments
        normalized["tool_calls"] = [
            {"id": call.id, "type": "function", "function": {"name": call.name, "arguments": call.arguments}}
            for call in calls]
    else:
        normalized.pop("tool_calls", None)
    return BrainReply((content or "").strip(), calls, normalized)
