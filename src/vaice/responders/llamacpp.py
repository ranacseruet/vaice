"""llama.cpp backend: llama-server's own speech, streamed from its /v1/realtime WebSocket."""
import base64
import json
import logging
import queue
import threading
import time
from collections.abc import Callable, Iterator, Sequence
from datetime import datetime

import requests
from websockets.exceptions import ConnectionClosed, InvalidStatus, WebSocketException
from websockets.sync.client import ClientConnection, connect

from vaice import config, lifecycle
from vaice.audio import from_pcm16, to_pcm16
from vaice.conversation import Message, Turn
from vaice.responders.base import AudioFrame, spoken_part

log = logging.getLogger(__name__)

# Tells an interleaved-mode model (LFM3-Audio) to write its text alongside the speech; a fixed constant, so
# VAICE_LLAMACPP_PROMPT only has to say how to answer
MODE_INSTRUCTION = "Respond with interleaved text and audio."
# How often a waiting thread checks whether the reply was abandoned or the app is stopping
POLL_SECONDS = 0.25
# The close handshake of a socket whose server is busy or gone must not hold a thread for long
CLOSE_TIMEOUT = 1.0
CHECK_TIMEOUT = 2  # seconds for the startup HTTP checks
# The maximum number of tool rounds a reply may make after its first response. A tool round is one
# response that asked for tools, their execution, and the response that speaks the results; the cap
# keeps a tool-calling loop from running away with the session (and the mic).
MAX_TOOL_ROUNDS = 2

# A reply that repeats the same 1-4-line block this many times back-to-back is degenerating: under
# greedy decoding LFM3-Audio sometimes cannot stop after answering and loops to the token cap
# (seconds of junk audio — a normalized timestamp ~60x, a 3-line block ~20x, live). The reply is
# cut there: the repeated tail is dropped from the record and the session closed, which also frees
# the server slot. 3 repeats keeps a legitimate refrain safe; the cut keeps the answer.
LOOP_REPEATS = 3
LOOP_MAX_BLOCK_LINES = 4

ABANDONED = "the app is stopping; the reply is abandoned"
_DONE = object()
_ROUND = object()  # a tool-round boundary: the reply starts over after the tool results


def _ws_url(base: str) -> str:
    return "ws" + base.removeprefix("http") + "/v1/realtime"  # http -> ws, https -> wss


def _open(url: str, timeout: float):
    """The websocket, as a context manager that closes it. `ping_interval=None`: the keep-alive
    pings would depend on llama-server's pong handling, and the idle timeout below already covers a
    server that went quiet."""
    ws_url = _ws_url(url)
    try:
        return connect(ws_url, open_timeout=timeout, close_timeout=CLOSE_TIMEOUT, max_size=None, ping_interval=None)
    except (OSError, WebSocketException) as exc:  # refused, timed out, or not a websocket server at all
        raise RuntimeError(f"cannot open {ws_url}: {type(exc).__name__}: {exc}") from exc


class _Session:
    """One realtime session, used from a single daemon thread (see `_converse`).

    `receive()` waits in short slices so the thread can notice that it was told to stop, and fails
    when the server has been silent for longer than `timeout`.
    """

    def __init__(self, ws: ClientConnection, timeout: float, stop: Callable[[], bool]):
        self._ws = ws
        self._timeout = timeout
        self._stop = stop
        self._last_event = time.monotonic()

    def send(self, event: dict) -> None:
        self._ws.send(json.dumps(event))

    def receive(self) -> dict | None:
        """The next server event; None when told to stop. Raises RuntimeError on an `error` event,
        a closed socket or a silent server."""
        while True:
            if self._stop():
                return None
            try:
                raw = self._ws.recv(timeout=POLL_SECONDS)
            except TimeoutError:
                if time.monotonic() - self._last_event > self._timeout:
                    raise RuntimeError(f"llama-server sent nothing for {self._timeout:g}s") from None
                continue
            except ConnectionClosed as exc:
                raise RuntimeError(f"llama-server closed the connection: {exc}") from exc
            self._last_event = time.monotonic()
            try:
                event = json.loads(raw)
            except ValueError as exc:
                raise RuntimeError(f"unexpected llama-server message: {str(raw)[:200]!r}") from exc
            if event.get("type") == "error":
                error = event.get("error")
                message = error.get("message") if isinstance(error, dict) else error
                raise RuntimeError(f"llama-server error: {message or event}")
            return event

    def expect(self, event_type: str) -> dict | None:
        """The next `event_type` event, skipping the others; None when told to stop."""
        while (event := self.receive()) is not None:
            if event.get("type") == event_type:
                return event
        return None

    def rates(self) -> tuple[int, int]:
        """(input, output) sample rates, from session.created. The server decides them; they are
        the model's (16 kHz in, 24 kHz out for LFM3-Audio), never assumed here."""
        created = self.expect("session.created")
        if created is None:
            raise _Stopped
        try:
            audio = created["session"]["audio"]
            return int(audio["input"]["format"]["rate"]), int(audio["output"]["format"]["rate"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(f"session.created lacks the audio rates: {str(created)[:300]}") from exc

    def update(self, session: dict) -> None:
        self.send({"type": "session.update", "session": session})
        if self.expect("session.updated") is None:
            raise _Stopped


class _Stopped(Exception):  # control flow inside a session thread, not an error
    """The thread was told to stop while it waited for the server."""


# What a function_call_output may hold: the model reads the result aloud, so a long raw
# result becomes long spoken junk (paths, kernel strings, JSON)
TOOL_OUTPUT_CHARS = config.LLAMACPP_TOOL_OUTPUT_CHARS
TOOL_VALUE_CHARS = config.LLAMACPP_TOOL_VALUE_CHARS


def _short(text: str, chars: int) -> str:
    return text if len(text) <= chars else text[:chars].rstrip() + "..."


def _speech_friendly(output: str) -> str:
    """Compact a tool result for the model to speak.

    The result goes back into the conversation as text and the voice model tends to read it
    aloud, so what we feed back is roughly what gets said: a JSON object becomes short
    "key: value" pairs with truncated values, and the whole thing is capped. A plain-text payload
    (read_file's `plain_text_response`) is kept up to the output cap instead, because the answer
    can be at the end of a short file. A spoken "kernel version 25.6.0" is fine; a dictated uname
    string is not.
    """
    try:
        data = json.loads(output)
    except ValueError:
        data = None
    if isinstance(data, dict):
        if isinstance(data.get("plain_text_response"), str):
            # read_file's payload is one text blob; capping it at the per-value limit would cut the
            # end of a short file (a 210-char file lost its last word at 200), so the whole-output
            # cap bounds it instead
            output = data["plain_text_response"]
        else:
            output = "; ".join(
                f"{key}: {_short(value if isinstance(value, str) else json.dumps(value), TOOL_VALUE_CHARS)}"
                for key, value in data.items())
    elif isinstance(data, list):
        output = "; ".join(_short(value if isinstance(value, str) else json.dumps(value), TOOL_VALUE_CHARS)
                           for value in data[:5])
    return _short(output, TOOL_OUTPUT_CHARS)


class _ToolCall:
    """A function_call item the model asked for, filled in by the argument deltas."""

    def __init__(self, item_id: str, name: str, call_id: str):
        self.item_id = item_id
        self.name = name
        self.call_id = call_id
        self.arguments = ""


def _strip_tool_tags(delta: str) -> str:
    """Tool rounds leak the tool-call answer tags into the text deltas; the audio does not."""
    for tag in ("<answer>", "</answer>"):
        delta = delta.replace(tag, "")
    return delta


def _normalized(line: str) -> str:
    """What makes two lines the same for loop detection: case and punctuation are noise (the
    observed loops repeat near-identical lines — "...EST" then "...EST."), spacing collapsed."""
    return " ".join("".join(ch.lower() if ch.isalnum() else " " for ch in line).split())


def _loop_block(text: str, repeats: int = LOOP_REPEATS, max_block: int = LOOP_MAX_BLOCK_LINES) -> int | None:
    """The size, in non-empty lines, of the block repeated `repeats` times back-to-back at the end
    of `text`; None when there is no such block. Line-based: the observed loops repeat
    newline-separated lines ("2026-11-14 21:34:00 EST" ~60x, a 3-line block ~20x); repetition
    inside a single line is not detected."""
    lines = [line for line in text.split("\n") if line.strip()]
    norm = [_normalized(line) for line in lines]
    for size in range(1, max_block + 1):
        need = size * repeats
        if len(lines) >= need and norm[-need:] == norm[-size:] * repeats:
            return size
    return None


def _loop_trimmed(joined: str, *, final: bool = False) -> str | None:
    """`joined` without the repeated tail when it ends with a loop, or None. While streaming the
    trailing partial line is not a repeat yet — only completed lines (everything up to the last
    newline) are compared, so a third "Status: ready" that is the start of "Status: ready for
    input" is not cut as a loop. At the end of the response every line is final and the last one
    counts too."""
    if final:
        probe = joined if joined.endswith("\n") else joined + "\n"
    else:
        cut = joined.rfind("\n")
        if cut < 0:
            return None  # no completed line yet
        probe = joined[: cut + 1]
    block = _loop_block(probe)
    if block is None:
        return None
    lines = [line for line in probe.split("\n") if line.strip()]
    return "\n".join(_trim_repeats(lines, block))


def _trim_repeats(lines: list[str], block: int) -> list[str]:
    """`lines` without the repeated tail: every copy of the looping block (compared normalized) is
    dropped, so the record keeps the answer, not the loop. Only called after `_loop_block` has
    found at least `LOOP_REPEATS` copies at the end. One copy is kept when the whole text is the
    loop (a terse correct answer the model then loops on), so the turn still knows the answer."""
    norm = [_normalized(line) for line in lines]
    end = len(lines)
    copies = 1
    while True:
        left = end - (copies + 1) * block
        if left < 0:
            break
        if norm[left:left + block] != norm[left + block:left + 2 * block]:
            break
        copies += 1
    if copies == 1:  # nothing repeats (defensive: only called after _loop_block fired)
        return list(lines)
    if copies * block == len(lines):  # nothing before the loop: keep one copy
        return lines[:block]
    return lines[: end - copies * block]


def _create_response(session: "_Session", tool_choice: str | None = None) -> None:
    """Ask for a response, optionally overriding the session's tool choice for this response only.

    The session sets `tool_choice: "required"` so the first response follows a tool request through.
    Left to inherit, that also forces the response after a tool result to call a tool again: the
    model repeats the raw result and can loop until the token cap. `auto` lets it answer (or chain),
    and `none` guarantees the final answer at the round cap.
    """
    event: dict = {"type": "response.create"}
    if tool_choice is not None:
        event["response"] = {"tool_choice": tool_choice}
    session.send(event)


def _converse(
    url: str,
    timeout: float,
    instructions: str,
    audio: AudioFrame,
    out: queue.Queue,
    cancelled: threading.Event,
    max_output_tokens: int,
    tools: list[dict] | None = None,
    executor: Callable[[str, str], str] | None = None,
    turn: Turn | None = None,
) -> None:
    """Daemon thread: one realtime session for one reply. Puts frames, `_ROUND` at each tool-round
    boundary, exceptions, and finally `(_DONE, (reply text, samples received))` on `out`.

    Every frame carries a reference to its round's `acc = [text, samples]`, so the consumer never reads
    shared state the session thread keeps changing: a preamble frame dequeued after the thread has
    raced into the next round still pairs with that round's text against that round's final samples.

    A thread of our own, because fastrtc resumes the reply generator on whichever worker thread is
    free and the socket must not depend on any of them: connect() can block for the whole open
    timeout and close() for the close timeout, and a thread of fastrtc's executor that is stuck there
    keeps the process alive after Ctrl+C (the executor's threads are joined at exit). Takes only
    plain data, never the responder, so nothing here keeps it alive.
    """

    def stop() -> bool:
        return cancelled.is_set() or lifecycle.stopping.is_set()

    acc = ["", 0]  # the current round's text and samples; a fresh one per round (frames keep the old)
    try:
        with _open(url, timeout) as ws:
            session = _Session(ws, timeout, stop)
            rate_in, rate_out = session.rates()
            update = {
                "instructions": instructions,
                "output_modalities": ["text", "audio"],
                "max_output_tokens": max_output_tokens,
            }
            if tools:
                # "required" is what makes the model follow a tool request through (with "auto" it
                # announces the check and stops); ordinary questions still answer without a call
                update["tools"] = tools
                update["tool_choice"] = "required"
            session.update(update)
            session.send({"type": "input_audio_buffer.append", "audio": _b64(to_pcm16(audio, rate_in))})
            _create_response(session)
            calls: list[_ToolCall] = []
            rounds = 0
            while (event := session.receive()) is not None:
                kind = event.get("type")
                if kind == "response.output_item.added":
                    item = event.get("item") or {}
                    if item.get("type") == "function_call":
                        calls.append(_ToolCall(str(item.get("id", "")), str(item.get("name", "")),
                                               str(item.get("call_id", ""))))
                elif kind == "response.function_call_arguments.delta":
                    for call in calls:
                        if call.item_id == event.get("item_id"):
                            call.arguments += event.get("delta", "")
                elif kind == "response.output_text.delta":
                    acc[0] += _strip_tool_tags(event.get("delta", ""))
                    trimmed = _loop_trimmed(acc[0])
                    if trimmed is not None:
                        # the reply cannot stop itself: record it without the repeated tail and end
                        # it here (closing the socket cancels the generation and frees the slot)
                        acc[0] = trimmed
                        log.warning("llamacpp reply fell into a repetition loop; cut it off after the answer")
                        break
                elif kind == "response.output_audio.delta":
                    frame = from_pcm16(base64.b64decode(event.get("delta", "")))
                    if len(frame):
                        acc[1] += len(frame)
                        # the frame carries its round's accumulator, so the consumer records this
                        # round's text against this round's audio, whatever the thread does next
                        out.put((rate_out, frame, acc))
                elif kind == "response.done":
                    trimmed = _loop_trimmed(acc[0], final=True)
                    if trimmed is not None:  # the repeated tail only completed in the last line
                        acc[0] = trimmed
                        log.warning("llamacpp reply fell into a repetition loop; cut it off after the answer")
                    had_calls = bool(calls)
                    _check_done(event, acc[1], max_output_tokens, had_calls)
                    if not calls or executor is None:
                        break
                    if rounds >= MAX_TOOL_ROUNDS:
                        # the last response was forced with tool_choice "none"; a call that still
                        # arrived is not executed, or the model could keep the loop alive forever
                        log.warning("llamacpp ignored a tool call in the forced final round (%s)", calls[0].name)
                        break
                    # the model asked for tools: run them, feed the results back, and let it speak them
                    rounds += 1
                    # What was said before the tool result was working out loud, not the answer: it
                    # stays spoken but is dropped from the turn, so a wrong first pass cannot end up
                    # in history and reinforce itself. Text and the spoken_part share restart with
                    # this round's audio (a fresh accumulator; the frames from the previous round
                    # keep the old one); the marker goes out ahead of this round's frames, so the
                    # consumer's handed-out count restarts with them too.
                    acc = ["", 0]
                    out.put(_ROUND)
                    for call in calls:
                        log.info("llamacpp tool call %s(%s)", call.name, call.arguments)
                        if stop():
                            raise _Stopped
                        output = _speech_friendly(executor(call.name, call.arguments))
                        if turn is not None:  # the UI's tool panel; reported by the thread that ran the call,
                            turn.add_tool_event(call.name, call.arguments, output)  # so barge-in cannot lose it
                        session.send({"type": "conversation.item.create", "item": {
                            "type": "function_call_output", "call_id": call.call_id, "output": output}})
                    calls = []
                    if stop():  # don't ask the model to continue after the user or the app stopped
                        raise _Stopped
                    # after the first round, let the model chain; at the cap, force the answer
                    _create_response(session, "auto" if rounds < MAX_TOOL_ROUNDS else "none")
    except _Stopped:
        pass
    except BaseException as exc:  # noqa: BLE001 - re-raised in the consumer, which must see any failure
        out.put(exc)
    finally:
        out.put((_DONE, acc))


def _check_done(event: dict, samples: int, max_output_tokens: int, had_calls: bool = False) -> None:
    response = event.get("response") or {}
    status = response.get("status")
    if status != "completed":  # "cancelled": we never ask for it, so someone else stopped the reply
        raise RuntimeError(f"llama-server ended the reply with status {status!r}")
    # A reply that hits max_output_tokens still reports "completed"; the usage is what gives it away
    tokens = (response.get("usage") or {}).get("output_tokens")
    if isinstance(tokens, int) and tokens >= max_output_tokens:
        log.warning("llamacpp reply hit the %d-token limit and was cut off", max_output_tokens)
    if not samples and not had_calls:
        # a tool-call response has no audio by design; the answer comes after the tool result
        log.warning("llama-server's reply had no audio (is the model one with audio output?)")


def fetch_tools(url: str, timeout: float = CHECK_TIMEOUT) -> list[dict]:
    """Every tool `llama-server` advertises, in the Responses API format `session.update` expects.

    `GET {url}/tools` returns the chat-completions nesting (`definition.function.name`,
    `.description`, `.parameters`), and the realtime endpoint needs the flat form
    `{"type": "function", **definition["function"]}`. The set is the server operator's decision
    (`--tools`): what is advertised here is also what `POST /tools` executes, so vaice does not
    filter it. Connection problems raise `requests.RequestException` — the caller decides what to
    do about an unavailable server.
    """
    r = requests.get(f"{url}/tools", timeout=timeout)
    r.raise_for_status()
    return [{"type": "function", **entry["definition"]["function"]} for entry in r.json()]


def write_capable_tools(url: str, timeout: float = CHECK_TIMEOUT) -> list[str]:
    """Names of the advertised tools llama-server marks as able to change this machine.

    `GET {url}/tools` reports `permissions.write` (true for exec_shell_command, write_file,
    edit_file in llama-server's built-ins). vaice does not filter the tool set — the server's
    `--tools` is the gate — so this only feeds a startup warning. A request or shape problem
    raises, like `fetch_tools`; the caller decides whether a failed check is worth a warning.
    """
    r = requests.get(f"{url}/tools", timeout=timeout)
    r.raise_for_status()
    return [str(entry.get("tool")) for entry in r.json() if (entry.get("permissions") or {}).get("write")]


def execute_tool(url: str, timeout: float, stop: Callable[[], bool] | None = None) -> Callable[[str, str], str]:
    """What a `function_call` is worth: `POST {url}/tools` with `{"tool": name, "params": {...}}`.

    The server reads the tool's parameters from `params` (a JSON object), not from the model's
    `arguments` JSON string; sending the string left `read_file` without its `path` (the server
    answered "key 'path' not found") while parameterless tools such as `get_info` worked by
    accident. The result goes back as the function_call_output, so the model can speak it. A failure
    must be spoken too, not raised: this runs on the session thread, and the result (not the
    traceback) is what the model sees and what `app.py` shows in the UI. A closure over `url` and
    `timeout`, mirroring how `_open` builds the websocket.

    The request itself is not cancelled (the HTTP library has no safe way to interrupt a synchronous
    call), but the caller does not have to wait for it: the POST runs on a daemon thread and this
    stops waiting when `stop` (when given) fires or `timeout` passes — the reply gives up promptly
    and the result is discarded. The thread ends when the server answers or `timeout` expires; it
    is a daemon, so it cannot keep the process alive.
    """

    def call(name: str, arguments: str) -> str:
        try:
            params = json.loads(arguments) if arguments.strip() else {}
        except ValueError:
            return f"tool {name} failed: arguments are not JSON: {arguments!r}"
        if not isinstance(params, dict):
            return f"tool {name} failed: arguments are not a JSON object: {arguments!r}"
        if stop is not None and stop():
            return f"tool {name} aborted: the reply was stopped"
        # requests' timeout applies per socket operation and does not help here; instead, the call
        # runs on a daemon thread and this polls it — a healthy slow tool is never destroyed, and
        # the timeout bounds the caller either way
        result: dict[str, str] = {}

        def work() -> None:
            try:
                r = requests.post(f"{url}/tools", json={"tool": name, "params": params}, timeout=timeout)
                if not r.ok:
                    result["out"] = f"tool {name} failed: HTTP {r.status_code}: {str(r.text)[:200]}"
                    return
                result["out"] = str(r.text)
            except Exception as exc:  # noqa: BLE001 - a tool failure must be spoken, not raised; the thread must never die silently
                result["out"] = f"tool {name} failed: {exc}"

        worker = threading.Thread(target=work, name=f"vaice-tool-{name}", daemon=True)
        worker.start()
        deadline = time.monotonic() + timeout
        while worker.is_alive():
            if stop is not None and stop():
                return f"tool {name} aborted: the reply was stopped"
            if time.monotonic() > deadline:
                return f"tool {name} failed: timed out after {timeout:g}s"
            worker.join(POLL_SECONDS)
        return result.get("out", f"tool {name} failed: the request thread died")

    return call


def _b64(pcm: bytes) -> str:
    return base64.b64encode(pcm).decode("ascii")


def _transcribe(url: str, timeout: float, audio: AudioFrame) -> str:
    """What `audio` says, from a transcription session of its own.

    A conversation session never sends a transcript of the user's audio. No language is set, so the
    model detects it ("Perform ASR."), which reads English the same and does not pin the user to it.

    The handshake uses CHECK_TIMEOUT, not the reply path's `timeout`: a transcription is
    best-effort and degrades to an empty transcript, so against a server that accepts the
    connection but never answers the upgrade the transcription thread must not park itself for the
    whole open timeout.
    """
    with _open(url, CHECK_TIMEOUT) as ws:
        session = _Session(ws, timeout, lifecycle.stopping.is_set)
        rate_in, _ = session.rates()
        session.update({"type": "transcription"})
        session.send({"type": "input_audio_buffer.append", "audio": _b64(to_pcm16(audio, rate_in))})
        session.send({"type": "input_audio_buffer.commit"})
        event = session.expect("conversation.item.input_audio_transcription.completed")
    return str((event or {}).get("transcript", "")).strip()


def _transcribe_into(url: str, timeout: float, audio: AudioFrame, turn: Turn) -> None:
    """Daemon thread: report the transcript on `turn`, once, whatever happens."""
    text = ""
    try:
        text = _transcribe(url, timeout, audio)
    except _Stopped:
        pass
    except Exception:
        log.exception("transcribing the question failed; history will show it as inaudible")
    finally:  # an interrupted or failed reply must not leave the next turn waiting for this
        turn.set_user_text(text)


_CONTROL_TOKENS = ("<|im_start|>", "<|im_end|>", "@@USER@@", "@@REPLY@@")


def _clean_turn_text(said: str) -> str:
    for token in _CONTROL_TOKENS:
        said = said.replace(token, "")
    return " ".join(said.split())


def _clock_line() -> str:
    """Ambient context for every reply: the local date and time, in the machine's zone.

    The model has no clock, so "what time is it" would end in "I don't have a way" — or in a tool
    result's build timestamp read as the current time. Built per reply (each reply gets a fresh
    session), so it stays current without any tool. Phrased as something the model knows and is
    told to use: a bare "Current date and time: ..." line was treated as inert text — the model
    echoed it verbatim and still answered "I don't have access" (live A/B, 6 runs), while this
    wording answered with the real time 6/6.
    """
    moment = datetime.now().astimezone()
    stamp = moment.strftime(f"%A, %B {moment.day}, %Y at %I:%M %p %Z")
    return f"You know the current date and time: {stamp}. Use it when asked about the date or time."


def _context_lines(clock: bool = True) -> str:
    """The context line(s) that ride inside the system block, before the first ChatML turn."""
    return _clock_line() if clock else ""


def _instructions(prompt: str, history: Sequence[Message], context: str = "") -> str:
    """The system prompt, with the earlier turns as ChatML user/assistant turns.

    A realtime session cannot be handed earlier assistant turns (conversation.item.create rejects
    the assistant role), so each reply gets a fresh session. llama-server wraps `instructions` in
    `<|im_start|>system\\n...<|im_end|>\\n<|im_start|>user\\n` and tokenizes the header with
    `parse_special=true` (server-realtime.cpp), so closing the system block and emitting each
    past turn as `<|im_start|>{role}\\n...<|im_end|>` gives the model native ChatML turns instead
    of a flat transcript inside the system prompt (which makes 2B models confuse roles or repeat
    earlier replies). `Conversation` stays the one owner of history: windowing, barge-in cuts and
    "New conversation" all apply. `context` lines ride inside the system block, before the first
    ChatML turn.
    """
    text = f"{MODE_INSTRUCTION} {prompt}".strip()
    if context:
        text += "\n" + context
    for role, said in history:
        text += f"<|im_end|>\n<|im_start|>{role}\n{_clean_turn_text(said)}"
    return text


class LlamaCppResponder:
    """Streams the reply of an audio-output model on llama-server (LFM3-Audio) over /v1/realtime.

    Each reply is a fresh realtime session on a daemon thread of its own, which hands the audio to the
    generator through a queue; a second session transcribes the same audio in parallel. The
    instructions carry the current local time (`_context_lines`) — switchable, so the live A/B can
    isolate the line's effect. Nothing here is bound to the thread that resumes the generator, so
    fastrtc may resume it from any.
    """

    name = "llamacpp"

    def __init__(
        self,
        url: str = config.LLAMACPP_URL,
        prompt: str = config.LLAMACPP_PROMPT,
        timeout: float = config.LLAMACPP_TIMEOUT,
        max_output_tokens: int = config.LLAMACPP_MAX_OUTPUT_TOKENS,
        tools: list[dict] | None = None,
        executor: Callable[[str, str], str] | None = None,
        clock: bool = True,
    ):
        self._url = url
        self._prompt = prompt
        self._timeout = timeout
        # Text tokens and audio frames count together; the model card's own voice-chat example uses 1024
        self._max_output_tokens = max_output_tokens
        # The server's built-in tools (fetch_tools), declared in session.update and executed through
        # the server's /tools endpoint; None = no tools, the session.update of before. The server
        # decides which tools those are (--tools) and refuses to execute anything it did not
        # advertise, so the executor passes every call through. The default executor is built per
        # reply (respond()), because it carries that reply's stop signal.
        self._tools = tools or None
        self._executor = executor
        # Switchable so the live A/B can isolate the clock line's effect
        self._clock = clock

    def respond(self, audio: AudioFrame, history: Sequence[Message], turn: Turn) -> Iterator[AudioFrame]:
        # First, so that nothing the reply does can keep the question from being transcribed
        threading.Thread(
            target=_transcribe_into,
            args=(self._url, self._timeout, audio, turn),
            name="vaice-llamacpp-asr",
            daemon=True,
        ).start()
        chunks: queue.Queue = queue.Queue()
        cancelled = threading.Event()
        # The default executor is per reply: it carries this reply's stop signal, so an in-flight
        # tool call gives up when the user barges in or the app stops
        executor = self._executor
        if executor is None and self._tools:
            executor = execute_tool(self._url, self._timeout,
                                    stop=lambda: cancelled.is_set() or lifecycle.stopping.is_set())
        context = _context_lines(clock=self._clock)
        threading.Thread(
            target=_converse,
            args=(self._url, self._timeout, _instructions(self._prompt, history, context), audio, chunks, cancelled,
                  self._max_output_tokens, self._tools, executor, turn),
            name="vaice-llamacpp",
            daemon=True,
        ).start()
        handed_out = 0
        final = ["", 0]  # the session thread's last accumulator: (text, samples) of the final round
        try:
            while True:
                try:
                    item = chunks.get(timeout=POLL_SECONDS)
                except queue.Empty:
                    if lifecycle.stopping.is_set():
                        # The app is stopping: don't wait for a server that may never answer
                        log.warning(ABANDONED)
                        return
                    if turn.closed.is_set():  # barge-in, or the conversation was reset
                        return
                    continue
                if isinstance(item, BaseException):
                    raise item
                if item is _ROUND:  # a tool round ended: only what follows is the recorded answer
                    handed_out = 0
                    turn.set_assistant_text("")
                    continue
                if isinstance(item, tuple) and item[0] is _DONE:  # the session thread's last state
                    final = item[1]
                    break
                if turn.closed.is_set():
                    return
                _rate, frame, (round_text, round_samples) = item  # the frame's own snapshot
                handed_out += len(frame)
                turn.set_assistant_text(spoken_part(round_text, round_samples, handed_out))
                yield _rate, frame
            if lifecycle.stopping.is_set():  # the session thread may have noticed first, and stopped for the app
                log.warning(ABANDONED)
                return
            turn.set_assistant_text(spoken_part(final[0], final[1], handed_out))  # all of it, now that it's all out
        finally:
            cancelled.set()  # the session thread closes the socket, which cancels the reply on the server


def _warn_text_temperature(url: str) -> None:
    """Warn when llama-server samples reply text at a temperature above 0.

    The realtime API has no temperature field (`session.update` and `response.create` take only
    instructions, modalities, tools and max_output_tokens), so text tokens are sampled with the
    server's own sampling params, set only at launch. The LFM3-Audio model card recommends greedy
    decoding for text (`--temp 0`); at the default 0.8 a 2B audio model drifts off-topic and loses
    the conversation thread. The audio sampling params are separate (they come from the GGUF), so
    the speech sounds fine while the content is not. Non-fatal: a props problem must not break
    startup (a router-style /props has no params at all).
    """
    try:
        r = requests.get(f"{url}/props", timeout=CHECK_TIMEOUT)
        settings = r.json().get("default_generation_settings") or {}
        temp = (settings.get("params") or {}).get("temperature")
    except (requests.RequestException, ValueError, AttributeError):
        return
    if isinstance(temp, (int, float)) and temp > 0:
        log.warning(
            "llama-server samples reply text at temperature %g; the LFM3-Audio model card recommends "
            "greedy decoding for text (start llama-server with --temp 0). Higher temperatures make a "
            "2B audio model drift off-topic and lose the conversation thread", temp,
        )


def check_server(url: str = config.LLAMACPP_URL) -> None:
    """Check at startup that llama-server is up and serves an audio-output model.

    A server that is down or still loading is only a warning or note, because it may be up by the
    first question. A server that is up with a model that cannot speak raises, because no wait
    would fix that.
    """

    def unavailable(message: str) -> None:
        log.warning("%s; replies will fail until it is up", message)

    health = f"{url}/health"
    try:
        r = requests.get(health, timeout=CHECK_TIMEOUT)
    except requests.RequestException as exc:
        unavailable(f"llama-server not reachable at {health} ({exc})")
        return
    if r.status_code == 503:  # llama-server answers 503 "Loading model" until its weights are in
        log.info("llama-server at %s is still loading its model", health)
        return
    if not r.ok:
        unavailable(f"llama-server health check at {health} returned {r.status_code}: {r.text[:200]}")
        return

    models = f"{url}/v1/models"
    try:
        r = requests.get(models, timeout=CHECK_TIMEOUT)
        r.raise_for_status()
        entries = r.json()["data"]
        names = [str(m.get("id")) for m in entries]
        speaks = any("audio" in (m.get("architecture") or {}).get("output_modalities", []) for m in entries)
    except (requests.RequestException, ValueError, KeyError, TypeError, AttributeError) as exc:
        unavailable(f"could not read the model list at {models} ({type(exc).__name__}: {exc})")
        return
    if not speaks:
        raise RuntimeError(
            f"the model on llama-server at {url} ({', '.join(names) or 'none listed'}) does not output audio; "
            "the llamacpp backend needs an audio-output model such as LiquidAI/LFM3-Audio-2B-GGUF"
        )
    _probe_realtime(url, unavailable)
    _warn_text_temperature(url)


def _probe_realtime(url: str, unavailable: Callable[[str], None]) -> None:
    """Open the /v1/realtime WebSocket once, so a server without the endpoint fails at startup.

    The HTTP checks can all pass on a server that does not route /v1/realtime (an older
    llama.cpp, a proxy in front of it); without this probe that surfaces only at the first
    question, as a cannot-open error after a full open-timeout stall. A rejected handshake is
    permanent (no wait fixes a missing endpoint) and raises; a connection-level failure is
    transient (the server may be momentarily busy) and only warns, like the health check.
    """
    ws_url = f"{url.replace('http', 'ws', 1)}/v1/realtime"  # http→ws, https→wss
    try:
        connection = connect(ws_url, open_timeout=CHECK_TIMEOUT, close_timeout=CLOSE_TIMEOUT, max_size=None)
    except InvalidStatus as exc:
        status = getattr(getattr(exc, "response", None), "status_code", None)
        if status == 404:
            # The server answered "no such endpoint": permanent, no wait fixes it.
            raise RuntimeError(
                f"llama-server at {url} does not serve {ws_url} (handshake returned {status}); "
                "the llamacpp backend needs a server with the realtime endpoint"
            ) from exc
        # A 429/5xx/3xx from the server or a proxy in front of it is transient —
        # treat it like the other startup checks: warn and let the server recover
        # before the first question.
        unavailable(f"the realtime endpoint at {ws_url} answered with HTTP {status} ({exc})")
        return
    except (WebSocketException, OSError, TimeoutError) as exc:
        unavailable(f"the realtime endpoint at {ws_url} did not accept a connection ({type(exc).__name__}: {exc})")
        return
    connection.close()
