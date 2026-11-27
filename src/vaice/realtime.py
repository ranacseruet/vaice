"""The realtime session: llama-server's own speech, streamed from its /v1/realtime WebSocket.

Every reply is one realtime session: the user's audio in, text and speech out, and the assistant's skills declared as
tools that vaice runs (`_converse`). Speech with no reply in flight (what a Run click or a browser task has to say) is
a session whose input is a text message (`speak`)."""
import base64
import json
import logging
import queue
import re
import threading
import time
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from datetime import datetime
from functools import partial
from itertools import pairwise

import requests
from websockets.exceptions import ConnectionClosed, InvalidStatus, WebSocketException
from websockets.sync.client import ClientConnection, connect

from vaice import config, lifecycle
from vaice.audio import AudioFrame, from_pcm16, to_pcm16
from vaice.conversation import Message, Turn
from vaice.outbox import Outbox

log = logging.getLogger(__name__)

# Tells the model (LFM3-Audio and Omni's voice mode) to write its text alongside the speech; a fixed constant, so
# VAICE_PROMPT only has to say how to answer. The word is the model's own name for the mode: do not reword it
# without a live A/B.
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

# A reply that repeats the same block of 1-4 lines, or of 1-4 sentences within its lines, this many
# times back-to-back is degenerating: under greedy decoding LFM3-Audio sometimes cannot stop after
# answering and loops to the token cap (seconds of junk audio — a normalized timestamp ~60x, a
# 3-line block ~20x, one sentence 3x on one line with no newline (#156), live). The reply is cut
# there: the repeated tail is dropped from the record and the session closed, which also frees the
# server slot. 3 repeats keeps a legitimate refrain safe; the cut keeps the answer.
LOOP_REPEATS = 3
LOOP_MAX_BLOCK_LINES = 4
_SENTENCE_END = re.compile(r"[.!?]+")

ABANDONED = "the app is stopping; the reply is abandoned"
_DONE = object()
_ROUND = object()  # a tool-round boundary: the reply starts over after the tool results


def _ws_url(base: str) -> str:
    return "ws" + base.removeprefix("http") + "/v1/realtime"  # http -> ws, https -> wss


def _close(ws: ClientConnection) -> None:
    try:
        ws.close()
    except (OSError, WebSocketException) as exc:  # the session is over either way; nothing waits for this
        log.debug("closing the llama-server websocket failed: %s: %s", type(exc).__name__, exc)


@contextmanager
def _open(url: str, timeout: float) -> Iterator[ClientConnection]:
    """The websocket, as a context manager that closes it. `ping_interval=None`: the keep-alive
    pings would depend on llama-server's pong handling, and the idle timeout below already covers a
    server that went quiet.

    The close runs on a daemon thread of its own: llama-server (b12359) does not complete the
    close handshake, so `close()` waits out the whole `CLOSE_TIMEOUT`, and a caller that waited
    for it reported every transcript and every reply's end one second late (#90). The close frame
    itself is sent at once, so the server still cancels the reply and frees the slot promptly."""
    ws_url = _ws_url(url)
    try:
        ws = connect(ws_url, open_timeout=timeout, close_timeout=CLOSE_TIMEOUT, max_size=None, ping_interval=None)
    except (OSError, WebSocketException) as exc:  # refused, timed out, or not a websocket server at all
        raise RuntimeError(f"cannot open {ws_url}: {type(exc).__name__}: {exc}") from exc
    try:
        yield ws
    finally:
        threading.Thread(target=_close, args=(ws,), name="vaice-realtime-close", daemon=True).start()


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


class EndReply(str):
    """A tool output that ends the reply once it is sent: `_converse` gives the model the result but does not ask it
    to continue (no `response.create`), so it says nothing more. What starts a task that outlives the reply returns
    one (`assistant._Executor`): the model's continuation made up results for work that had only begun."""


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


# Simple tags: dropped outright. The closing ">" is optional on the thinking/answer names on purpose:
# a trailing or orphan `</think` (observed live) is dropped too. Closes of tool_call/function are
# dropped the same way. The lookahead keeps `<answerful>` from matching `<answer` + prose.
_TAG_SIMPLE = re.compile(
    r"</?(?:thinking|think|answer)(?![A-Za-z0-9])(=[^>\s]*)?>?"
    r"|</(?:tool_call|function)(?![A-Za-z0-9])(=[^>\s]*)?>?"
)
# Block openers must be complete (a ">"): they start dropping through their close tag. A truncated
# opener is never opened, so a split `<to` + `ol_call>` cannot leak its JSON body as prose either.
_OPEN_BLOCK = re.compile(r"<tool_call>|<function(=[^>\s]*)?>")
_CONTROL = re.compile(r"<\|.*?\|>")
# Simple tokens the stray-markup rules know about; used to hold back a trailing partial tag while streaming
_TOKEN_PREFIXES = ("<tool_call>", "</tool_call>", "<think>", "</think>", "<thinking>", "</thinking>",
                   "<answer>", "</answer>", "<function>", "</function>")


def _next_markup(text: str):
    """The leftmost complete markup token in `text`, as (start, end, token); None when there is none."""
    best = None
    for rx in (_CONTROL, _TAG_SIMPLE, _OPEN_BLOCK):
        m = rx.search(text)
        if m and (best is None or m.start() < best[0]):
            best = (m.start(), m.end(), m.group(0))
    return best


def _holdback(text: str) -> int:
    """The length of the trailing suffix of `text` that could grow into a markup token.

    Only such a suffix is withheld; everything else is emitted. A bare "<" holds back one char,
    "</think" holds back seven (an observed trailing leak), "<|im" holds back four, and so on.
    """
    for m in re.finditer(r"<", text):
        rem = text[m.start():]
        low = rem.lower()
        if any(prefix.lower().startswith(low) for prefix in _TOKEN_PREFIXES):
            return len(rem)
        if low.startswith(("<tool_call", "<function", "</function")) and ">" not in low:
            return len(rem)
        if low.startswith("<|") and "|>" not in low and len(low) <= 100:
            return len(rem)
    return 0


def _partial_suffix(text: str, target: str) -> int:
    """The length of the longest suffix of `text` that is a prefix of `target` (a split close tag)."""
    for n in range(min(len(text), len(target) - 1), 0, -1):
        if text.endswith(target[:n]):
            return n
    return 0


class _MarkupStripper:
    """Drops model markup from the streaming text, across deltas.

    LFM3-Audio leaks its tool/reasoning scaffolding into the text it speaks and writes: a raw
    `<tool_call>{...}</tool_call>` block, orphan `</think` fragments, stray `<|im_start|>`-style
    control tokens, the `<answer>` delimiters. The audio is unaffected; the recorded transcript
    and the UI are not. Tokens can be split across deltas, so the stripper holds back a trailing
    suffix that could grow into one; anything withheld is markup, and a reply that ends without it
    arriving simply never emits it. A `<tool_call>`/`<function=...>` block (which can span deltas)
    is dropped through its close tag; an unterminated block is dropped entirely.
    """

    def __init__(self) -> None:
        self._pending = ""
        self._drop_close: str | None = None

    def feed(self, delta: str) -> str:
        self._pending += delta
        out: list[str] = []
        while self._pending:
            if self._drop_close is not None:
                i = self._pending.find(self._drop_close)
                if i == -1:  # still inside the block: hold back only a split close tag
                    self._pending = self._pending[len(self._pending) - _partial_suffix(self._pending,
                                                                                    self._drop_close):]
                    return "".join(out)
                self._pending = self._pending[i + len(self._drop_close):]
                self._drop_close = None
                continue
            found = _next_markup(self._pending)
            if found is not None and found[1] == len(self._pending) and not found[2].endswith(">"):
                # an orphan tag sitting at the very end of what we have seen: it may still grow
                # (<answer -> <answer>), so hold it for the next delta instead of consuming it —
                # only a boundary (a ">" or any other character) settles it
                found = None
            if found is not None:
                s, e, tok = found
                if s:
                    out.append(self._pending[:s])
                self._pending = self._pending[e:]
                low = tok.lower()
                if low.startswith("<tool_call>"):
                    self._drop_close = "</tool_call>"
                elif low.startswith("<function") and low.endswith(">"):
                    self._drop_close = "</function>"
                continue  # a simple tag is dropped outright
            hb = _holdback(self._pending)
            if hb == 0:
                out.append(self._pending)
                self._pending = ""
            else:  # a partial token: hold it back for the next delta; a reply that ends never emits it
                out.append(self._pending[:len(self._pending) - hb])
                self._pending = self._pending[len(self._pending) - hb:]
            break
        return "".join(out)


def _normalized(line: str) -> str:
    """What makes two lines the same for loop detection: case and punctuation are noise (the
    observed loops repeat near-identical lines — "...EST" then "...EST."), spacing collapsed."""
    return " ".join("".join(ch.lower() if ch.isalnum() else " " for ch in line).split())


def _loop_block(units: Sequence[str], repeats: int = LOOP_REPEATS,
                max_block: int = LOOP_MAX_BLOCK_LINES) -> int | None:
    """The size, in units, of the block repeated `repeats` times back-to-back at the end of
    `units` (non-empty lines or sentences, compared normalized); None when there is no such block."""
    norm = [_normalized(unit) for unit in units]
    for size in range(1, max_block + 1):
        need = size * repeats
        if len(norm) >= need and norm[-need:] == norm[-size:] * repeats:
            return size
    return None


def _sentence_cuts(line: str) -> list[int]:
    """Where `line` splits into sentences: after a run of `.`, `!` or `?` that a space, a capital
    letter or a list marker follows ("…MacBook Pro.The Mac…", "…Pro.- Name"). A terminator with
    nothing after it yet is no cut, so while streaming the sentence still being written is not
    compared; a period inside "3.5 GB" or "e.g. this" is no cut either."""
    cuts = []
    for match in _SENTENCE_END.finditer(line):
        end = match.end()
        if end < len(line) and (line[end].isspace() or line[end].isupper() or line[end] in "-*#•"):
            cuts.append(end)
    return cuts


def _units(text: str, *, final: bool, sentences: bool) -> list[tuple[int, int]]:
    """The completed non-empty units of `text` as (start, end) offsets, whitespace trimmed: its
    lines, or with `sentences` its lines split at `_sentence_cuts`. A line is complete once its
    newline arrived, a sentence once its cut is seen; at the end of the response (`final`) the
    trailing unit counts too."""
    spans = []
    offset = 0
    for line in text.split("\n"):
        line_end = offset + len(line)
        complete = final or line_end < len(text)  # a newline follows it
        bounds = [0, *(_sentence_cuts(line) if sentences else []), len(line)]
        for i, (a, b) in enumerate(pairwise(bounds)):
            last = i == len(bounds) - 2
            if last and not complete:
                break  # the part still being written
            piece = line[a:b]
            if piece.strip():
                lead = len(piece) - len(piece.lstrip())
                spans.append((offset + a + lead, offset + a + len(piece.rstrip())))
        offset = line_end + 1
    return spans


def _loop_trimmed(joined: str, *, final: bool = False) -> str | None:
    """`joined` cut after the answer when it ends with a loop, or None. Repeated lines are looked
    for first, then repeated sentences, which also catches a sentence the model repeats on one
    line with no newline (#156). While streaming only completed units are compared, so a third
    "Status: ready" that is the start of "Status: ready for input" is not cut as a loop; at the
    end of the response the last unit counts too. The text before the cut is kept as written."""
    for sentences in (False, True):
        spans = _units(joined, final=final, sentences=sentences)
        units = [joined[a:b] for a, b in spans]
        block = _loop_block(units)
        if block is not None:
            kept = _trim_repeats(units, block)
            return joined[: spans[len(kept) - 1][1]]
    return None


def _trim_repeats(lines: list[str], block: int) -> list[str]:
    """`lines` (lines or sentences) without the repeated tail: every copy of the looping block
    (compared normalized) is dropped, so the record keeps the answer, not the loop. Only called
    after `_loop_block` has found at least `LOOP_REPEATS` copies at the end. One copy is kept when
    the whole text is the loop (a terse correct answer the model then loops on), so the turn still
    knows the answer."""
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
    heard: AudioFrame | str,
    out: "queue.Queue | _OutboxSink",
    cancelled: threading.Event,
    max_output_tokens: int,
    tools: list[dict] | None = None,
    executor: Callable[[str, str], str] | None = None,
    external_stop: Callable[[], bool] | None = None,
) -> None:
    """Daemon thread: one realtime session for one reply. Puts frames, `_ROUND` at each tool-round
    boundary, exceptions, and finally `(_DONE, (reply text, samples received))` on `out` (a queue, or anything else
    with `put`: `speak` hands it a sink). `heard` is the user's audio, or text to speak (`speak`: a message of the
    user's, which the model answers). `executor(name, arguments)` runs a tool call and returns what the model reads
    (an `EndReply` also ends the reply once the round's results are sent); it runs here on the session thread and
    never raises. `external_stop` is a stop of the caller's besides
    `cancelled` and the app's.

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
        return cancelled.is_set() or lifecycle.stopping.is_set() or (external_stop is not None and external_stop())

    acc = ["", 0]  # the current round's text and samples; a fresh one per round (frames keep the old)
    stripper = _MarkupStripper()  # fresh with each round, like acc: block state must not leak across rounds
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
            if isinstance(heard, str):
                # `response.create` with instructions alone fails (`conversation_empty`): there must be an item
                session.send({"type": "conversation.item.create", "item": {
                    "type": "message", "role": "user", "content": [{"type": "input_text", "text": heard}]}})
            else:
                session.send({"type": "input_audio_buffer.append", "audio": _b64(to_pcm16(heard, rate_in))})
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
                    acc[0] += stripper.feed(event.get("delta", ""))
                    trimmed = _loop_trimmed(acc[0])
                    if trimmed is not None:
                        # the reply cannot stop itself: record it without the repeated tail and end
                        # it here (closing the socket cancels the generation and frees the slot)
                        acc[0] = trimmed
                        log.warning("realtime reply fell into a repetition loop; cut it off after the answer")
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
                        log.warning("realtime reply fell into a repetition loop; cut it off after the answer")
                    had_calls = bool(calls)
                    _check_done(event, acc[1], max_output_tokens, had_calls)
                    if not calls or executor is None:
                        break
                    if rounds >= MAX_TOOL_ROUNDS:
                        # the last response was forced with tool_choice "none"; a call that still
                        # arrived is not executed, or the model could keep the loop alive forever
                        log.warning("realtime ignored a tool call in the forced final round (%s)", calls[0].name)
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
                    stripper = _MarkupStripper()
                    out.put(_ROUND)
                    ends = False
                    for call in calls:
                        log.info("realtime tool call %s(%s)", call.name, call.arguments)
                        if stop():
                            raise _Stopped
                        result = executor(call.name, call.arguments)
                        ends = ends or isinstance(result, EndReply)
                        session.send({"type": "conversation.item.create", "item": {
                            "type": "function_call_output", "call_id": call.call_id,
                            "output": _speech_friendly(result)}})
                    calls = []
                    if stop():  # don't ask the model to continue after the user or the app stopped
                        raise _Stopped
                    if ends:
                        # An `EndReply` in the round ends the reply here. The round's other calls ran and were
                        # recorded (the executor does that), but their results are not spoken: the model is not asked
                        # to continue. The reply has only the audio before the call, usually none, by design.
                        break
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
        log.warning("realtime reply hit the %d-token limit and was cut off", max_output_tokens)
    if not samples and not had_calls:
        # a tool-call response has no audio by design; the answer comes after the tool result
        log.warning("llama-server's reply had no audio (is the model one with audio output?)")


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


_SPECIAL_TOKEN = re.compile(r"<\|(?:(?!<\||\|>).)*\|>")
_LEGACY_MARKERS = ("@@USER@@", "@@REPLY@@")


def _clean_turn_text(said: str) -> str:
    """Strip special/control tokens and legacy turn markers iteratively so nested forms leave no residue."""
    while (stripped := _SPECIAL_TOKEN.sub(" ", said)) != said:
        said = stripped
    said = said.replace("<|", " ").replace("|>", " ")
    for marker in _LEGACY_MARKERS:
        while marker in said:
            said = said.replace(marker, "")
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


class RealtimeResponder:
    """Streams the reply of an audio-output model on llama-server (LFM3-Omni, LFM3-Audio) over /v1/realtime.

    Each reply is a fresh realtime session on a daemon thread of its own, which hands the audio to the
    generator through a queue; a second session transcribes the same audio in parallel. The instructions carry the
    current local time (`_context_lines`). Nothing here is bound to the thread that resumes the generator, so
    fastrtc may resume it from any.

    `tools` are declared in the session (the Responses API's flat format); `executor_for(turn, stop)` builds each
    reply's executor, which runs the calls on the session thread and carries that reply's stop signal. `prompt_extra()`
    is text appended to the prompt at the start of every reply (the user's notes).
    """

    def __init__(
        self,
        url: str = config.LLAMACPP_URL,
        prompt: str = config.PROMPT,
        timeout: float = config.LLAMACPP_TIMEOUT,
        max_output_tokens: int = config.LLAMACPP_MAX_OUTPUT_TOKENS,
        tools: list[dict] | None = None,
        executor_for: Callable[[Turn, Callable[[], bool]], Callable[[str, str], str]] | None = None,
        prompt_extra: Callable[[], str] | None = None,
        clock: bool = True,
    ):
        self._url = url
        self._prompt = prompt
        self._timeout = timeout
        # Text tokens and audio frames count together; the model card's own voice-chat example uses 1024
        self._max_output_tokens = max_output_tokens
        self._tools = tools or None
        self._executor_for = executor_for
        self._prompt_extra = prompt_extra
        # Switchable so a live A/B can isolate the clock line's effect
        self._clock = clock

    def respond(self, audio: AudioFrame, history: Sequence[Message], turn: Turn) -> Iterator[AudioFrame]:
        # First, so that nothing the reply does can keep the question from being transcribed
        threading.Thread(
            target=_transcribe_into,
            args=(self._url, self._timeout, audio, turn),
            name="vaice-realtime-asr",
            daemon=True,
        ).start()
        chunks: queue.Queue = queue.Queue()
        cancelled = threading.Event()

        def stop() -> bool:
            return cancelled.is_set() or lifecycle.stopping.is_set()

        # The executor is per reply: it carries this reply's stop signal, so a call in flight gives up when the
        # user barges in or the app stops
        executor = self._executor_for(turn, stop) if self._executor_for is not None and self._tools else None
        prompt = self._prompt + (self._prompt_extra() if self._prompt_extra is not None else "")
        context = _context_lines(clock=self._clock)
        threading.Thread(
            target=_converse,
            args=(self._url, self._timeout, _instructions(prompt, history, context), audio, chunks, cancelled,
                  self._max_output_tokens, self._tools, executor),
            name="vaice-realtime",
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


# What a text-input session is told: it speaks what it is given. Not measured live like the assistant's prompt: it
# only has to say the text, so there is no tool to call and nothing to decide.
SPEAK_PROMPT = ("You are a voice assistant. What the user's message tells you to say or report, you say, as "
                "spoken words: one or two short plain sentences, with no lists, symbols or file paths.")


class _OutboxSink:
    """Where `speak` puts a session's frames (it stands in for `_converse`'s queue): they go to the `Outbox`, and the
    text is recorded on the asking turn by a marker behind the last frame, so it is recorded only once its audio was
    served (a barge-in that clears the outbox leaves it unrecorded). A frame the outbox refuses (it was cleared since
    `generation`: speech start, a new conversation, Stop) or a stop ends the speech: `cancelled` is set, which
    `_converse` notices at its next event."""

    def __init__(self, outbox: Outbox, turn: Turn, generation: int, cancelled: threading.Event):
        self._outbox, self._turn, self.generation, self._cancelled = outbox, turn, generation, cancelled
        self.final: list = ["", 0]  # the session's last round: (text, samples)
        self.error: BaseException | None = None
        self.frames = 0

    def put(self, item: object) -> None:
        if isinstance(item, BaseException):
            self.error = item
        elif item is _ROUND:
            pass  # no tools are offered here, so there is no second round
        elif isinstance(item, tuple) and item[0] is _DONE:
            self.final = item[1]
        elif not self._cancelled.is_set():
            rate, frame, _acc = item  # type: ignore[misc]
            if self._outbox.put((rate, frame), self.generation):
                self.frames += 1
            else:
                self._cancelled.set()


def speak(url: str, timeout: float, max_output_tokens: int, outbox: Outbox, turn: Turn, request: str,
          stop: Callable[[], bool], generation: int | None = None) -> None:
    """Say `request` through the outbox, in a realtime session whose input is that text (no tools: it can never start
    more work), and record what was said on `turn` once it was served. Blocks until the session is over, on the
    caller's own daemon thread; best effort: a failure is a warning. `generation`: the outbox's, as the caller saw it
    when it began (default: now), so that a clear since then (the user spoke, "New conversation", Stop) drops this."""
    if stop():
        return
    cancelled = threading.Event()
    sink = _OutboxSink(outbox, turn, outbox.generation if generation is None else generation, cancelled)
    _converse(url, timeout, _instructions(SPEAK_PROMPT, []), request, sink, cancelled, max_output_tokens,
              external_stop=stop)
    if sink.error is not None:
        log.warning("could not speak: %s: %s", type(sink.error).__name__, sink.error)
    if sink.frames == 0 or cancelled.is_set() or stop():
        return
    text = spoken_part(sink.final[0], sink.final[1], sink.final[1])
    outbox.put_marker(partial(turn.append_assistant_text, text), generation=sink.generation)


def _warn_text_temperature(url: str) -> None:
    """Warn when llama-server samples reply text at a temperature above 0.

    The realtime API has no temperature field (`session.update` and `response.create` take only
    instructions, modalities, tools and max_output_tokens), so text tokens are sampled with the
    server's own sampling params, set only at launch. The LFM3-Audio model card recommends greedy
    decoding for text (`--temp 0`); at the default 0.8 a 2B audio model drifts off-topic and loses
    the conversation thread. The audio sampling params are separate (they come from the GGUF), so
    the speech sounds fine while the content is not. Non-fatal: a props problem must not break
    startup. When the temperature cannot be read — a router-style /props has no params, or the
    check itself fails — this logs a hint instead of staying silent: an absent warning must not
    be mistaken for "greedy is fine".
    """
    try:
        r = requests.get(f"{url}/props", timeout=CHECK_TIMEOUT)
        settings = r.json().get("default_generation_settings") or {}
        temp = (settings.get("params") or {}).get("temperature")
    except (requests.RequestException, ValueError, AttributeError) as exc:
        log.warning(
            "could not verify the server's temperature (/props failed: %s: %s); replies may drift — "
            "start llama-server with --temp 0", type(exc).__name__, exc)
        return
    if not isinstance(temp, (int, float)):
        log.warning(
            "could not verify the server's temperature (/props reports none, e.g. a router-style "
            "/props without params); replies may drift — start llama-server with --temp 0")
        return
    if temp > 0:
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
            "vaice needs an audio-output model such as LiquidAI/LFM3-Omni-2B-GGUF"
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
                "vaice needs a server with the realtime endpoint"
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


def spoken_part(text: str, total: int, handed_out: int) -> str:
    """The words of `text` that the first `handed_out` of `total` units of audio cover (approximate).

    The model writes its text about twice as fast as speaking speed, so the text is
    often complete after the first second of audio. Taking the text generated so far would record a
    whole reply that was cut off early; split it by the share of audio handed out instead. The units
    (frames, samples) are the caller's; `total` is the audio generated so far, not the final length.
    """
    if not total:  # text-only reply: nothing was said
        return ""
    if handed_out >= total:
        return text
    words = text.split()
    # truncate, don't round: the same conservative estimate as Turn.close(), which floors and keeps
    # at least one word because some audio played — words the user had no time to hear stay uncounted
    keep = max(1, int(len(words) * handed_out / total)) if handed_out else 0
    return " ".join(words[:keep])
