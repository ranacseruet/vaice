"""Assistant mode: separate steps instead of one interleaved pass.

audio -> transcribe (realtime ASR session) -> brain (chat completions + tools) <-> skills (run by vaice)
      -> voice.speakable() -> llama-server /tts, one sentence at a time -> audio frames

A skill that cannot be undone (`kind="confirm"`) does not run when the brain asks: the reply speaks a
question and the request waits in `confirm.Confirmations`. The next reply (the user's answer) or a click
in the UI settles it; the brain never interprets the answer.

Threading follows `LlamaCppResponder`: `respond()` starts one daemon thread that does all the blocking
work and hands frames to the generator through a queue, so fastrtc may resume the generator from any
worker thread and a server that never answers cannot keep the process alive.
"""
import logging
import queue
import re
import threading
import time
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field

import requests

from vaice import config, lifecycle, voice
from vaice.audio import from_pcm16
from vaice.brain import ChatBrain
from vaice.confirm import Confirmations, Pending, spoken_answer
from vaice.conversation import Message, Turn
from vaice.responders.base import AudioFrame
from vaice.responders.llamacpp import ABANDONED, _clock_line, _MarkupStripper, _Stopped, _transcribe
from vaice.shell import NEEDS_BUTTON
from vaice.skills import Question, SkillResult, Skills, default_skills

log = logging.getLogger(__name__)

POLL_SECONDS = 0.25  # how often the generator checks whether the reply was abandoned or the app is stopping
MAX_SKILL_ROUNDS = 3  # brain calls per reply; the last one is made without tools, so it must answer
TTS_CHUNK_BYTES = 8192  # read size of the streamed /tts body (about 0.17 s of speech)
NOT_HEARD = "Sorry, I didn't catch that."
NO_ANSWER = "Sorry, I couldn't work that out."
CANCELLED = "Okay, I won't."  # spoken when the user says no to a confirmation
_DONE = object()
_RATE = re.compile(r"\brate=(\d+)")


def _spoken_text(text: str) -> str:
    """The brain's answer without the tool-call markup the model sometimes writes as prose (seen live in the
    round made without tools: `<tool_call><function=open_web>...`), which would otherwise be spoken. The same
    stripper chat mode uses: `<tool_call>` / `<function=...>` blocks go through their close tag, an unterminated
    one to the end, stray `<think>`/`<answer>` tags and `<|...|>` control tokens with them. Empty when nothing
    is left."""
    cleaned = _MarkupStripper().feed(text).strip()
    if cleaned != text.strip():
        log.warning("assistant answer carried model markup, removed: %r", text[:200])
    return cleaned


@dataclass(frozen=True)
class SkillCall:
    name: str
    arguments: str
    result: SkillResult


@dataclass
class Decision:
    """What the brain made of the question: the text to speak, the skills it ran, seconds per brain call.

    `pending`: the brain asked for a `confirm` skill, which has not run; `answer` is then its question."""

    answer: str
    calls: list[SkillCall] = field(default_factory=list)
    brain_seconds: list[float] = field(default_factory=list)
    pending: Pending | None = None


def build_messages(prompt: str, history: Sequence[Message], transcript: str) -> list[dict]:
    """The chat for the brain: the system prompt with the clock, the earlier turns, the question.

    A `("tool", "<name>: <output>")` history entry (earlier skill results, see `tool_history`) becomes a
    real tool exchange, so the brain can answer a follow-up from it: one assistant message holding the
    turn's consecutive calls, then a tool message per call. Their ids (`h0`, `h1`, ...) are unique.
    """
    messages = [{"role": "system", "content": f"{prompt}\n{_clock_line()}"}]
    entries = list(history)
    index = 0
    while index < len(entries):
        role, text = entries[index]
        if role != "tool":
            messages.append({"role": role, "content": text})
            index += 1
            continue
        end = index
        while end < len(entries) and entries[end][0] == "tool":
            end += 1
        calls, results = [], []
        for n, (_, entry) in enumerate(entries[index:end], start=index):
            name, _, output = entry.partition(": ")
            calls.append({"id": f"h{n}", "type": "function", "function": {"name": name, "arguments": "{}"}})
            results.append({"role": "tool", "tool_call_id": f"h{n}", "content": output})
        messages.append({"role": "assistant", "content": "", "tool_calls": calls})
        messages += results
        index = end
    messages.append({"role": "user", "content": transcript})
    return messages


def decide(brain: ChatBrain, skills: Skills, messages: list[dict], stop: Callable[[], bool],
           on_call: Callable[[SkillCall], None] | None = None, *, turn: Turn) -> Decision:
    """The brain/skills loop: ask the brain, run the skills it asks for, let it answer from their results.

    Appends to `messages`. The last of the `MAX_SKILL_ROUNDS` brain calls is made without tools, so the
    loop always ends with an answer. `on_call` is told of each skill as it has run. Raises `_Stopped` when
    `stop()` turns true between steps; a brain failure propagates.

    A `confirm` skill does not run: the loop ends there with `Decision.pending` (asked by `turn`) and its
    question as the answer. Calls the brain made after it in the same message are dropped, so `messages`
    (kept in the `Pending`) is a valid chat that only needs the skill's result appended.
    """
    decision = Decision("")
    tools = skills.definitions()
    for round_ in range(1, MAX_SKILL_ROUNDS + 1):
        if stop():
            raise _Stopped
        final = round_ == MAX_SKILL_ROUNDS
        started = time.perf_counter()
        reply = brain.complete(messages, None if final else tools)
        decision.brain_seconds.append(time.perf_counter() - started)
        if final and reply.calls:
            log.warning("assistant ignored a skill call in the final round (%s)", reply.calls[0].name)
        if final or not reply.calls:
            decision.answer = _spoken_text(reply.text)
            break
        messages.append({**reply.message, "content": reply.message.get("content") or ""})
        asking = len(messages) - 1  # the assistant message; the tool results of earlier calls follow it
        for index, call in enumerate(reply.calls):
            if stop():
                raise _Stopped
            log.info("assistant skill %s(%s)", call.name, call.arguments)
            prepared = skills.prepare(call.name, call.arguments)
            if isinstance(prepared, Question):
                if index + 1 < len(reply.calls):
                    log.warning("assistant dropped %d skill call(s) requested after %s, which waits for confirmation",
                                len(reply.calls) - index - 1, call.name)
                    kept = messages[asking].get("tool_calls") or []
                    messages[asking] = {**messages[asking], "tool_calls": kept[: index + 1]}
                decision.pending = Pending(call.name, call.arguments, prepared.text, call.id, messages, turn,
                                           voice=prepared.voice)
                decision.answer = prepared.text
                return decision
            result = prepared if prepared is not None else skills.run(call.name, call.arguments)
            skill_call = SkillCall(call.name, call.arguments, result)
            decision.calls.append(skill_call)
            if on_call is not None:
                on_call(skill_call)
            messages.append({"role": "tool", "tool_call_id": call.id, "content": result.speech})
    decision.answer = decision.answer or NO_ANSWER
    return decision


def _tts(url: str, timeout: float, voice_name: str, text: str, stop: Callable[[], bool]) -> Iterator[AudioFrame]:
    """The speech for `text`, streamed from `POST {url}/tts` as 16-bit PCM; the sample rate is in the
    response's Content-Type (`audio/pcm;rate=24000;...`), never assumed. Raises `_Stopped` when told to stop."""
    r = requests.post(f"{url}/tts", json={"input": text, "voice": voice_name, "response_format": "pcm16",
                                          "stream": True}, stream=True, timeout=timeout)
    try:
        if not r.ok:
            raise RuntimeError(f"llama-server /tts answered HTTP {r.status_code}: {r.text[:200]}")
        content_type = r.headers.get("Content-Type", "")
        match = _RATE.search(content_type)
        if match is None:
            raise RuntimeError(f"llama-server /tts sent no sample rate (Content-Type: {content_type!r})")
        rate = int(match.group(1))
        carry = b""  # a chunk can end between the two bytes of a sample
        for chunk in r.iter_content(chunk_size=TTS_CHUNK_BYTES):
            if stop():
                raise _Stopped
            data = carry + chunk
            carry = data[len(data) - len(data) % 2:]
            data = data[: len(data) - len(carry)]
            if data:
                yield rate, from_pcm16(data)
        if carry:
            log.warning("llama-server /tts ended in the middle of a sample; dropped 1 byte")
    finally:
        r.close()


def _speak(url: str, timeout: float, voice_name: str, answer: str, out: queue.Queue, stop: Callable[[], bool],
           started: float) -> float | None:
    """Put the audio of `answer` on `out`, a sentence at a time, each frame with the text spoken so far
    (the sentences up to and including its own). Returns the seconds to the first audio, if there was any."""
    first: float | None = None
    said: list[str] = []
    for sentence in voice.sentences(voice.speakable(answer)):
        if stop():
            raise _Stopped
        said.append(sentence)
        text = " ".join(said)
        for rate, frame in _tts(url, timeout, voice_name, sentence, stop):
            if first is None:
                first = time.perf_counter() - started
            out.put((rate, frame, text))
    return first


def _approved(pending: Pending, skills: Skills, brain: ChatBrain, turn: Turn, stop: Callable[[], bool]) -> Decision:
    """The user said yes: run what was asked (the stored name and arguments, nothing the brain says now),
    record it on `turn`, and let the brain say in one tool-less call how it went."""
    if stop():
        raise _Stopped
    log.info("assistant skill %s(%s) approved by voice", pending.name, pending.arguments)
    result = skills.run(pending.name, pending.arguments, stop)  # a barge-in or the app ending stops a command
    turn.add_tool_event(pending.name, pending.arguments, result.detail or result.speech)
    if stop():
        raise _Stopped
    messages = [*pending.messages, {"role": "tool", "tool_call_id": pending.call_id, "content": result.speech}]
    started = time.perf_counter()
    reply = brain.complete(messages, None)
    return Decision(_spoken_text(reply.text) or result.speech, [SkillCall(pending.name, pending.arguments, result)],
                    [time.perf_counter() - started])


def _decision(prompt: str, history: Sequence[Message], transcript: str, skills: Skills, brain: ChatBrain,
              confirmations: Confirmations, turn: Turn, stop: Callable[[], bool]) -> Decision:
    """What to say to `transcript`. A request waiting for confirmation is settled first, and only by the
    exact words of `spoken_answer`: yes runs it (that utterance is no question for the brain), no cancels
    it, anything else drops it and is an ordinary question. A request that is not `voice` (a shell command
    that is not provably harmless) survives a yes: it is told to use the button, and stays pending."""
    pending = confirmations.current()
    if pending is not None:
        if not pending.voice and pending.armed_at is not None and spoken_answer(transcript) == "yes":
            return Decision(NEEDS_BUTTON)
        pending = confirmations.take(pending.id)  # atomic: a click that came in meanwhile wins, this is then a question
    if pending is not None:
        answer = spoken_answer(transcript) if pending.armed_at is not None else None
        if answer == "yes":
            return _approved(pending, skills, brain, turn, stop)
        if answer == "no":
            turn.add_tool_event(pending.name, pending.arguments, "cancelled")
            return Decision(CANCELLED)
        log.info("assistant dropped the request to run %s: not confirmed (armed: %s)", pending.name,
                 pending.armed_at is not None)
        pending.turn.add_tool_event(pending.name, pending.arguments, "not confirmed")
    return decide(brain, skills, build_messages(prompt, history, transcript), stop,
                  lambda call: turn.add_tool_event(call.name, call.arguments, call.result.detail or call.result.speech),
                  turn=turn)


def _work(url: str, timeout: float, prompt: str, voice_name: str, skills: Skills, brain: ChatBrain,
          confirmations: Confirmations, audio: AudioFrame, history: Sequence[Message], turn: Turn,
          out: queue.Queue, cancelled: threading.Event) -> None:
    """Daemon thread: one reply. Puts `(rate, frame, text so far)` tuples, then an exception if the reply
    failed, and finally `_DONE` on `out`. The question's transcript is reported on `turn` exactly once,
    whatever happens. Takes only plain data and its helpers, never the responder."""

    def stop() -> bool:
        return cancelled.is_set() or lifecycle.stopping.is_set()

    started = time.perf_counter()
    try:
        transcript = ""
        try:
            transcript = _transcribe(url, timeout, audio)
        finally:  # failed, stopped or heard: the next turn's history must never wait for this in vain
            turn.set_user_text(transcript)
        transcribed = time.perf_counter() - started
        if stop():
            raise _Stopped
        if transcript:
            decision = _decision(prompt, history, transcript, skills, brain, confirmations, turn, stop)
        else:
            decision = Decision(NOT_HEARD)
        pending = decision.pending
        if pending is not None:  # on the screen at once; answerable by voice only once the question was spoken
            confirmations.ask(pending)
            turn.add_tool_event(pending.name, pending.arguments, "waiting for confirmation")
        try:
            first = _speak(url, timeout, voice_name, decision.answer, out, stop, started)
        except BaseException:  # stopped or failed before the question was out: nobody heard it, nobody can say yes
            if pending is not None:
                confirmations.take(pending.id)
            raise
        if pending is not None:
            confirmations.arm(pending.id)  # not before: the question's own audio can echo into the mic
        if first is not None:
            thinking = "+".join(f"{seconds:.2f}s" for seconds in decision.brain_seconds) or "none"
            log.info("assistant timings: transcript %.2fs, brain %s, first audio %.2fs", transcribed, thinking, first)
    except _Stopped:
        pass
    except BaseException as exc:  # noqa: BLE001 - re-raised in the generator, which must see any failure
        if cancelled.is_set():
            log.warning("assistant reply failed after it was abandoned: %s: %s", type(exc).__name__, exc)
        out.put(exc)
    finally:
        out.put(_DONE)


class AssistantResponder:
    """Answers in separate steps (see the module docstring) and speaks through llama-server's /tts.

    Nothing here is bound to the thread that resumes the generator; the reply's thread is a daemon of its own.
    """

    name = "assistant"
    tool_history = True  # earlier skill results come in `history` as ("tool", "<name>: <output>"); see Responder

    def __init__(
        self,
        url: str = config.LLAMACPP_URL,
        timeout: float = config.LLAMACPP_TIMEOUT,
        prompt: str = config.ASSISTANT_PROMPT,
        voice: str = config.VOICE,
        skills: Skills | None = None,
        brain: ChatBrain | None = None,
        confirmations: Confirmations | None = None,
    ):
        self._url = url
        self._timeout = timeout
        self._prompt = prompt
        self._voice = voice
        self.skills = skills or default_skills(shell=config.ASSISTANT_SHELL)
        self._brain = brain or ChatBrain(url, timeout)
        self._confirmations = Confirmations() if confirmations is None else confirmations
        self._runs: set[threading.Event] = set()  # per click-approved run: set by `clear_pending` to stop it
        self._runs_lock = threading.Lock()

    # The UI's side of a confirmation (see Responder). These run on the caller's thread (a Gradio worker),
    # not on a reply's: there is no reply in flight to carry audio, so nothing is spoken on this path; the
    # outcome lands on the asking turn's tool events, which the Tool activity panel shows.

    def pending(self) -> Pending | None:
        """The request waiting for the user, if any."""
        return self._confirmations.current()

    def approve(self, pending_id: str) -> str | None:
        """Run the request `pending_id` names, if it is still the current one; its result text, else None.

        A skill that can be stopped (a shell command) runs for as long as it takes on this thread, which is a
        Gradio worker's: "New conversation" (`clear_pending`) or the app ending stops it; otherwise only the
        skill's own time limit does."""
        cancel = threading.Event()
        with self._runs_lock:  # before `take`, so a "New conversation" in between cannot miss this run
            self._runs.add(cancel)
        try:
            pending = self._confirmations.take(pending_id)
            if pending is None:
                return None
            result = self.skills.run(pending.name, pending.arguments,
                                     lambda: cancel.is_set() or lifecycle.stopping.is_set())
        finally:
            with self._runs_lock:
                self._runs.discard(cancel)
        pending.turn.add_tool_event(pending.name, pending.arguments, result.detail or result.speech)
        return result.speech

    def cancel(self, pending_id: str) -> None:
        pending = self._confirmations.take(pending_id)
        if pending is not None:
            pending.turn.add_tool_event(pending.name, pending.arguments, "cancelled")

    def clear_pending(self) -> None:
        """"New conversation": a request from the old one must not be approvable, and a command a click
        started for it stops."""
        self._confirmations.clear()
        with self._runs_lock:
            for cancel in self._runs:
                cancel.set()

    def respond(self, audio: AudioFrame, history: Sequence[Message], turn: Turn) -> Iterator[AudioFrame]:
        chunks: queue.Queue = queue.Queue()
        cancelled = threading.Event()
        threading.Thread(
            target=_work,
            args=(self._url, self._timeout, self._prompt, self._voice, self.skills, self._brain,
                  self._confirmations, audio, list(history), turn, chunks, cancelled),
            name="vaice-assistant",
            daemon=True,
        ).start()
        spoken = ""
        try:
            while True:
                if lifecycle.stopping.is_set():  # the app is stopping: don't wait for a server that may never answer
                    log.warning(ABANDONED)
                    return
                try:
                    item = chunks.get(timeout=POLL_SECONDS)
                except queue.Empty:
                    if turn.closed.is_set():  # barge-in, or the conversation was reset
                        return
                    continue
                if isinstance(item, BaseException):
                    raise item
                if item is _DONE:
                    if lifecycle.stopping.is_set():  # the reply thread may have stopped for the app first
                        log.warning(ABANDONED)
                    return
                if turn.closed.is_set():
                    return
                rate, frame, text = item
                if text != spoken:  # a new sentence starts: only now does the record include it
                    spoken = text
                    turn.set_assistant_text(text)
                yield rate, frame
        finally:
            cancelled.set()  # the reply's thread stops, and closes a streaming /tts response
