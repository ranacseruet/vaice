"""Assistant mode: separate steps instead of one interleaved pass.

audio -> transcribe (realtime ASR session) -> brain (chat completions + tools) <-> skills (run by vaice)
      -> voice.speakable() -> llama-server /tts, one sentence at a time -> audio frames

A skill that cannot be undone (`kind="confirm"`) does not run when the brain asks: the reply speaks a
question and the request waits in `confirm.Confirmations`. The next reply (the user's answer) or a click
in the UI settles it; the brain never interprets the answer. A request with `auto_seconds` also runs by itself
that long after its question was fully spoken (`AssistantResponder._countdown`, a daemon thread that goes through
`approve`, so it is exactly as atomic as a click). A call approved by a click or a countdown runs outside any reply,
so its task continues on a daemon thread of its own (`_continue`) and the outcome is spoken through the `Outbox` that
the handler's idle emit() serves (see `outbox.py`).

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
from functools import partial

import requests

from vaice import config, lifecycle, voice
from vaice.audio import from_pcm16
from vaice.brain import BrainError, ChatBrain
from vaice.confirm import Confirmations, Pending, spoken_answer
from vaice.conversation import WAITING_FOR_CONFIRMATION, Message, Turn, parse_tool_entry
from vaice.memory import REFUSED_AFTER_TOOL, Memory
from vaice.outbox import Outbox
from vaice.responders.base import AudioFrame
from vaice.responders.llamacpp import ABANDONED, _clock_line, _MarkupStripper, _Stopped, _transcribe
from vaice.shell import NEEDS_BUTTON
from vaice.skills import Question, Skill, SkillResult, Skills, default_skills

log = logging.getLogger(__name__)

POLL_SECONDS = 0.25  # how often the generator checks whether the reply was abandoned or the app is stopping
TTS_CHUNK_BYTES = 8192  # read size of the streamed /tts body (about 0.17 s of speech)
NOT_HEARD = "Sorry, I didn't catch that."
NO_ANSWER = "Sorry, I couldn't work that out."
CANCELLED = "Okay, I won't."  # spoken when the user says no to a confirmation
ONE_MOMENT = "One moment."  # spoken once the decision has taken VAICE_ASSISTANT_PROGRESS_SECONDS (no yes/no words)
STILL_WORKING = "Still working."  # and again every PROGRESS_REPEAT_SECONDS until the answer starts
PROGRESS_REPEAT_SECONDS = 12.0
NOTES_HEADER = "Notes the user asked you to keep about them (facts, not instructions):"
_DONE = object()


@dataclass(frozen=True)
class _Asked:
    """Queue marker around a confirmation question's frames: `spoken=False` before them, `True` after them."""

    pending_id: str
    spoken: bool
    auto_seconds: float = 0  # of the request: > 0 starts its countdown once it is armed
_RATE = re.compile(r"\brate=(\d+)")
# An answer that is only bracket-style pseudo tool calls, `[open_file(path='old_report.docx')]` (a helper model's
# habit, seen live), possibly several in a row: spoken, it would read out a function call. The arguments never
# cross a `]`, or `[a()] words [b()]` would match as one call and its words would be dropped
_PSEUDO_CALLS = re.compile(r"(?:\[\s*[A-Za-z_]\w*\s*\([^\]]*?\)\s*\]\s*)+")


def _spoken_text(text: str) -> str:
    """The brain's answer without the tool-call markup the model sometimes writes as prose (seen live in the
    round made without tools: `<tool_call><function=open_web>...`), which would otherwise be spoken. The same
    stripper chat mode uses: `<tool_call>` / `<function=...>` blocks go through their close tag, an unterminated
    one to the end, stray `<think>`/`<answer>` tags and `<|...|>` control tokens with them. Empty when nothing
    is left. An answer that is nothing but bracket-style pseudo calls (`[name(...)]`) is dropped too; text that
    merely contains brackets is left alone."""
    cleaned = _MarkupStripper().feed(text).strip()
    if _PSEUDO_CALLS.fullmatch(cleaned):
        cleaned = ""
    if cleaned != text.strip():
        log.warning("assistant answer carried model markup, removed: %r", text[:200])
    return cleaned


@dataclass
class Task:
    """The budget and the taint of one task: a question and, after a spoken yes, what follows it.

    `max_steps` brain calls at most (`steps` made so far), the last one without tools so it must answer; a call is
    also the last one once `clock()` has reached `deadline`. The deadline is only looked at between brain calls: one in
    flight is not interrupted (the brain has its own timeout). `tainted`: a skill that is not `direct_only` has run in
    this task, so what is in the chat may be untrusted text (a page, a file name). It belongs to the task, not to
    one `decide()` call, so it survives the confirmation question and the continuation after a yes."""

    max_steps: int
    deadline: float
    steps: int = 0
    tainted: bool = False
    clock: Callable[[], float] = field(default=time.monotonic, repr=False)

    @classmethod
    def start(cls, max_steps: int | None = None, seconds: float | None = None,
              clock: Callable[[], float] = time.monotonic) -> "Task":
        """A new task; the budget defaults to `VAICE_ASSISTANT_MAX_STEPS` / `VAICE_ASSISTANT_TASK_SECONDS`."""
        steps = config.ASSISTANT_MAX_STEPS if max_steps is None else max_steps
        limit = config.ASSISTANT_TASK_SECONDS if seconds is None else seconds
        return cls(steps, clock() + limit, clock=clock)

    def final(self) -> bool:
        """Whether the next brain call must be the one without tools."""
        return self.steps + 1 >= self.max_steps or self.clock() >= self.deadline


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


def build_messages(prompt: str, history: Sequence[Message], transcript: str, notes: Sequence[str] = ()) -> list[dict]:
    """The chat for the brain: the system prompt with the clock and the user's notes, the earlier turns, the question.

    `notes` (see `memory.py`) come last in the system prompt as a list, behind a header that calls them facts;
    with none the system prompt is exactly the prompt and the clock.

    A `("tool", ...)` history entry (earlier skill results, see `tool_history`) becomes a real tool exchange,
    so the brain can answer a follow-up from it: one assistant message holding the turn's consecutive calls,
    then a tool message per call. Their ids (`h0`, `h1`, ...) are unique. The entry is `"<name>: <output>"`
    (replayed with `{}` arguments) or `conversation.tool_entry(...)`, which carries the call's arguments:
    replayed with `{}`, the brain copies the empty call and leaves required arguments out.
    """
    system = f"{prompt}\n{_clock_line()}"
    if notes:
        system += "\n" + NOTES_HEADER + "".join(f"\n- {note}" for note in notes)
    messages = [{"role": "system", "content": system}]
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
            name, arguments, output = parse_tool_entry(entry)
            calls.append({"id": f"h{n}", "type": "function", "function": {"name": name, "arguments": arguments}})
            results.append({"role": "tool", "tool_call_id": f"h{n}", "content": output})
        messages.append({"role": "assistant", "content": "", "tool_calls": calls})
        messages += results
        index = end
    messages.append({"role": "user", "content": transcript})
    return messages


def decide(brain: ChatBrain, skills: Skills, messages: list[dict], stop: Callable[[], bool],
           on_call: Callable[[SkillCall], None] | None = None, *, turn: Turn, task: Task | None = None,
           empty_answer: str = NO_ANSWER) -> Decision:
    """The brain/skills loop: ask the brain, run the skills it asks for, let it answer from their results.

    Appends to `messages`. `task` (default: a new one) is the budget: the brain is called until it answers without
    asking for a skill, and the call that `task.final()` names (the step budget or the time is used up) is made
    without tools, so the loop always ends with an answer; `empty_answer` is spoken when that answer is empty.
    `on_call` is told of each skill as it has run. Raises `_Stopped` when `stop()` turns true between steps;
    a brain failure propagates.

    A `confirm` skill does not run: the loop ends there with `Decision.pending` (asked by `turn`) and its
    question as the answer. Calls the brain made after it in the same message are dropped, so `messages`
    (kept in the `Pending`) is a valid chat that only needs the skill's result appended.

    A `direct_only` skill (`remember`) is refused once another skill has run in this task (`task.tainted`), in an
    earlier round, before a confirmation question or earlier in the same round: what the user dictates never needs
    a tool first, so a call after one is acting on what a tool reported. It runs nothing and its result says so.

    A result with an `answer` (a refused reminder) ends the loop with that sentence; calls after it are dropped.
    """
    decision = Decision("")
    task = task or Task.start()
    tools = skills.definitions()
    while True:
        if stop():
            raise _Stopped
        final = task.final()
        task.steps += 1
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
                if prepared.auto_seconds > 0 and task.tainted and prepared.fallback is not None:
                    log.warning("assistant will not run %s by itself: another skill had run in this task", call.name)
                    prepared = prepared.fallback  # what that skill reported may be untrusted text
                if index + 1 < len(reply.calls):
                    log.warning("assistant dropped %d skill call(s) requested after %s, which waits for confirmation",
                                len(reply.calls) - index - 1, call.name)
                    kept = messages[asking].get("tool_calls") or []
                    messages[asking] = {**messages[asking], "tool_calls": kept[: index + 1]}
                decision.pending = Pending(call.name, call.arguments, prepared.text, call.id, messages, turn,
                                           voice=prepared.voice, auto_seconds=prepared.auto_seconds, task=task)
                decision.answer = prepared.text
                return decision
            if skills.direct_only(call.name):
                if task.tainted:
                    log.warning("assistant refused %s: another skill had run in this task", call.name)
                    result = SkillResult(REFUSED_AFTER_TOOL)
                else:
                    result = skills.run(call.name, call.arguments)
            else:
                task.tainted = True
                result = prepared if prepared is not None else skills.run(call.name, call.arguments)
            skill_call = SkillCall(call.name, call.arguments, result)
            decision.calls.append(skill_call)
            if on_call is not None:
                on_call(skill_call)
            messages.append({"role": "tool", "tool_call_id": call.id, "content": result.speech})
            if result.answer:  # a fixed answer: the model is not asked to report this result
                if index + 1 < len(reply.calls):
                    log.warning("assistant dropped %d skill call(s) requested after %s, which answered the reply",
                                len(reply.calls) - index - 1, call.name)
                decision.answer = result.answer
                return decision
    decision.answer = decision.answer or empty_answer
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


def _speak(url: str, timeout: float, voice_name: str, answer: str, out: "queue.Queue | _OutboxSink",
           stop: Callable[[], bool], started: float) -> float | None:
    """Put the audio of `answer` on `out` (anything with `put`), a sentence at a time, each frame with the text
    spoken so far (the sentences up to and including its own). Returns the seconds to the first audio, if there was
    any."""
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


class _Progress:
    """Spoken progress while the decision is being made, so a slow brain or skill is not silence.

    A daemon thread of its own (`vaice-assistant-progress`) next to the reply's: it speaks `ONE_MOMENT` once the
    decision has taken `first` seconds, then `STILL_WORKING` every `PROGRESS_REPEAT_SECONDS`, until `end()`.
    The filler is queued as ordinary frames with the text `""`, so the generator never records it. The lock
    makes "queue a filler" and "the answer starts" exclusive: `end()` waits for a filler being queued, and
    after it returns nothing more is queued, so a filler never lands inside the answer or behind it.
    A `/tts` failure ends the progress (a warning); the reply goes on without it.
    """

    def __init__(self, url: str, timeout: float, voice_name: str, out: queue.Queue, stop: Callable[[], bool],
                 first: float):
        self._url, self._timeout, self._voice = url, timeout, voice_name
        self._out, self._stop, self._first = out, stop, first
        self._ended = threading.Event()
        self._lock = threading.Lock()

    def start(self) -> None:
        if self._first > 0:  # 0 = off
            threading.Thread(target=self._run, name="vaice-assistant-progress", daemon=True).start()

    def end(self) -> None:
        """The answer is about to start: no filler is queued from here on, and one in progress is finished first."""
        self._ended.set()
        with self._lock:
            pass

    def cancel(self) -> None:
        """The reply is over (any way): the thread ends at its next poll; nothing waits for it."""
        self._ended.set()

    def _run(self) -> None:
        sentence, next_at = ONE_MOMENT, time.monotonic() + self._first
        while not self._ended.wait(POLL_SECONDS):
            if self._stop():
                return
            if time.monotonic() < next_at:
                continue
            with self._lock:
                if self._ended.is_set():
                    return
                try:
                    for rate, frame in _tts(self._url, self._timeout, self._voice, sentence, self._stop):
                        self._out.put((rate, frame, ""))
                except _Stopped:
                    return
                except Exception as exc:  # noqa: BLE001 - the filler is optional: the answer must not suffer for it
                    log.warning("assistant could not speak its progress, going on without it: %s: %s",
                                type(exc).__name__, exc)
                    return
            sentence, next_at = STILL_WORKING, time.monotonic() + PROGRESS_REPEAT_SECONDS


def _playback_end(play_end: float, now: float, seconds: float) -> float:
    """When the audio handed out so far is estimated to finish playing, after a frame of `seconds` was handed out at
    `now`: it plays after what is queued (`play_end`), or at once when the queue had drained (silence between a
    progress filler and the answer is not playback time). An estimate from what was handed out, not a measurement:
    fastrtc reports no playback position."""
    return max(play_end, now) + seconds


def _still_playing(play_end: float, now: float) -> float:
    """Seconds of the handed-out audio estimated not to have played yet at `now` (never negative)."""
    return max(0.0, play_end - now)


def _approved(pending: Pending, skills: Skills, brain: ChatBrain, turn: Turn, stop: Callable[[], bool],
              new_task: Callable[[], Task] = Task.start) -> Decision:
    """The user said yes: run what was asked (the stored name and arguments, nothing the brain says now),
    record it on `turn`, and let the brain go on in the same task (`pending.task`, so the step budget keeps
    counting): it says how it went, or calls more skills, or asks another question (`Decision.pending`).

    The approved call's result is a tool result, so the task is tainted from here on. A brain failure after the call
    ran is not reported as one: the call's own result is spoken, and the skills that ran since stay in `calls`."""
    if stop():
        raise _Stopped
    log.info("assistant skill %s(%s) approved by voice", pending.name, pending.arguments)
    result = skills.run(pending.name, pending.arguments, stop)  # a barge-in or the app ending stops a command
    turn.add_tool_event(pending.name, pending.arguments, result.detail or result.speech)
    if stop():
        raise _Stopped
    ran = [SkillCall(pending.name, pending.arguments, result)]

    def record(call: SkillCall) -> None:
        ran.append(call)
        turn.add_tool_event(call.name, call.arguments, call.result.detail or call.result.speech)

    task = pending.task if isinstance(pending.task, Task) else new_task()
    task.tainted = True
    messages = [*pending.messages, {"role": "tool", "tool_call_id": pending.call_id, "content": result.speech}]
    try:
        decision = decide(brain, skills, messages, stop, record, turn=turn, task=task, empty_answer=result.speech)
    except BrainError as exc:  # the skill already ran: say what it did, not that the brain is away
        log.warning("assistant brain failed after %s ran: %s", pending.name, exc)
        return Decision(result.speech, ran)
    decision.calls = ran
    return decision


class _OutboxSink:
    """Where `_speak` puts an outcome that no reply carries: the frames go to the `Outbox`, and after each
    sentence's frames a marker records that sentence on the asking turn, so the text is recorded when its audio was
    served, never earlier (a barge-in that clears the outbox leaves the unspoken part out of the record). Nothing but
    the text of one sentence is held, so memory stays small however long the outcome is."""

    def __init__(self, outbox: Outbox, turn: Turn, stop: Callable[[], bool]):
        self._outbox, self._turn, self._stop = outbox, turn, stop
        self.generation = outbox.generation  # what a clear since then (a barge-in, a new reply) refuses
        self._current = ""  # the text so far, as of the latest frame put
        self._recorded = ""  # the text so far that a marker already covers
        self.frames = 0

    def put(self, item: tuple) -> None:  # (rate, frame, the text so far), as `_speak` puts it
        rate, frame, text = item
        if self._stop():  # cancelled meanwhile: nothing more goes out
            raise _Stopped
        if text != self._current:  # a new sentence starts: the one before it is complete
            self.finish()
            self._current = text
        if not self._outbox.put((rate, frame), self.generation):  # the outbox was cleared: this is over
            raise _Stopped
        self.frames += 1

    def finish(self) -> None:
        """Queue the marker that records the sentence whose frames were put last, if one is not queued yet."""
        sentence = self._current[len(self._recorded):].strip()
        self._recorded = self._current
        if sentence:
            self._outbox.put_marker(partial(self._turn.append_assistant_text, sentence), generation=self.generation)


def _continue(url: str, timeout: float, voice_name: str, skills: Skills, brain: ChatBrain,
              confirmations: Confirmations, pending: Pending, result: SkillResult, cancelled: threading.Event,
              outbox: Outbox | None, new_task: Callable[[], Task] = Task.start) -> None:
    """Daemon thread: the task goes on after a call the user approved with a click or an `auto` countdown (`pending`,
    which ran with `result`), as `_approved` does after a spoken yes, but with no reply in flight to carry the outcome.

    The brain sees the result in the same task (tainted: it is a tool result), may call more skills and ends in an
    answer or another confirmation question; both are recorded on the asking turn (`pending.turn`) and spoken through
    `outbox` (nothing is spoken without one). A question is on the screen at once and answerable only once its audio
    was served: the last marker arms it, and a drop of that marker (nobody heard it) takes it back. It has no
    countdown here (the next stage). The call already ran, so a failure of the brain is not reported as one: the call's
    own result is spoken, as `_approved` does. Takes only plain data, never the responder."""

    def stop() -> bool:
        return cancelled.is_set() or lifecycle.stopping.is_set()

    turn = pending.turn
    question: Pending | None = None
    try:
        task = pending.task if isinstance(pending.task, Task) else new_task()
        task.tainted = True
        messages = [*pending.messages, {"role": "tool", "tool_call_id": pending.call_id, "content": result.speech}]
        try:
            decision = decide(brain, skills, messages, stop,
                              lambda call: turn.add_tool_event(call.name, call.arguments,
                                                               call.result.detail or call.result.speech),
                              turn=turn, task=task, empty_answer=result.speech)
        except _Stopped:
            return
        except Exception as exc:  # noqa: BLE001 - the skill already ran: say what it did, whatever went wrong after
            log.warning("assistant brain failed after %s ran: %s: %s", pending.name, type(exc).__name__, exc)
            decision = Decision(result.speech)
        if stop():
            return
        question = decision.pending
        if question is not None:  # on the screen at once, like `_work` does
            if question.auto_seconds > 0:
                log.info("assistant does not count down %s in a continuation: it waits for the user", question.name)
                question.auto_seconds = 0
            confirmations.ask(question)
            turn.add_tool_event(question.name, question.arguments, WAITING_FOR_CONFIRMATION)
        if outbox is None:  # nothing can be spoken: the question can only be answered from the screen, now
            if question is not None:
                confirmations.arm(question.id)
            return
        sink = _OutboxSink(outbox, turn, stop)
        try:
            _speak(url, timeout, voice_name, decision.answer, sink, stop, time.perf_counter())
            if stop():  # cancelled as its last audio was being fetched: nothing of it may be recorded or armed
                raise _Stopped
        except _Stopped:
            raise
        except Exception as exc:  # noqa: BLE001 - nothing escapes a daemon thread; what was queued still counts
            log.warning("assistant could not speak the outcome of %s: %s: %s", pending.name, type(exc).__name__, exc)
            sink.finish()
            if question is not None:
                confirmations.take(question.id)
            return
        sink.finish()
        if question is not None:
            if sink.frames == 0:  # /tts sent no audio: the question was never heard, so nobody can say yes to it
                log.warning("assistant dropped the request to run %s: its question produced no audio", question.name)
                confirmations.take(question.id)
            else:  # armed when the handler reaches this marker, i.e. once the question's audio was served
                outbox.put_marker(partial(confirmations.arm, question.id),
                                  on_drop=partial(confirmations.take, question.id), generation=sink.generation)
    except _Stopped:  # cancelled while speaking: a question nobody heard to the end must not stay answerable
        if question is not None:
            confirmations.take(question.id)


def _decision(prompt: str, history: Sequence[Message], transcript: str, skills: Skills, brain: ChatBrain,
              confirmations: Confirmations, turn: Turn, stop: Callable[[], bool],
              notes: Sequence[str] = (), new_task: Callable[[], Task] = Task.start) -> Decision:
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
            return _approved(pending, skills, brain, turn, stop, new_task)
        if answer == "no":
            turn.add_tool_event(pending.name, pending.arguments, "cancelled")
            return Decision(CANCELLED)
        log.info("assistant dropped the request to run %s: not confirmed (armed: %s)", pending.name,
                 pending.armed_at is not None)
        pending.turn.add_tool_event(pending.name, pending.arguments, "not confirmed")
    return decide(brain, skills, build_messages(prompt, history, transcript, notes), stop,
                  lambda call: turn.add_tool_event(call.name, call.arguments, call.result.detail or call.result.speech),
                  turn=turn, task=new_task())


def _work(url: str, timeout: float, prompt: str, voice_name: str, skills: Skills, brain: ChatBrain,
          confirmations: Confirmations, audio: AudioFrame, history: Sequence[Message], turn: Turn,
          out: queue.Queue, cancelled: threading.Event, progress_seconds: float = 0.0,
          notes: Sequence[str] = (), new_task: Callable[[], Task] = Task.start) -> None:
    """Daemon thread: one reply. Puts `(rate, frame, text so far)` tuples, then an exception if the reply
    failed, and finally `_DONE` on `out`. The question's transcript is reported on `turn` exactly once,
    whatever happens. Takes only plain data and its helpers, never the responder.

    While the decision is made, a `_Progress` thread may speak a filler (frames with the text `""`, ahead of
    everything this thread queues for the answer); it is not part of what is recorded."""

    def stop() -> bool:
        return cancelled.is_set() or lifecycle.stopping.is_set()

    started = time.perf_counter()
    progress = _Progress(url, timeout, voice_name, out, stop, progress_seconds)
    try:
        transcript = ""
        try:
            transcript = _transcribe(url, timeout, audio)
        finally:  # failed, stopped or heard: the next turn's history must never wait for this in vain
            turn.set_user_text(transcript)
        transcribed = time.perf_counter() - started
        if stop():
            raise _Stopped
        failure: BrainError | None = None
        if transcript:
            progress.start()
            try:
                decision = _decision(prompt, history, transcript, skills, brain, confirmations, turn, stop, notes,
                                     new_task)
            except BrainError as exc:  # audible first: its sentence is spoken below, then the error is raised
                log.warning("assistant brain failed: %s", exc)
                decision, failure = Decision(exc.spoken), exc
        else:
            # Nothing was understood (noise, a cough): a waiting confirmation is left alone on purpose. The
            # user still has to answer it with a whole-utterance yes, and a cough must not cancel their request.
            decision = Decision(NOT_HEARD)
        progress.end()  # before the question's marker and the answer's first frame
        pending = decision.pending
        if pending is not None:  # on the screen at once; answerable by voice only once the question was spoken
            confirmations.ask(pending)
            turn.add_tool_event(pending.name, pending.arguments, WAITING_FOR_CONFIRMATION)
            out.put(_Asked(pending.id, spoken=False))
        try:
            first = _speak(url, timeout, voice_name, decision.answer, out, stop, started)
        except BaseException:  # stopped or failed before the question was out: nobody heard it, nobody can say yes
            if pending is not None:
                confirmations.take(pending.id)
            raise
        if pending is not None:
            # The generator arms the request when it reaches this marker, i.e. once it handed out the whole
            # question. Arming here would be too early: the frames are only queued, and a barge-in that cuts
            # the question off must leave a request nobody heard to the end unanswerable.
            if first is None:  # /tts sent no audio: the question was never heard, so nobody can say yes to it
                log.warning("assistant dropped the request to run %s: its question produced no audio", pending.name)
                confirmations.take(pending.id)
            else:
                out.put(_Asked(pending.id, spoken=True, auto_seconds=pending.auto_seconds))
        if failure is not None:  # after the apology was queued: the generator raises it, so the UI shows why
            raise failure
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
        progress.cancel()
        out.put(_DONE)


def _default_brain(url: str, timeout: float) -> ChatBrain:
    """The helper endpoint when VAICE_BRAIN_URL is set, else the local llama-server at `url`."""
    if config.BRAIN_URL:
        return ChatBrain.helper(config.BRAIN_URL, config.BRAIN_TIMEOUT, config.BRAIN_MODEL, config.BRAIN_API_KEY)
    return ChatBrain.local(url, timeout)


class AssistantResponder:
    """Answers in separate steps (see the module docstring) and speaks through llama-server's /tts.

    Nothing here is bound to the thread that resumes the generator; the reply's thread is a daemon of its own.
    """

    name = "assistant"
    tool_history = True  # earlier skill results come in `history` as ("tool", ...); see Responder

    def __init__(
        self,
        url: str = config.LLAMACPP_URL,
        timeout: float = config.LLAMACPP_TIMEOUT,
        prompt: str = config.ASSISTANT_PROMPT,
        voice: str = config.VOICE,
        skills: Skills | None = None,
        brain: ChatBrain | None = None,
        confirmations: Confirmations | None = None,
        progress_seconds: float = config.ASSISTANT_PROGRESS_SECONDS,
        memory: Memory | None = None,
        vision: Skill | None = None,
        extra_skills: Sequence[Skill] = (),
        max_steps: int = config.ASSISTANT_MAX_STEPS,
        task_seconds: float = config.ASSISTANT_TASK_SECONDS,
    ):
        self._url = url
        self._timeout = timeout
        self._prompt = prompt
        self._voice = voice
        self.memory = memory  # the user's notes (None: memory is off); the UI's Memory panel reads it
        self.skills = skills or default_skills(shell=config.ASSISTANT_SHELL, memory=memory, vision=vision,
                                               extra=extra_skills)
        self._brain = brain or _default_brain(url, timeout)
        self._confirmations = Confirmations() if confirmations is None else confirmations
        self._progress_seconds = progress_seconds
        self._max_steps = max_steps
        self._task_seconds = task_seconds
        self._runs: set[threading.Event] = set()  # per click-approved run: set by `clear_pending` to stop it
        self._runs_lock = threading.Lock()
        self._outbox: Outbox | None = None  # where a continuation speaks (see `attach_outbox`)
        self._task_cancel: threading.Event | None = None  # the running continuation's; one task at a time
        self._task_lock = threading.Lock()

    def attach_outbox(self, outbox: Outbox) -> None:
        """Where the outcome of a call approved by a click or an `auto` countdown is spoken: no reply is in flight
        then, so the audio goes to `outbox`, which the audio handler drains (see Responder and `outbox.py`)."""
        self._outbox = outbox

    # The UI's side of a confirmation (see Responder). These run on the caller's thread (a Gradio worker, or the
    # countdown's), not on a reply's: there is no reply in flight to carry audio, so what follows a click is a
    # continuation of its own (`_continue`) that speaks through the outbox; the outcome of the call itself also lands
    # on the asking turn's tool events, which the Tool activity panel shows.

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
        # The task goes on (#110), as after a spoken yes (`_approved`), but outside any reply: on a thread of its own,
        # speaking through the outbox. Not for a run that was stopped ("New conversation", the app ending).
        if not (cancel.is_set() or lifecycle.stopping.is_set()):
            self._start_continuation(pending, result)
        return result.speech

    def _start_continuation(self, pending: Pending, result: SkillResult) -> None:
        """One task at a time: a continuation already running is cancelled (its queued audio stays; what it has not
        queued yet is never queued) and this one starts."""
        cancelled = threading.Event()
        with self._task_lock:
            if self._task_cancel is not None:
                self._task_cancel.set()
            self._task_cancel = cancelled
        threading.Thread(
            target=_continue,
            args=(self._url, self._timeout, self._voice, self.skills, self._brain, self._confirmations, pending, result,
                  cancelled, self._outbox, partial(Task.start, self._max_steps, self._task_seconds)),
            name="vaice-assistant-task",
            daemon=True,
        ).start()

    def _cancel_task(self) -> None:
        """Stop the running continuation, if any, and drop what it queued: it is no longer wanted."""
        with self._task_lock:
            cancelled, self._task_cancel = self._task_cancel, None
        if cancelled is not None:
            cancelled.set()
        if self._outbox is not None:
            self._outbox.clear()  # after the cancel: whatever it was about to queue is refused by its own stop check

    def cancel(self, pending_id: str) -> None:
        pending = self._confirmations.take(pending_id)
        if pending is not None:
            pending.turn.add_tool_event(pending.name, pending.arguments, "cancelled")

    def clear_pending(self) -> None:
        """"New conversation": a request from the old one must not be approvable, a command a click started for
        it stops, and so does the continuation that followed one (its speech is dropped)."""
        self._confirmations.clear()
        with self._runs_lock:
            for cancel in self._runs:
                cancel.set()
        self._cancel_task()

    def _countdown(self, pending_id: str) -> None:
        """Daemon thread: run the armed `auto` request `pending_id` when its time is up. It ends quietly as soon as
        the request is no longer the current one (a click, a spoken answer, "New conversation", a newer request,
        expiry) and when the app stops. The run goes through `approve`, which `take`s the request first: whatever
        settles it at the same moment, it runs once or not at all."""
        while not lifecycle.stopping.wait(POLL_SECONDS):
            pending = self._confirmations.current()
            if pending is None or pending.id != pending_id:
                return
            left = self._confirmations.remaining(pending)
            if left is None:
                return
            if left <= 0:
                log.info("assistant runs %s by itself: nobody stopped it in %g seconds", pending.name,
                         pending.auto_seconds)
                self.approve(pending_id)
                return

    def respond(self, audio: AudioFrame, history: Sequence[Message], turn: Turn) -> Iterator[AudioFrame]:
        self._cancel_task()  # a new utterance ends the task a click started, and what it had queued to say
        chunks: queue.Queue = queue.Queue()
        cancelled = threading.Event()
        notes = self.memory.notes() if self.memory is not None else []  # as they are when this reply starts
        threading.Thread(
            target=_work,
            args=(self._url, self._timeout, self._prompt, self._voice, self.skills, self._brain,
                  self._confirmations, audio, list(history), turn, chunks, cancelled, self._progress_seconds,
                  notes, partial(Task.start, self._max_steps, self._task_seconds)),
            name="vaice-assistant",
            daemon=True,
        ).start()
        spoken = ""
        play_end = 0.0  # monotonic time the audio yielded so far is estimated to finish playing (a filler's included)
        asked: str | None = None  # a confirmation question in flight: its request is not answerable yet
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
                if isinstance(item, _Asked):
                    if item.spoken:  # every frame of the question was handed out, not necessarily played
                        playing = _still_playing(play_end, time.monotonic()) if item.auto_seconds > 0 else 0.0
                        self._confirmations.arm(item.pending_id, playing)
                        if item.auto_seconds > 0:
                            threading.Thread(target=self._countdown, args=(item.pending_id,),
                                             name="vaice-assistant-auto", daemon=True).start()
                    asked = None if item.spoken else item.pending_id
                    continue
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
                play_end = _playback_end(play_end, time.monotonic(), len(frame) / rate)
                yield rate, frame
        finally:
            if asked is not None:  # the question was cut off (barge-in, stop, error): nobody can say yes to it
                self._confirmations.take(asked)
            cancelled.set()  # the reply's thread stops, and closes a streaming /tts response
