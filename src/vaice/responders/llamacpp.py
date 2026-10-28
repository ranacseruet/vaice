"""llama.cpp backend: llama-server's own speech, streamed from its /v1/realtime WebSocket."""
import base64
import json
import logging
import queue
import threading
import time
from collections.abc import Callable, Iterator, Sequence

import requests
from websockets.exceptions import ConnectionClosed, WebSocketException
from websockets.sync.client import ClientConnection, connect

from vaice import config, lifecycle
from vaice.audio import from_pcm16, to_pcm16
from vaice.conversation import Message, Turn
from vaice.responders.base import AudioFrame, spoken_part

log = logging.getLogger(__name__)

# Tells an interleaved-mode model (LFM3-Audio) to write its text alongside the speech; a fixed constant, so
# a VAICE_LLAMACPP_PROMPT (or the older VAICE_LLM_PROMPT) only has to say how to answer
MODE_INSTRUCTION = "Respond with interleaved text and audio."
MAX_OUTPUT_TOKENS = 512  # counts text tokens and audio frames together, like the native backend's limit
# How often a waiting thread checks whether the reply was abandoned or the app is stopping
POLL_SECONDS = 0.25
# The close handshake of a socket whose server is busy or gone must not hold a thread for long
CLOSE_TIMEOUT = 1.0
CHECK_TIMEOUT = 2  # seconds for the startup HTTP checks

ABANDONED = "the app is stopping; the reply is abandoned"
_DONE = object()


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


def _converse(
    url: str,
    timeout: float,
    instructions: str,
    audio: AudioFrame,
    out: queue.Queue,
    cancelled: threading.Event,
    progress: list[tuple[str, int]],
) -> None:
    """Daemon thread: one realtime session for one reply. Puts (rate, samples) frames, then _DONE, on `out`
    (or the exception that ended it before _DONE) and keeps progress[0] = (reply text, samples received).

    A thread of our own, because fastrtc resumes the reply generator on whichever worker thread is
    free and the socket must not depend on any of them: connect() can block for the whole open
    timeout and close() for the close timeout, and a thread of fastrtc's executor that is stuck there
    keeps the process alive after Ctrl+C (the executor's threads are joined at exit). Takes only
    plain data, never the responder, so nothing here keeps it alive.
    """

    def stop() -> bool:
        return cancelled.is_set() or lifecycle.stopping.is_set()

    try:
        with _open(url, timeout) as ws:
            session = _Session(ws, timeout, stop)
            rate_in, rate_out = session.rates()
            session.update(
                {
                    "instructions": instructions,
                    "output_modalities": ["text", "audio"],
                    "max_output_tokens": MAX_OUTPUT_TOKENS,
                }
            )
            session.send({"type": "input_audio_buffer.append", "audio": _b64(to_pcm16(audio, rate_in))})
            session.send({"type": "response.create"})
            text: list[str] = []
            samples = 0
            while (event := session.receive()) is not None:
                kind = event.get("type")
                if kind == "response.output_text.delta":
                    text.append(event.get("delta", ""))
                    progress[0] = ("".join(text), samples)
                elif kind == "response.output_audio.delta":
                    frame = from_pcm16(base64.b64decode(event.get("delta", "")))
                    if len(frame):
                        samples += len(frame)
                        progress[0] = ("".join(text), samples)  # before the frame is handed over
                        out.put((rate_out, frame))
                elif kind == "response.done":
                    _check_done(event, samples)
                    break
    except _Stopped:
        pass
    except BaseException as exc:  # noqa: BLE001 - re-raised in the consumer, which must see any failure
        out.put(exc)
    finally:
        out.put(_DONE)


def _check_done(event: dict, samples: int) -> None:
    response = event.get("response") or {}
    status = response.get("status")
    if status != "completed":  # "cancelled": we never ask for it, so someone else stopped the reply
        raise RuntimeError(f"llama-server ended the reply with status {status!r}")
    # A reply that hits max_output_tokens still reports "completed"; the usage is what gives it away
    tokens = (response.get("usage") or {}).get("output_tokens")
    if isinstance(tokens, int) and tokens >= MAX_OUTPUT_TOKENS:
        log.warning("llamacpp reply hit the %d-token limit and was cut off", MAX_OUTPUT_TOKENS)
    if not samples:
        log.warning("llama-server's reply had no audio (is the model one with audio output?)")


def _b64(pcm: bytes) -> str:
    return base64.b64encode(pcm).decode("ascii")


def _transcribe(url: str, timeout: float, audio: AudioFrame) -> str:
    """What `audio` says, from a transcription session of its own.

    A conversation session never sends a transcript of the user's audio. No language is set, so the
    model detects it ("Perform ASR."), which reads English the same and does not pin the user to it.
    """
    with _open(url, timeout) as ws:
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


def _instructions(prompt: str, history: Sequence[Message]) -> str:
    """The system prompt, with the earlier turns as text.

    A realtime session cannot be handed earlier assistant turns (conversation.item.create rejects
    the assistant role), so each reply gets a fresh session and the history goes in as text. That
    also leaves `Conversation` the one owner of history: windowing, barge-in cuts and "New
    conversation" all apply.
    """
    text = f"{MODE_INSTRUCTION} {prompt}".strip()
    if history:
        lines = (f"{role.capitalize()}: {' '.join(said.split())}" for role, said in history)
        text += "\n\nThe conversation so far:\n" + "\n".join(lines)
    return text


class LlamaCppResponder:
    """Streams the reply of an audio-output model on llama-server (LFM3-Audio) over /v1/realtime.

    Each reply is a fresh realtime session on a daemon thread of its own, which hands the audio to the
    generator through a queue; a second session transcribes the same audio in parallel. Nothing
    here is bound to the thread that resumes the generator, so fastrtc may resume it from any.
    """

    name = "llamacpp"

    def __init__(
        self,
        url: str = config.LLAMACPP_URL,
        prompt: str = config.LLAMACPP_PROMPT,
        timeout: float = config.LLAMACPP_TIMEOUT,
    ):
        self._url = url
        self._prompt = prompt
        self._timeout = timeout

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
        progress = [("", 0)]  # (reply text, audio samples) received so far; replaced whole by the session thread
        threading.Thread(
            target=_converse,
            args=(self._url, self._timeout, _instructions(self._prompt, history), audio, chunks, cancelled, progress),
            name="vaice-llamacpp",
            daemon=True,
        ).start()
        handed_out = 0
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
                if item is _DONE:
                    break
                if isinstance(item, BaseException):
                    raise item
                if turn.closed.is_set():
                    return
                handed_out += len(item[1])
                turn.set_assistant_text(spoken_part(*progress[0], handed_out))
                yield item
            if lifecycle.stopping.is_set():  # the session thread may have noticed first, and stopped for the app
                log.warning(ABANDONED)
                return
            turn.set_assistant_text(spoken_part(*progress[0], handed_out))  # all of it, now that it's all out
        finally:
            cancelled.set()  # the session thread closes the socket, which cancels the reply on the server


def check_server(url: str = config.LLAMACPP_URL, strict: bool = False) -> None:
    """Check at startup that llama-server is up and serves an audio-output model.

    `strict` (backend `auto`) raises on anything short of that, so the factory falls back to the
    native backend. Otherwise (`llamacpp` asked for explicitly) a server that is down or still
    loading is only a warning or note, because it may be up by the first question; a server that
    is up with a model that cannot speak raises, because no wait would fix that.
    """

    def unavailable(message: str) -> None:
        if strict:
            raise RuntimeError(message)
        log.warning("%s; replies will fail until it is up", message)

    health = f"{url}/health"
    try:
        r = requests.get(health, timeout=CHECK_TIMEOUT)
    except requests.RequestException as exc:
        unavailable(f"llama-server not reachable at {health} ({exc})")
        return
    if r.status_code == 503:  # llama-server answers 503 "Loading model" until its weights are in
        if strict:
            raise RuntimeError(f"llama-server at {health} is still loading its model; cannot tell if it speaks")
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
