"""Text chat with local sherpa-onnx Whisper/Kokoro speech adapters."""
import json
import logging
import math
import queue
import re
import threading
from collections.abc import Iterator, Sequence
from pathlib import Path

import numpy as np
import requests
from scipy.signal import resample_poly

from vaice import config, lifecycle
from vaice.audio import to_mono_float
from vaice.conversation import Message, Turn
from vaice.responders.base import AudioFrame, spoken_part

log = logging.getLogger(__name__)

SAMPLE_RATE = 16000
POLL_SECONDS = 0.25
FRAME_SAMPLES = 4800  # 200 ms at Kokoro's 24 kHz output rate
_DONE = object()


def complete_sentences(buffer: str, *, final: bool) -> tuple[list[str], str]:
    """Take sentences ending in punctuation from a streamed text buffer.

    A punctuation mark is a boundary when followed by a quote/bracket and whitespace. The
    remainder stays buffered until another chunk or the completed response confirms the sentence.
    """
    sentences: list[str] = []
    start = 0
    for match in re.finditer(r"[.!?]+[\"'\u201d\u2019)]*(?=\s)", buffer):
        sentence = buffer[start : match.end()].strip()
        if sentence:
            sentences.append(sentence)
        start = match.end()
    remainder = buffer[start:]
    if final and remainder.strip():
        sentences.append(remainder.strip())
        remainder = ""
    return sentences, remainder.lstrip()


def _stopped(cancelled: threading.Event, turn: Turn) -> bool:
    return cancelled.is_set() or turn.closed.is_set() or lifecycle.stopping.is_set()


def _required_files(paths: Sequence[Path], label: str) -> None:
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"{label} model files are missing: {', '.join(missing)}")


def _required_directory(path: Path, label: str) -> None:
    if not path.is_dir():
        raise FileNotFoundError(f"{label} directory is missing: {path}")


class _TextPipelineState:
    """Lazily loaded local voice models, serialized away from fastrtc's worker threads."""

    def __init__(
        self,
        recognizer=None,
        tts=None,
        *,
        whisper_encoder: Path = config.WHISPER_ENCODER,
        whisper_decoder: Path = config.WHISPER_DECODER,
        whisper_tokens: Path = config.WHISPER_TOKENS,
        whisper_language: str = config.WHISPER_LANGUAGE,
        kokoro_model: Path = config.KOKORO_MODEL,
        kokoro_voices: Path = config.KOKORO_VOICES,
        kokoro_tokens: Path = config.KOKORO_TOKENS,
        kokoro_lexicon: Path = config.KOKORO_LEXICON,
        kokoro_data_dir: Path = config.KOKORO_DATA_DIR,
        speaker: str = config.KOKORO_SPEAKER,
        speed: float = config.KOKORO_SPEED,
    ):
        self.recognizer = recognizer
        self.tts = tts
        self.whisper_encoder = whisper_encoder
        self.whisper_decoder = whisper_decoder
        self.whisper_tokens = whisper_tokens
        self.whisper_language = whisper_language
        self.kokoro_model = kokoro_model
        self.kokoro_voices = kokoro_voices
        self.kokoro_tokens = kokoro_tokens
        self.kokoro_lexicon = kokoro_lexicon
        self.kokoro_data_dir = kokoro_data_dir
        try:
            self.speaker = int(speaker)
        except ValueError:
            raise ValueError(
                f"VAICE_KOKORO_SPEAKER must be an integer speaker ID, got {speaker!r}"
            ) from None
        self.speed = speed
        self.model_lock = threading.Lock()

    def _get_recognizer(self):
        if self.recognizer is None:
            _required_files(
                (self.whisper_encoder, self.whisper_decoder, self.whisper_tokens),
                "sherpa-onnx Whisper",
            )
            try:
                import sherpa_onnx
            except ImportError as exc:
                raise RuntimeError(
                    "the text pipeline requires optional dependencies; install vaice with `.[local-voice]`"
                ) from exc
            self.recognizer = sherpa_onnx.OfflineRecognizer.from_whisper(
                encoder=str(self.whisper_encoder),
                decoder=str(self.whisper_decoder),
                tokens=str(self.whisper_tokens),
                language=self.whisper_language,
                task="transcribe",
                num_threads=2,
            )
        return self.recognizer

    def transcribe(self, audio: AudioFrame) -> str:
        sample_rate, samples = audio
        mono = to_mono_float(samples)
        if sample_rate != SAMPLE_RATE:
            divisor = math.gcd(sample_rate, SAMPLE_RATE)
            mono = resample_poly(mono, SAMPLE_RATE // divisor, sample_rate // divisor).astype(np.float32)
        with self.model_lock:
            recognizer = self._get_recognizer()
            stream = recognizer.create_stream()
            stream.accept_waveform(SAMPLE_RATE, mono)
            recognizer.decode_stream(stream)
            return str(stream.result.text).strip()

    def _get_tts(self):
        if self.tts is None:
            _required_files(
                (self.kokoro_model, self.kokoro_voices, self.kokoro_tokens, self.kokoro_lexicon),
                "sherpa-onnx Kokoro",
            )
            _required_directory(self.kokoro_data_dir, "sherpa-onnx Kokoro espeak-ng-data")
            try:
                import sherpa_onnx
            except ImportError as exc:
                raise RuntimeError(
                    "the text pipeline requires optional dependencies; install vaice with `.[local-voice]`"
                ) from exc
            kokoro = sherpa_onnx.OfflineTtsKokoroModelConfig(
                model=str(self.kokoro_model),
                voices=str(self.kokoro_voices),
                tokens=str(self.kokoro_tokens),
                lexicon=str(self.kokoro_lexicon),
                data_dir=str(self.kokoro_data_dir),
                lang="en-us",
            )
            model = sherpa_onnx.OfflineTtsModelConfig(kokoro=kokoro, num_threads=2, provider="cpu")
            self.tts = sherpa_onnx.OfflineTts(sherpa_onnx.OfflineTtsConfig(model=model))
        return self.tts

    def synthesize(self, sentence: str) -> tuple[int, np.ndarray]:
        with self.model_lock:
            tts = self._get_tts()
            generated = tts.generate(sentence, sid=self.speaker, speed=self.speed)
            samples = np.asarray(generated.samples, dtype=np.float32).reshape(-1)
            return int(generated.sample_rate), samples


class _RequestControl:
    """Lets the consumer cancel the current HTTP stream while the worker is blocked reading it."""

    def __init__(self):
        self.cancelled = threading.Event()
        self._lock = threading.Lock()
        self._response = None
        self._close_started = False

    @staticmethod
    def _close_response(response) -> None:
        try:
            response.close()
        except Exception:
            log.debug("closing the text stream failed", exc_info=True)

    def _start_close(self, response, *, daemon: bool) -> None:
        with self._lock:
            if self._close_started:
                return
            self._close_started = True
        if daemon:
            threading.Thread(
                target=self._close_response,
                args=(response,),
                name="vaice-text-http-close",
                daemon=True,
            ).start()
        else:
            self._close_response(response)

    def attach(self, response) -> None:
        with self._lock:
            self._response = response
            cancelled = self.cancelled.is_set()
        if cancelled:
            self._start_close(response, daemon=False)

    def release(self, response) -> None:
        with self._lock:
            if self._response is response:
                self._response = None

    def cancel(self) -> None:
        self.cancelled.set()
        with self._lock:
            response = self._response
        if response is not None:
            self._start_close(response, daemon=True)


def _model_id(url: str, timeout: float) -> str:
    endpoint = f"{url}/models"
    try:
        with requests.get(endpoint, timeout=min(timeout, 2.0)) as response:
            response.raise_for_status()
            models = response.json().get("data", [])
            model_id = models[0].get("id") if models else None
    except (requests.RequestException, ValueError, AttributeError, TypeError, IndexError) as exc:
        raise RuntimeError(f"could not discover a text model at {endpoint}: {type(exc).__name__}: {exc}") from exc
    if not isinstance(model_id, str) or not model_id:
        raise RuntimeError(f"no model ID was returned by {endpoint}; set VAICE_TEXT_MODEL")
    return model_id


def _chat_deltas(response, cancelled: threading.Event) -> Iterator[str]:
    # requests defaults to 512 bytes, which can hold a complete short SSE event until more output
    # or the response closes. Read by line as soon as bytes arrive so TTS starts mid-stream.
    for raw in response.iter_lines(chunk_size=1, decode_unicode=True):
        if cancelled.is_set() or lifecycle.stopping.is_set():
            return
        if raw is None:
            continue
        line = raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else str(raw)
        line = line.strip()
        if not line or line.startswith(":") or not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            return
        try:
            event = json.loads(data)
        except ValueError as exc:
            raise RuntimeError(f"unexpected text-chat event: {data[:200]!r}") from exc
        if isinstance(event.get("error"), dict):
            message = event["error"].get("message", "unknown error")
            raise RuntimeError(f"text server error: {message}")
        choices = event.get("choices")
        if not isinstance(choices, list) or not choices:
            continue
        delta = choices[0].get("delta", {})
        content = delta.get("content") if isinstance(delta, dict) else None
        if isinstance(content, str) and content:
            yield content


def _try_publish(out: queue.Queue, item) -> bool:
    try:
        out.put_nowait(item)
        return True
    except queue.Full:
        return False


def _publish_until_cancelled(out: queue.Queue, item, control: _RequestControl, turn: Turn) -> bool:
    while not _stopped(control.cancelled, turn):
        if _try_publish(out, item):
            return True
        control.cancelled.wait(POLL_SECONDS)
    return False


def _text_for_handed_audio(segments: Sequence[tuple[str, int]], handed_out: int) -> str:
    """Return cumulative text covered by handed-out audio, allocating samples sentence by sentence."""
    spoken: list[str] = []
    remaining = handed_out
    for sentence, samples in segments:
        used = min(remaining, samples)
        part = spoken_part(sentence, samples, used)
        if part:
            spoken.append(part)
        remaining -= used
        if used < samples:
            break
    return " ".join(spoken)


def _run_turn(
    state: _TextPipelineState,
    url: str,
    timeout: float,
    model_id: str,
    prompt: str,
    audio: AudioFrame,
    history: Sequence[Message],
    turn: Turn,
    out: queue.Queue,
    control: _RequestControl,
) -> None:
    transcript_reported = False
    try:
        if _stopped(control.cancelled, turn):
            transcript_reported = True
            turn.set_user_text("")
            return
        transcript = state.transcribe(audio)
        transcript_reported = True
        turn.set_user_text(transcript)
        if _stopped(control.cancelled, turn):
            return
        if not transcript:
            raise RuntimeError("Whisper did not recognize speech; the question is marked inaudible")

        messages = [{"role": "system", "content": prompt}]
        messages.extend({"role": role, "content": text} for role, text in history)
        messages.append({"role": "user", "content": transcript})
        model = model_id or _model_id(url, timeout)
        endpoint = f"{url}/chat/completions"
        try:
            response = requests.post(
                endpoint,
                json={"model": model, "messages": messages, "stream": True},
                stream=True,
                timeout=(min(timeout, 5.0), timeout),
            )
        except requests.RequestException as exc:
            raise RuntimeError(f"cannot request {endpoint}: {type(exc).__name__}: {exc}") from exc
        control.attach(response)
        pending = ""
        segments: list[tuple[str, int]] = []
        generated_any = False

        def speak(sentence: str) -> bool:
            nonlocal generated_any
            if _stopped(control.cancelled, turn):
                return False
            sample_rate, samples = state.synthesize(sentence)
            if _stopped(control.cancelled, turn):
                return False
            if samples.size == 0:
                return True
            segments.append((sentence, len(samples)))
            generated_any = True
            for offset in range(0, len(samples), FRAME_SAMPLES):
                frame = (sample_rate, samples[offset : offset + FRAME_SAMPLES])
                if not _publish_until_cancelled(out, (frame, tuple(segments)), control, turn):
                    return False
            return True

        try:
            response.raise_for_status()
            for delta in _chat_deltas(response, control.cancelled):
                pending += delta
                sentences, pending = complete_sentences(pending, final=False)
                for sentence in sentences:
                    if not speak(sentence):
                        return
            sentences, pending = complete_sentences(pending, final=True)
            for sentence in sentences:
                if not speak(sentence):
                    return
        finally:
            control._close_response(response)
            control.release(response)
        if not generated_any and not _stopped(control.cancelled, turn):
            raise RuntimeError("text server produced no speech")
    except Exception as exc:  # noqa: BLE001 - the worker reports all failures through the consumer queue
        if not _stopped(control.cancelled, turn):
            _publish_until_cancelled(out, exc, control, turn)
    finally:
        if not transcript_reported:
            # Even cancellation/failure completes the transcript contract exactly once.
            turn.set_user_text("")
        while not control.cancelled.is_set():
            if _try_publish(out, _DONE):
                break
            control.cancelled.wait(POLL_SECONDS)


class TextResponder:
    """Local Whisper → OpenAI-compatible text chat → local Kokoro, streamed by sentence.

    The long-lived ONNX objects stay in a single daemon worker thread for each turn rather than on
    fastrtc's thread-per-resume generator. The generator only reads queued audio and updates `Turn`.
    """

    name = "text"

    def __init__(
        self,
        url: str = config.TEXT_URL,
        prompt: str = config.TEXT_PROMPT,
        timeout: float = config.LLAMACPP_TIMEOUT,
        model_id: str = config.TEXT_MODEL,
        *,
        recognizer=None,
        tts=None,
    ):
        self._url = url.rstrip("/")
        self._prompt = prompt
        self._timeout = timeout
        self._model_id = model_id
        self._state = _TextPipelineState(recognizer=recognizer, tts=tts)

    def respond(self, audio: AudioFrame, history: Sequence[Message], turn: Turn) -> Iterator[AudioFrame]:
        out: queue.Queue = queue.Queue(maxsize=16)
        control = _RequestControl()
        worker = threading.Thread(
            target=_run_turn,
            args=(
                self._state,
                self._url,
                self._timeout,
                self._model_id,
                self._prompt,
                audio,
                history,
                turn,
                out,
                control,
            ),
            name="vaice-text-pipeline",
            daemon=True,
        )
        worker.start()
        handed_out = 0
        try:
            while True:
                try:
                    item = out.get(timeout=POLL_SECONDS)
                except queue.Empty:
                    if lifecycle.stopping.is_set():
                        log.warning("the app is stopping; the text reply is abandoned")
                        return
                    if turn.closed.is_set():
                        return
                    continue
                if item is _DONE:
                    break
                if isinstance(item, BaseException):
                    raise item
                if turn.closed.is_set():
                    return
                chunk, segments = item
                handed_out += len(chunk[1])
                turn.set_assistant_text(_text_for_handed_audio(segments, handed_out))
                yield chunk
            if lifecycle.stopping.is_set():
                log.warning("the app is stopping; the text reply is abandoned")
                return
            if "segments" in locals():
                turn.set_assistant_text(_text_for_handed_audio(segments, handed_out))
        finally:
            control.cancel()
