"""tests for the native backend"""
import gc
import logging
import threading
import time

import numpy as np
import pytest

pytest.importorskip("mlx.core")  # native backend is Apple-Silicon only
pytest.importorskip("mlx_audio")

import mlx.core as mx  # noqa: E402
from mlx_audio.sts.models.lfm_audio.model import (  # noqa: E402
    AUDIO_EOS_TOKEN,
    AUDIO_START_TOKEN,
    IM_END_TOKEN,
    TEXT_END_TOKEN,
    LFMModality,
)

from vaice.conversation import Turn  # noqa: E402
from vaice.responders import native  # noqa: E402
from vaice.responders.native import (  # noqa: E402
    FRAMES_PER_CHUNK,
    LEFT_CONTEXT_FRAMES,
    MAX_NEW_TOKENS,
    OUTPUT_SAMPLE_RATE,
    NativeResponder,
)

SAMPLES_PER_FRAME = 10  # the fake detokenizer's output length per frame


class FakeTokenizer:
    bos_token_id = 1

    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text]

    def decode(self, ids):
        return "".join(chr(i) for i in ids)  # spells control tokens out, like the real one


class FakeProcessor:
    tokenizer = FakeTokenizer()

    def __init__(self):
        self.decoded_shapes = []

    def preprocess_audio(self, audio, sample_rate):
        return mx.zeros((8, 4))

    def decode_text(self, tokens):
        return self.tokenizer.decode(tokens)

    def decode_audio(self, codes):
        self.decoded_shapes.append(tuple(codes.shape))
        return mx.zeros((1, 1, codes.shape[-1] * SAMPLES_PER_FRAME))


def text_tokens(text):
    return [mx.array([ord(c)]) for c in text]


class FakeModel:
    """Interleaved mode: one text token before each audio frame (then the rest), like the real one.
    Sequential mode (ASR): the transcript as text tokens, then <|im_end|>."""

    def __init__(self, n_frames, with_eos=True, delay=0.0, fail_after=None, reply_text="ab", asr_text="hello",
                 asr_error=None, asr_delay=0.0):
        self.n_frames, self.with_eos, self.delay, self.fail_after = n_frames, with_eos, delay, fail_after
        self.reply_text, self.asr_text, self.asr_error, self.asr_delay = reply_text, asr_text, asr_error, asr_delay
        self.step_threads = []
        self.steps = 0
        self.prefilled = False
        self.calls = []  # (mode, chat, kwargs, thread name) per generate_from_chat_state

    def generate_from_chat_state(self, chat, mode="interleaved", **kwargs):
        self.calls.append((mode, chat, kwargs, threading.current_thread().name))
        if mode == "sequential":
            yield from self._transcribe()
            return
        self.prefilled = True  # the real model encodes the audio and prefills before its first token
        text = text_tokens(self.reply_text)
        for i in range(self.n_frames):
            if i < len(text):
                yield text[i], LFMModality.TEXT
            self.step_threads.append(threading.current_thread().name)
            self.steps += 1
            if self.fail_after is not None and i == self.fail_after:
                raise RuntimeError("model blew up")
            time.sleep(self.delay)
            yield mx.array([7] * 8), LFMModality.AUDIO_OUT
        for token in text[self.n_frames :]:
            yield token, LFMModality.TEXT
        yield mx.array([TEXT_END_TOKEN]), LFMModality.TEXT
        if self.with_eos:
            yield mx.array([AUDIO_EOS_TOKEN] * 8), LFMModality.AUDIO_OUT

    def _transcribe(self):
        time.sleep(self.asr_delay)
        if self.asr_error:
            raise self.asr_error
        yield mx.array([AUDIO_START_TOKEN]), LFMModality.TEXT
        yield from ((token, LFMModality.TEXT) for token in text_tokens(self.asr_text))
        yield mx.array([IM_END_TOKEN]), LFMModality.TEXT


def chat_text(chat):
    """The ChatState's text tokens as a string (the fake tokenizer is char -> ord)."""
    return "".join(chr(i) for i in chat.text_tokens[1:])  # [0] is BOS


def reply(responder, history=(), turn=None):
    return responder.respond(AUDIO, history, turn or Turn())


AUDIO = (48000, np.zeros((1, 480), dtype=np.int16))


def test_chunks_are_decoded_with_left_context_and_cover_every_frame_exactly_once():
    processor = FakeProcessor()
    chunks = list(NativeResponder(FakeModel(25), processor).respond(AUDIO, [], Turn()))

    # 25 real frames -> chunks of 10, 10, 5. Later chunks are decoded with up to
    # LEFT_CONTEXT_FRAMES of earlier frames, and the EOS frame never reaches the decoder.
    assert [s[2] for s in processor.decoded_shapes] == [
        FRAMES_PER_CHUNK,
        LEFT_CONTEXT_FRAMES + FRAMES_PER_CHUNK,
        LEFT_CONTEXT_FRAMES + 5,
    ]
    assert all(s[:2] == (1, 8) for s in processor.decoded_shapes)
    # ...but the context is trimmed from the output: no duplicated or missing audio
    assert [len(wav) for _, wav in chunks] == [10 * SAMPLES_PER_FRAME, 10 * SAMPLES_PER_FRAME, 5 * SAMPLES_PER_FRAME]
    assert all(rate == OUTPUT_SAMPLE_RATE and wav.dtype == np.float32 and wav.ndim == 1 for rate, wav in chunks)


def test_text_only_output_yields_no_audio():
    assert list(NativeResponder(FakeModel(0), FakeProcessor()).respond(AUDIO, [], Turn())) == []


def next_on_new_thread(gen, name):
    """Resume `gen` from a fresh thread, like fastrtc does for each emitted chunk."""
    box = {}

    def run():
        try:
            box["value"] = next(gen)
        except BaseException as exc:
            box["error"] = exc

    t = threading.Thread(target=run, name=name)
    t.start()
    t.join()
    if "error" in box:
        raise box["error"]
    return box["value"]


def test_mlx_work_stays_on_one_worker_thread_however_the_generator_is_resumed():
    # Regression: MLX streams are thread-bound; resuming the generator from different
    # threads used to raise "There is no Stream(gpu, N) in current thread".
    model = FakeModel(25)
    gen = NativeResponder(model, FakeProcessor()).respond(AUDIO, [], Turn())

    chunks = [next_on_new_thread(gen, f"consumer-{i}") for i in range(3)]

    assert len(chunks) == 3
    assert len(set(model.step_threads)) == 1
    assert model.step_threads[0].startswith("vaice-mlx")


def test_abandoning_a_reply_stops_generation_and_frees_the_worker():
    model = FakeModel(2000, delay=0.002)  # ~4s if it ran to the end
    responder = NativeResponder(model, FakeProcessor())
    gen = responder.respond(AUDIO, [], Turn())
    next(gen)
    gen.close()  # user interrupted

    responder._executor.submit(lambda: None).result(timeout=10)  # worker is free for the next request
    assert model.steps < 2000


def test_closing_a_reply_from_another_thread_while_it_is_generating_fails_but_dropping_it_later_stops_the_work():
    # What fastrtc does on barge-in (#2): receive() calls close() on its thread while emit() is inside
    # next() on an executor thread. close() raises; the in-flight next() still returns its chunk; the
    # reply stops only when the generator is finalized afterwards.
    model = FakeModel(2000, delay=0.02)  # a 10-frame chunk takes ~0.2 s; the full reply would take ~40 s
    responder = NativeResponder(model, FakeProcessor())
    gen = responder.respond(AUDIO, [], Turn())
    next(gen)

    in_flight = {}
    consumer = threading.Thread(target=lambda: in_flight.setdefault("chunk", next(gen)), name="fastrtc-emit")
    consumer.start()
    deadline = time.monotonic() + 10
    while model.steps < FRAMES_PER_CHUNK + 5 and time.monotonic() < deadline:  # halfway through the second chunk
        time.sleep(0.001)
    assert consumer.is_alive()

    closer = {}

    def close_from_receive_thread():
        try:
            gen.close()
        except ValueError as exc:
            closer["error"] = exc

    receive = threading.Thread(target=close_from_receive_thread, name="fastrtc-receive")
    receive.start()
    receive.join(timeout=10)
    assert "generator already executing" in str(closer["error"])

    consumer.join(timeout=10)
    assert len(in_flight["chunk"][1]) == FRAMES_PER_CHUNK * SAMPLES_PER_FRAME  # it came back normally

    gen.close()  # CPython finalizing the dropped generator
    responder._executor.submit(lambda: None).result(timeout=10)  # worker is free for the next request
    assert model.steps < 2000


def test_a_cancelled_reply_does_no_model_work():
    model = FakeModel(100)
    cancelled = threading.Event()
    cancelled.set()
    assert list(NativeResponder(model, FakeProcessor())._generate(AUDIO, [], Turn(), cancelled, [("", 0)])) == []
    assert not model.prefilled


def test_model_errors_reach_the_caller():
    responder = NativeResponder(FakeModel(25, fail_after=3), FakeProcessor())
    with pytest.raises(RuntimeError, match="model blew up"):
        list(responder.respond(AUDIO, [], Turn()))


def test_hitting_the_token_limit_is_logged(caplog):
    with caplog.at_level(logging.WARNING, logger=native.__name__):
        list(NativeResponder(FakeModel(MAX_NEW_TOKENS), FakeProcessor()).respond(AUDIO, [], Turn()))
    assert "cut off" in caplog.text


def test_normal_replies_do_not_log_a_truncation_warning(caplog):
    with caplog.at_level(logging.WARNING, logger=native.__name__):
        list(NativeResponder(FakeModel(25), FakeProcessor()).respond(AUDIO, [], Turn()))
    assert caplog.text == ""


def test_failed_load_leaves_no_worker_thread_behind(monkeypatch):
    def boom(name):
        raise RuntimeError("weights missing")

    monkeypatch.setattr(native.LFM2AudioModel, "from_pretrained", boom)

    # Earlier tests drop their executors; their idle workers exit on their own, but not
    # synchronously. Wait (and break any reference cycle holding an executor), so the
    # assertion below only judges this test's own load failure.
    gc.collect()
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline and [t for t in threading.enumerate() if t.name.startswith("vaice-mlx")]:
        time.sleep(0.01)

    with pytest.raises(RuntimeError, match="weights missing"):
        NativeResponder.load("nonexistent/model")

    assert not [t for t in threading.enumerate() if t.name.startswith("vaice-mlx")]


def test_history_is_added_as_text_turns_before_the_current_audio_question():
    model = FakeModel(3)
    history = [("user", "capital of France?"), ("assistant", "Paris."), ("user", "(inaudible)")]
    list(NativeResponder(model, FakeProcessor(), prompt="BE BRIEF").respond(AUDIO, history, Turn()))

    mode, chat, kwargs, _ = model.calls[0]
    assert mode == "interleaved"
    assert chat_text(chat) == (
        "<|im_start|>system\nBE BRIEF<|im_end|>\n"
        "<|im_start|>user\ncapital of France?<|im_end|>\n"
        "<|im_start|>assistant\nParis.<|im_end|>\n"
        "<|im_start|>user\n(inaudible)<|im_end|>\n"
        "<|im_start|>user\n<|im_end|>\n"  # the audio goes here, as modality flags rather than text
        "<|im_start|>assistant\n"
    )
    audio_at = [i for i, m in enumerate(chat.modalities) if m == LFMModality.AUDIO_IN]
    assert len(audio_at) == 1 and all(chat.modalities[i] == LFMModality.TEXT for i in range(audio_at[0]))
    assert audio_at[0] > len("<|im_start|>system\nBE BRIEF<|im_end|>\n<|im_start|>user\ncapital of France?<|im_end|>\n")


def test_closing_the_turn_stops_generation_without_waiting_for_the_generator_to_be_dropped():
    model = FakeModel(2000, delay=0.002)  # ~4s if it ran to the end
    turn = Turn()
    gen = reply(NativeResponder(model, FakeProcessor()), turn=turn)
    next(gen)

    turn.close()  # the next turn started
    list(gen)  # the worker stops, so the stream ends instead of running to 2000 frames

    assert model.steps < 2000


def test_a_turn_closed_while_queued_does_no_model_work():
    model = FakeModel(100)
    turn = Turn()
    turn.close()
    assert list(NativeResponder(model, FakeProcessor())._generate(AUDIO, [], turn, threading.Event(), [("", 0)])) == []
    assert not model.prefilled


def test_assistant_text_is_the_share_of_the_reply_covered_by_the_audio_handed_out():
    text = "Hello there friend, how are you doing today?"  # 8 words
    turn = Turn()
    gen = reply(NativeResponder(FakeModel(25, reply_text=text), FakeProcessor()), turn=turn)

    next(gen)  # chunk 1 of 10, 10 and 5 frames
    assert turn.transcribed.wait(2)  # the worker transcribes after generating: the whole reply is known now
    next(gen)
    # 20 of 25 frames handed out -> 6 of 8 words, although all the text was already generated
    assert turn.assistant_text == "Hello there friend, how are you"
    list(gen)
    assert turn.assistant_text == text  # <|text_end|> never shows up


def test_spoken_part_splits_the_text_by_the_share_of_audio():
    assert native._spoken_part("one two three four", 40, 10) == "one"
    assert native._spoken_part("one two three four", 40, 20) == "one two"
    assert native._spoken_part("one two three four", 40, 40) == "one two three four"
    assert native._spoken_part("one two", 0, 0) == ""  # no audio, nothing said


def test_no_audio_means_no_assistant_text():
    turn = Turn()
    assert list(reply(NativeResponder(FakeModel(0, reply_text="text only"), FakeProcessor()), turn=turn)) == []
    assert turn.assistant_text == ""


def test_the_question_is_transcribed_on_the_worker_after_generation():
    model = FakeModel(25, asr_text="what is two plus two")
    turn = Turn()
    chunks = list(reply(NativeResponder(model, FakeProcessor()), history=[("user", "earlier")], turn=turn))

    assert len(chunks) == 3
    assert turn.transcribed.wait(5)
    assert turn.user_text == "what is two plus two"  # no <|audio_start|> / <|im_end|>
    assert [call[0] for call in model.calls] == ["interleaved", "sequential"]  # ASR runs after the reply
    mode, chat, kwargs, thread = model.calls[1]
    assert thread == model.calls[0][3] and thread.startswith("vaice-mlx")
    assert chat_text(chat) == "<|im_start|>system\nPerform ASR.<|im_end|>\n<|im_start|>user\n<|im_end|>\n<|im_start|>assistant\n"
    assert kwargs == {"max_new_tokens": native.ASR_MAX_NEW_TOKENS, "temperature": None, "top_k": None}


def test_transcription_does_not_hold_up_the_reply():
    model = FakeModel(25, asr_delay=0.3)
    turn = Turn()

    assert len(list(reply(NativeResponder(model, FakeProcessor()), turn=turn))) == 3

    assert turn.user_text is None  # reply finished while ASR is still running
    assert turn.transcribed.wait(5) and turn.user_text == "hello"


def test_an_abandoned_reply_is_still_transcribed():
    model = FakeModel(2000, delay=0.002)
    turn = Turn()
    gen = reply(NativeResponder(model, FakeProcessor()), turn=turn)
    next(gen)
    gen.close()

    assert turn.transcribed.wait(5) and turn.user_text == "hello"
    assert model.steps < 2000


def test_a_failed_reply_is_still_transcribed():
    turn = Turn()
    with pytest.raises(RuntimeError, match="model blew up"):
        list(reply(NativeResponder(FakeModel(25, fail_after=3), FakeProcessor()), turn=turn))

    assert turn.transcribed.wait(5) and turn.user_text == "hello"


def test_a_failed_transcription_is_logged_and_leaves_an_empty_transcript(caplog):
    model = FakeModel(25, asr_error=RuntimeError("asr blew up"))
    turn = Turn()
    with caplog.at_level(logging.ERROR, logger=native.__name__):
        assert len(list(reply(NativeResponder(model, FakeProcessor()), turn=turn))) == 3  # the reply is unaffected
        assert turn.transcribed.wait(5)

    assert turn.user_text == ""
    assert "asr blew up" in caplog.text


def test_loading_warms_up_both_generation_and_transcription(monkeypatch):
    model = FakeModel(5)
    monkeypatch.setattr(native.LFM2AudioModel, "from_pretrained", lambda name: model)
    monkeypatch.setattr(native.LFM2AudioProcessor, "from_pretrained", lambda name: FakeProcessor())

    responder = NativeResponder.load("fake/model")
    try:
        assert [call[0] for call in model.calls] == ["interleaved", "sequential"]
    finally:
        responder._executor.shutdown(wait=True)


def test_a_warm_up_transcription_that_never_finishes_fails_fast(monkeypatch):
    model = FakeModel(5, asr_delay=0.5)
    monkeypatch.setattr(native, "WARMUP_ASR_WAIT", 0.01)
    monkeypatch.setattr(native.LFM2AudioModel, "from_pretrained", lambda name: model)
    monkeypatch.setattr(native.LFM2AudioProcessor, "from_pretrained", lambda name: FakeProcessor())

    with pytest.raises(RuntimeError, match="warm-up"):
        NativeResponder.load("fake/model")  # the stuck ASR worker was abandoned, not waited for

    # it finishes (or exits) on its own; the test doesn't leave a worker behind
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline and [t for t in threading.enumerate() if t.name.startswith("vaice-mlx")]:
        time.sleep(0.01)
    assert not [t for t in threading.enumerate() if t.name.startswith("vaice-mlx")]
