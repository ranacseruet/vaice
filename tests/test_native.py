import logging
import threading
import time

import numpy as np
import pytest

pytest.importorskip("mlx.core")  # native backend is Apple-Silicon only
pytest.importorskip("mlx_audio")

import mlx.core as mx  # noqa: E402
from mlx_audio.sts.models.lfm_audio.model import AUDIO_EOS_TOKEN, LFMModality  # noqa: E402

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


class FakeProcessor:
    tokenizer = FakeTokenizer()

    def __init__(self):
        self.decoded_shapes = []

    def preprocess_audio(self, audio, sample_rate):
        return mx.zeros((8, 4))

    def decode_audio(self, codes):
        self.decoded_shapes.append(tuple(codes.shape))
        return mx.zeros((1, 1, codes.shape[-1] * SAMPLES_PER_FRAME))


class FakeModel:
    def __init__(self, n_frames, with_eos=True, delay=0.0, fail_after=None):
        self.n_frames, self.with_eos, self.delay, self.fail_after = n_frames, with_eos, delay, fail_after
        self.step_threads = []
        self.steps = 0
        self.prefilled = False

    def generate_from_chat_state(self, chat, **kwargs):
        self.prefilled = True  # the real model encodes the audio and prefills before its first token
        yield mx.array([5]), LFMModality.TEXT
        for i in range(self.n_frames):
            self.step_threads.append(threading.current_thread().name)
            self.steps += 1
            if self.fail_after is not None and i == self.fail_after:
                raise RuntimeError("model blew up")
            time.sleep(self.delay)
            yield mx.array([7] * 8), LFMModality.AUDIO_OUT
        if self.with_eos:
            yield mx.array([AUDIO_EOS_TOKEN] * 8), LFMModality.AUDIO_OUT


AUDIO = (48000, np.zeros((1, 480), dtype=np.int16))


def test_chunks_are_decoded_with_left_context_and_cover_every_frame_exactly_once():
    processor = FakeProcessor()
    chunks = list(NativeResponder(FakeModel(25), processor).respond(AUDIO))

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
    assert list(NativeResponder(FakeModel(0), FakeProcessor()).respond(AUDIO)) == []


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
    gen = NativeResponder(model, FakeProcessor()).respond(AUDIO)

    chunks = [next_on_new_thread(gen, f"consumer-{i}") for i in range(3)]

    assert len(chunks) == 3
    assert len(set(model.step_threads)) == 1
    assert model.step_threads[0].startswith("vaice-mlx")


def test_abandoning_a_reply_stops_generation_and_frees_the_worker():
    model = FakeModel(2000, delay=0.002)  # ~4s if it ran to the end
    responder = NativeResponder(model, FakeProcessor())
    gen = responder.respond(AUDIO)
    next(gen)
    gen.close()  # user interrupted

    responder._executor.submit(lambda: None).result(timeout=10)  # worker is free for the next request
    assert model.steps < 2000


def test_a_cancelled_reply_does_no_model_work():
    model = FakeModel(100)
    cancelled = threading.Event()
    cancelled.set()
    assert list(NativeResponder(model, FakeProcessor())._generate(AUDIO, cancelled)) == []
    assert not model.prefilled


def test_model_errors_reach_the_caller():
    responder = NativeResponder(FakeModel(25, fail_after=3), FakeProcessor())
    with pytest.raises(RuntimeError, match="model blew up"):
        list(responder.respond(AUDIO))


def test_hitting_the_token_limit_is_logged(caplog):
    with caplog.at_level(logging.WARNING, logger=native.__name__):
        list(NativeResponder(FakeModel(MAX_NEW_TOKENS), FakeProcessor()).respond(AUDIO))
    assert "cut off" in caplog.text


def test_normal_replies_do_not_log_a_truncation_warning(caplog):
    with caplog.at_level(logging.WARNING, logger=native.__name__):
        list(NativeResponder(FakeModel(25), FakeProcessor()).respond(AUDIO))
    assert caplog.text == ""


def test_failed_load_leaves_no_worker_thread_behind(monkeypatch):
    def boom(name):
        raise RuntimeError("weights missing")

    monkeypatch.setattr(native.LFM2AudioModel, "from_pretrained", boom)
    with pytest.raises(RuntimeError, match="weights missing"):
        NativeResponder.load("nonexistent/model")

    assert not [t for t in threading.enumerate() if t.name.startswith("vaice-mlx")]
