import threading
import time

import mlx.core as mx
import pytest
import numpy as np
from mlx_audio.sts.models.lfm_audio.model import AUDIO_EOS_TOKEN, LFMModality

from vaice.responders.native import FRAMES_PER_CHUNK, OUTPUT_SAMPLE_RATE, NativeResponder


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
        return mx.zeros((1, 1, codes.shape[-1] * 10))


class FakeModel:
    def __init__(self, n_frames, with_eos=True, delay=0.0, fail_after=None):
        self.n_frames, self.with_eos, self.delay, self.fail_after = n_frames, with_eos, delay, fail_after
        self.step_threads = []
        self.steps = 0

    def generate_from_chat_state(self, chat, **kwargs):
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


def test_audio_frames_are_decoded_in_chunks_and_eos_is_dropped():
    processor = FakeProcessor()
    chunks = list(NativeResponder(FakeModel(25), processor).respond(AUDIO))

    # 25 real frames -> 10 + 10 + 5; the EOS frame never reaches the decoder
    assert [s[2] for s in processor.decoded_shapes] == [FRAMES_PER_CHUNK, FRAMES_PER_CHUNK, 5]
    assert all(s[:2] == (1, 8) for s in processor.decoded_shapes)
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


def test_abandoning_a_reply_frees_the_worker():
    model = FakeModel(1000, delay=0.005)  # would take ~5s if it ran to the end
    responder = NativeResponder(model, FakeProcessor())
    gen = responder.respond(AUDIO)
    next(gen)
    gen.close()  # user interrupted

    start = time.perf_counter()
    responder._executor.submit(lambda: None).result(timeout=5)  # next request can start
    assert time.perf_counter() - start < 1.5
    assert model.steps < 1000


def test_model_errors_reach_the_caller():
    responder = NativeResponder(FakeModel(25, fail_after=3), FakeProcessor())
    with pytest.raises(RuntimeError, match="model blew up"):
        list(responder.respond(AUDIO))
