import mlx.core as mx
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
    def __init__(self, n_frames, with_eos=True):
        self.n_frames, self.with_eos = n_frames, with_eos

    def generate_from_chat_state(self, chat, **kwargs):
        yield mx.array([5]), LFMModality.TEXT
        for _ in range(self.n_frames):
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
