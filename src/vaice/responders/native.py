"""Native backend: one speech-to-speech model (mlx-audio) turns audio into audio directly."""
from collections.abc import Iterator

import mlx.core as mx
import numpy as np
from mlx_audio.sts.models.lfm_audio.model import AUDIO_EOS_TOKEN, LFM2AudioModel, LFMModality
from mlx_audio.sts.models.lfm_audio.processor import ChatState, LFM2AudioProcessor

from vaice import config
from vaice.audio import to_mono_float
from vaice.responders.base import AudioFrame

OUTPUT_SAMPLE_RATE = 24000
# Decode this many audio frames at a time so playback starts before generation ends
FRAMES_PER_CHUNK = 10
MAX_NEW_TOKENS = 512


class NativeResponder:
    def __init__(self, model, processor, prompt: str = config.NATIVE_PROMPT):
        self._model = model
        self._processor = processor
        self._prompt = prompt

    @classmethod
    def load(cls, model_name: str = config.NATIVE_MODEL) -> "NativeResponder":
        responder = cls(LFM2AudioModel.from_pretrained(model_name), LFM2AudioProcessor.from_pretrained(model_name))
        # The first generation costs ~2s extra (kernel compilation); pay it at startup, not on the first question
        for _ in responder.respond((16000, np.zeros((1, 8000), dtype=np.int16))):
            pass
        return responder

    def respond(self, audio: AudioFrame) -> Iterator[AudioFrame]:
        sample_rate, samples = audio
        # The processor resamples to the model's rate itself
        chat = ChatState(self._processor)
        chat.new_turn("system")
        chat.add_text(self._prompt)
        chat.end_turn()
        chat.new_turn("user")
        chat.add_audio(mx.array(to_mono_float(samples)), sample_rate=sample_rate)
        chat.end_turn()
        chat.new_turn("assistant")

        frames: list[mx.array] = []
        stream = self._model.generate_from_chat_state(
            chat, mode="interleaved", max_new_tokens=MAX_NEW_TOKENS, audio_temperature=1.0, audio_top_k=4
        )
        for token, modality in stream:
            if modality != LFMModality.AUDIO_OUT or int(token[0].item()) == AUDIO_EOS_TOKEN:
                continue
            frames.append(token)
            if len(frames) == FRAMES_PER_CHUNK:
                yield self._decode(frames)
                frames = []
        if frames:
            yield self._decode(frames)

    def _decode(self, frames: list[mx.array]) -> AudioFrame:
        codes = mx.stack(frames, axis=1)[None, :]  # detokenizer wants (batch, codebooks, time)
        wav = np.array(self._processor.decode_audio(codes), dtype=np.float32).reshape(-1)
        return (OUTPUT_SAMPLE_RATE, wav)
