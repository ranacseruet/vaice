"""Gradio/WebRTC voice UI: pause detection -> Gemma (audio in) -> Kokoro (audio out)."""
import numpy as np
from fastrtc import ReplyOnPause, Stream

from vaice import config, tts
from vaice.audio import to_wav_b64
from vaice.llm import stream_reply
from vaice.sentences import split_sentences


def reply(audio: tuple[int, np.ndarray]):
    """Called by ReplyOnPause when the speaker stops; yields audio as each sentence completes."""
    for sentence in split_sentences(stream_reply(to_wav_b64(audio))):
        yield from tts.speak(sentence)


def main() -> None:
    tts.load_model()  # load once at startup so the first reply isn't slow
    stream = Stream(handler=ReplyOnPause(reply), modality="audio", mode="send-receive")
    stream.ui.launch(server_port=config.PORT)
