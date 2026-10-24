"""Gradio/WebRTC voice UI: pause detection → responder (audio in, audio out)."""
import argparse
import logging
import time

import numpy as np
from fastrtc import ReplyOnPause, Stream

from vaice import config
from vaice.responders import BACKENDS, Responder, create_responder

log = logging.getLogger(__name__)


def make_handler(responder: Responder):
    def reply(audio: tuple[int, np.ndarray]):
        """Called by ReplyOnPause when the speaker stops; streams audio back as it is produced."""
        start = time.perf_counter()
        first = True
        for chunk in responder.respond(audio):
            if first:
                log.info("time to first audio: %.2fs", time.perf_counter() - start)
                first = False
            yield chunk
        log.info("reply finished in %.2fs", time.perf_counter() - start)

    return reply


def main() -> None:
    parser = argparse.ArgumentParser(prog="vaice", description="Talk to a local model, hear it answer.")
    parser.add_argument("--backend", choices=BACKENDS, default=config.BACKEND, help="default: $VAICE_BACKEND or auto")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    responder = create_responder(args.backend)
    log.info("using %s backend", type(responder).__name__)

    stream = Stream(handler=ReplyOnPause(make_handler(responder)), modality="audio", mode="send-receive")
    stream.ui.launch(server_port=config.PORT)
