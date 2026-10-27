"""Gradio/WebRTC voice UI: pause detection → responder (audio in, audio out)."""
import argparse
import logging
import time

import gradio as gr
import numpy as np
from fastrtc import AdditionalOutputs, Stream

from vaice import config
from vaice.conversation import Conversation
from vaice.interruptible import InterruptSafeReplyOnPause, stop_playing_reply
from vaice.responders import BACKENDS, Responder, create_responder

log = logging.getLogger(__name__)

UI_TRANSCRIPT_WAIT = 2.0  # seconds to hold the transcript update for the user's transcript
ERROR_TEXT_LIMIT = 200  # characters of a failure shown in the UI; the traceback goes to the log
CHATBOT_HEIGHT = 500  # pixels; the default (400) leaves the transcript cramped next to the mic
UI_TITLE = "vaice"  # the heading above the mic


def _subtitle(responder: Responder) -> str:
    """The line under the heading: which backend is answering."""
    return f"{responder.name} backend · running locally"


def _error_text(exc: Exception) -> str:
    return " ".join(f"{type(exc).__name__}: {exc}".split())[:ERROR_TEXT_LIMIT]


def make_handler(responder: Responder, conversation: Conversation):
    def reply(audio: tuple[int, np.ndarray]):
        """Called by ReplyOnPause when the speaker stops; streams audio back as it is produced."""
        start = time.perf_counter()  # before history(): waiting for the last transcript counts as latency
        turn = conversation.start_turn()
        history = conversation.history(before=turn)
        first = True
        try:
            for chunk in responder.respond(audio, history, turn):
                if first:
                    log.info("[%s] time to first audio: %.2fs", responder.name, time.perf_counter() - start)
                    first = False
                turn.audio_handed_out(len(chunk[1]) / chunk[0])
                yield chunk
        except Exception as exc:
            # Swallowed on purpose: fastrtc turns an emit() error into a WebRTCError that kills the
            # connection's audio task, so re-raising here would leave the page mute until a reload.
            try:
                turn.fail(_error_text(exc))  # what was heard of the reply, plus why it stopped
            except OSError:
                # e.g. the conversation file can't be written: the error is still in the turn's
                # memory and shown below; only the persistence failed. Re-raising would brick the
                # connection just like the original bug.
                log.exception("[%s] could not save the failure on the turn", responder.name)
            log.exception("[%s] reply failed", responder.name)
            yield AdditionalOutputs(conversation.messages_for_ui())  # show the error, keep the connection
            return
        log.info("[%s] reply finished in %.2fs", responder.name, time.perf_counter() - start)
        # Not reached when the reply is interrupted: the next turn marks this one interrupted instead
        turn.finish()
        turn.transcribed.wait(UI_TRANSCRIPT_WAIT)
        yield AdditionalOutputs(conversation.messages_for_ui())

    return reply


def build_stream(responder: Responder, conversation: Conversation) -> Stream:
    chatbot = gr.Chatbot(type="messages", label="Conversation", height=CHATBOT_HEIGHT, allow_tags=False)
    stream = Stream(
        handler=InterruptSafeReplyOnPause(make_handler(responder, conversation)),
        modality="audio",
        mode="send-receive",
        additional_outputs=[chatbot],
        additional_outputs_handler=lambda old, new: new,
        # The heading says which backend answers, so the page tells you what you are talking to
        ui_args={"title": UI_TITLE, "subtitle": _subtitle(responder)},
    )
    with stream.ui:
        stream.ui.load(conversation.messages_for_ui, outputs=chatbot)  # show the stored conversation on every page load
        new_conversation = gr.Button("New conversation")

        def reset() -> list:
            for handler in list(getattr(stream, "handlers", {}).values()):  # per-connection handler copies
                stop_playing_reply(handler)  # silence a reply in flight; its turn is detached below
            conversation.reset()
            return []

        new_conversation.click(reset, outputs=chatbot)
        chatbot.clear(reset)  # the chatbot's own trash icon would otherwise only clear the view
    return stream


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="vaice", description="Talk to a local model, hear it answer.")
    parser.add_argument("--backend", choices=BACKENDS, default=config.BACKEND, help="default: $VAICE_BACKEND or auto")
    args = parser.parse_args(argv)
    if args.backend not in BACKENDS:  # argparse doesn't validate defaults, e.g. from VAICE_BACKEND
        parser.error(f"invalid backend {args.backend!r} (from VAICE_BACKEND): choose from {', '.join(BACKENDS)}")
    return args


def main() -> None:
    args = parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    responder = create_responder(args.backend)
    log.info("using %s backend", responder.name)

    conversation = Conversation(config.CONVERSATION_FILE, window=config.HISTORY_TURNS)
    log.info("conversation history in %s (%d turns loaded)", config.CONVERSATION_FILE, len(conversation))

    stream = build_stream(responder, conversation)
    stream.ui.launch(server_port=config.PORT)
