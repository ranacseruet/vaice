"""Gradio/WebRTC voice UI: pause detection → responder (audio in, audio out)."""
import argparse
import json
import logging
import time
from functools import partial

import gradio as gr
import numpy as np
from fastrtc import AdditionalOutputs, Stream

from vaice import config
from vaice.conversation import Conversation, Turn
from vaice.interruptible import InterruptSafeReplyOnPause, stop_playing_reply
from vaice.lifecycle import install_stop_handler, request_stop
from vaice.responders import BACKENDS, Responder, create_responder

log = logging.getLogger(__name__)

UI_TRANSCRIPT_WAIT = 2.0  # seconds to hold the transcript update; the timer backstops one that lands late
VIEW_REFRESH_SECONDS = 1.0  # how often the chatbot re-checks the conversation (see refresh_view)
ERROR_TEXT_LIMIT = 200  # characters of a failure shown in the UI; the traceback goes to the log
CHATBOT_HEIGHT = 500  # pixels; the default (400) leaves the transcript cramped next to the mic
UI_TITLE = "vaice"  # the heading above the mic


def _subtitle(responder: Responder) -> str:
    """The line under the heading: which backend is answering."""
    return f"{responder.name} backend · running locally"


def _error_text(exc: Exception) -> str:
    return " ".join(f"{type(exc).__name__}: {exc}".split())[:ERROR_TEXT_LIMIT]


def refresh_view(conversation: Conversation, known: str) -> tuple:
    """The timer's tick: the chatbot's new view when the conversation changed, silence when not.

    The reply pushes its last update before a late transcript arrives (native transcribes after
    answering), so without this the "…" would sit in the chatbot until the next exchange or a
    page reload. Views are compared as JSON so an unchanged transcript is never re-rendered.
    """
    view = conversation.messages_for_ui()
    rendered = json.dumps(view, ensure_ascii=False)
    if rendered == known:
        return gr.skip(), known  # gr.skip(): leave the chatbot alone
    return view, rendered


def make_handler(responder: Responder, conversation: Conversation):
    def reply(audio: tuple[int, np.ndarray]):
        """Called by ReplyOnPause when the speaker stops; streams audio back as it is produced."""
        start = time.perf_counter()  # before history(): waiting for the last transcript counts as latency
        turn: Turn | None = None
        try:
            turn = conversation.start_turn()
            history = conversation.history(before=turn)
            first = True
            for chunk in responder.respond(audio, history, turn):
                if first:
                    log.info("[%s] time to first audio: %.2fs", responder.name, time.perf_counter() - start)
                    first = False
                turn.audio_handed_out(len(chunk[1]) / chunk[0])
                yield chunk
            log.info("[%s] reply finished in %.2fs", responder.name, time.perf_counter() - start)
            # Not reached when the reply is interrupted: the next turn marks this one interrupted instead
            turn.finish()
        except Exception as exc:
            # Swallowed on purpose: fastrtc wraps the per-connection emit() in
            # webrtc_error_handler (webrtc_connection_mixin.py), so an error that escapes the reply
            # generator comes back as a WebRTCError, which the player loop re-raises (utils.py:
            # player_worker_decode) — the connection's audio task dies and the page stays mute
            # until a reload. Catching the error here keeps the connection and shows the reason
            # in the chatbot instead of only in the log.
            log.exception("[%s] reply failed", responder.name)
            if turn is not None:
                try:
                    turn.fail(_error_text(exc))  # what was heard of the reply, plus why it stopped
                except Exception:
                    # The turn could not be recorded at all: the error is still in the turn's memory
                    # and shown in the chatbot; only the recording failed.
                    log.exception("[%s] could not record the failure on the turn", responder.name)
            try:
                # Guarded too: even reading the conversation back for the UI could fail (a bad turn
                # record). The log now holds both errors, the chatbot keeps what it is showing, and
                # the connection survives.
                messages = conversation.messages_for_ui()
                if turn is None:  # the turn could not even be started: the reason is shown on its own
                    messages = messages + [{"role": "assistant", "content": f"(error: {_error_text(exc)})"}]
            except Exception:
                log.exception("[%s] could not show the failure in the chatbot", responder.name)
                return
            yield AdditionalOutputs(messages)  # show the error, keep the connection
            return
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

        # refresh_view's backstop: catches conversation changes no reply update covered (a late
        # transcript, a barge-in's trimmed text); seen holds the view it last rendered
        seen = gr.State("")
        gr.Timer(VIEW_REFRESH_SECONDS).tick(partial(refresh_view, conversation), inputs=[seen], outputs=[chatbot, seen])
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
    install_stop_handler()  # Ctrl+C: the responders give up, gradio closes the server
    try:
        responder = create_responder(args.backend)
        log.info("using %s backend", responder.name)

        conversation = Conversation(config.CONVERSATION_FILE, window=config.HISTORY_TURNS)
        log.info("conversation history in %s (%d turns loaded)", config.CONVERSATION_FILE, len(conversation))

        stream = build_stream(responder, conversation)
        stream.ui.launch(server_port=config.PORT)
    except KeyboardInterrupt:
        # Ctrl+C while the model was loading: nothing to close, just leave
        log.info("stopped")
    finally:
        request_stop()  # the app is over: don't wait for a model call that may never return
