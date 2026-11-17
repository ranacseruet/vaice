"""Gradio/WebRTC voice UI: pause detection → responder (audio in, audio out)."""
import argparse
import json
import logging
import re
import time
from collections.abc import Callable
from functools import partial
from urllib.parse import urlsplit

import gradio as gr
import numpy as np
import requests
from fastrtc import AdditionalOutputs, Stream

from vaice import config
from vaice.confirm import Pending
from vaice.conversation import Conversation, Turn
from vaice.interruptible import InterruptSafeReplyOnPause, stop_playing_reply
from vaice.lifecycle import install_stop_handler, request_stop
from vaice.responders.assistant import AssistantResponder
from vaice.responders.base import Responder
from vaice.responders.llamacpp import LlamaCppResponder, check_server, fetch_tools, write_capable_tools
from vaice.server import ensure_server
from vaice.shell import RUN_SHELL

log = logging.getLogger(__name__)

UI_TRANSCRIPT_WAIT = 2.0  # seconds to hold the transcript update; the timer backstops one that lands late
VIEW_REFRESH_SECONDS = 1.0  # how often the chatbot re-checks the conversation (see refresh_view)
ERROR_TEXT_LIMIT = 200  # characters of a failure shown in the UI; the traceback goes to the log
CHATBOT_HEIGHT = 500  # pixels; the default (400) leaves the transcript cramped next to the mic
UI_TITLE = "vaice"  # the heading above the mic
BUTTON_ONLY = "Only the Run button can approve this one; saying yes will not."


def _subtitle() -> str:
    """The line under the heading names the voice service."""
    return "llama-server · assistant" if config.MODE == "assistant" else "llama-server · speech-to-speech"


def _error_text(exc: Exception) -> str:
    return " ".join(f"{type(exc).__name__}: {exc}".split())[:ERROR_TEXT_LIMIT]


def pending_view(pending: Pending | None) -> dict | None:
    """What the confirmation panel shows of a waiting request (None: nothing waits)."""
    if pending is None:
        return None
    return {"id": pending.id, "question": pending.question, "name": pending.name, "arguments": pending.arguments,
            "voice": pending.voice}


def _fence(text: str) -> str:
    """A code fence longer than any run of backticks in `text`, so the text cannot close it early."""
    return "`" * max(3, max((len(run) for run in re.findall(r"`+", text)), default=0) + 1)


def _shell_command(view: dict) -> str | None:
    """The command line of a `run_shell` request, parsed as the skill parses it, so it is what will run."""
    try:
        command = json.loads(view["arguments"])["command"]
    except (ValueError, KeyError, TypeError):
        return None
    return command if isinstance(command, str) else None


def confirmation_panel(view: dict | None) -> tuple:
    """The confirmation panel's components for `pending_view`: its visibility, its text, the request's id.

    The text holds the question and, in a code block, the skill and its arguments exactly as they will run
    (for a shell command the command line itself). A request a spoken yes cannot approve says so."""
    if view is None:
        return gr.update(visible=False), "", ""
    command = _shell_command(view) if view["name"] == RUN_SHELL else None
    if command is None:
        command = f"{view['name']} {view['arguments']}".rstrip()
    fence = _fence(command)
    only_button = "" if view.get("voice", True) else f"{BUTTON_ONLY}\n\n"
    return gr.update(visible=True), f"{view['question']}\n\n{only_button}{fence}\n{command}\n{fence}", view["id"]


def refresh_view(conversation: Conversation, known: str, pending: Callable[[], Pending | None] | None = None) -> tuple:
    """The timer's tick: the chatbot's and tool panel's new views when the conversation changed.

    The reply can finish before its parallel transcription session reports the transcript, so
    without this the "…" would sit in the chatbot until the next exchange or a page reload. Views
    are compared as JSON so an unchanged transcript is never re-rendered.

    With `pending` (a responder that asks for confirmation) the confirmation panel is part of the view:
    its visibility, text and the request's id (which the buttons pass back) come after the two views.
    """
    view = (conversation.messages_for_ui(), conversation.tool_activity_for_ui())
    components = view
    if pending is not None:
        asked = pending_view(pending())
        view += (asked,)
        components += confirmation_panel(asked)
    rendered = json.dumps(view, ensure_ascii=False)
    if rendered == known:
        return (*(gr.skip() for _ in components), known)  # gr.skip(): leave the components alone
    return (*components, rendered)


def make_handler(responder: Responder, conversation: Conversation):
    def reply(audio: tuple[int, np.ndarray]):
        """Called by ReplyOnPause when the speaker stops; streams audio back as it is produced."""
        start = time.perf_counter()  # before history(): waiting for the last transcript counts as latency
        turn: Turn | None = None
        try:
            turn = conversation.start_turn()
            history = conversation.history(before=turn, tools=getattr(responder, "tool_history", False))
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
                    messages.append({"role": "assistant", "content": f"(error: {_error_text(exc)})"})
            except Exception:
                log.exception("[%s] could not show the failure in the chatbot", responder.name)
                return
            yield AdditionalOutputs(messages)  # show the error, keep the connection; the timer fills the tool panel
            return
        turn.transcribed.wait(UI_TRANSCRIPT_WAIT)
        yield AdditionalOutputs(conversation.messages_for_ui())  # the timer fills the tool panel within a second

    return reply


def build_stream(responder: Responder, conversation: Conversation) -> Stream:
    chatbot = gr.Chatbot(type="messages", label="Conversation", height=CHATBOT_HEIGHT, allow_tags=False)
    stream = Stream(
        handler=InterruptSafeReplyOnPause(make_handler(responder, conversation), can_interrupt=config.CAN_INTERRUPT),
        modality="audio",
        mode="send-receive",
        additional_outputs=[chatbot],
        additional_outputs_handler=lambda old, new: new,
        ui_args={"title": UI_TITLE, "subtitle": _subtitle()},
    )
    with stream.ui:
        stream.ui.load(conversation.messages_for_ui, outputs=chatbot)  # show the stored conversation on every page load
        new_conversation = gr.Button("New conversation")
        # refresh_view's backstop (the timer below) catches conversation changes no reply update covered
        # (a late transcript, a barge-in's trimmed text, a tool round); seen holds the views it last rendered
        seen = gr.State("")
        # Assistant mode: a request waiting for the user's yes (the responder's `pending`, see Responder)
        # shows here, above the tool panel; chat mode has no such panel
        pending = getattr(responder, "pending", None)
        confirm_outputs: list = []
        if pending is not None:
            with gr.Column(visible=False) as confirm_panel:
                confirm_text = gr.Markdown()
                with gr.Row():
                    run_button = gr.Button("Run", variant="primary")
                    cancel_button = gr.Button("Cancel")
            pending_id = gr.State("")  # the request the panel shows: a click can only settle that one
            confirm_outputs = [confirm_panel, confirm_text, pending_id]
        # Tool calls and their results, kept out of the conversation: collapsed by default, and the
        # 1 s timer (below) fills it when a tool round lands
        with gr.Accordion("Tool activity", open=False):
            tool_panel = gr.Markdown()

        def reset() -> tuple[list, str]:
            for handler in list(getattr(stream, "handlers", {}).values()):  # per-connection handler copies
                stop_playing_reply(handler)  # silence a reply in flight; its turn is detached below
            conversation.reset()
            clear_pending = getattr(responder, "clear_pending", None)
            if clear_pending is not None:  # a request from the old conversation must not be approvable
                clear_pending()
            return [], ""

        new_conversation.click(reset, outputs=[chatbot, tool_panel])
        chatbot.clear(reset)  # the chatbot's own trash icon would otherwise only clear the view

        if pending is not None:
            def settle(settle_request: Callable[[str], object], request_id: str) -> tuple:
                settle_request(request_id)  # a stale id settles nothing
                # hide at once; the blank `seen` makes the next tick draw whatever is pending now
                return gr.update(visible=False), "", ""

            run_button.click(partial(settle, responder.approve), inputs=[pending_id],
                             outputs=[confirm_panel, pending_id, seen])
            cancel_button.click(partial(settle, responder.cancel), inputs=[pending_id],
                                outputs=[confirm_panel, pending_id, seen])

        gr.Timer(VIEW_REFRESH_SECONDS).tick(
            partial(refresh_view, conversation, pending=pending), inputs=[seen],
            outputs=[chatbot, tool_panel, *confirm_outputs, seen])
    return stream


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="vaice", description="Talk to a local llama-server audio model.")
    parser.add_argument(
        "command", nargs="?", choices=["up"],
        help="up: start llama-server first if none is running (and stop it at exit), then open the UI")
    return parser.parse_args(argv)


def _chat_responder() -> LlamaCppResponder:
    """Chat mode: the interleaved responder, with llama-server's built-in tools when VAICE_LLAMACPP_TOOLS asks."""
    tools: list[dict] = []
    if config.LLAMACPP_TOOLS:
        try:
            tools = fetch_tools(config.LLAMACPP_URL)
        except requests.RequestException as exc:
            log.warning("tools requested, but llama-server's /tools is unavailable (%s); continuing without", exc)
        if tools:
            log.info("tools enabled: %s", ", ".join(str(tool.get("name")) for tool in tools))
            try:
                risky = write_capable_tools(config.LLAMACPP_URL)
            except (requests.RequestException, ValueError, AttributeError, TypeError) as exc:
                log.warning("could not check whether the advertised tools can modify this machine (%s)", exc)
            else:
                if risky:
                    log.warning(
                        "llama-server advertises tools that can change this machine (%s); they run with "
                        "the server's permissions. Start llama-server with a read-only set, e.g. "
                        "--tools get_info,read_file", ", ".join(risky))
        else:
            log.warning(
                "tools requested (VAICE_LLAMACPP_TOOLS), but llama-server advertises none; start it with "
                "--tools (see README)"
            )
    return LlamaCppResponder(tools=tools or None)


def _announce_brain() -> None:
    """Assistant mode: say which brain answers. A helper off this machine gets a warning that says what is sent;
    nothing here touches the API key, which is never logged (not even its length)."""
    if not config.BRAIN_URL:
        return
    parts = urlsplit(config.BRAIN_URL)
    log.info("assistant brain: %s, model %s", parts.netloc, config.BRAIN_MODEL or "(none set)")
    if not config.BRAIN_MODEL:
        log.warning("VAICE_BRAIN_MODEL is empty: most hosted endpoints require a model name")
    if parts.hostname not in config.LOCAL_HOSTS:
        log.warning(
            "the brain at %s is not on this machine: what you say (as text), the earlier turns of the "
            "conversation and the skills' results (file names, process names, this computer's name, shell command "
            "output) are sent to it. Audio does not leave: speech recognition and speech output stay on the "
            "local llama-server", parts.netloc)


def main() -> None:
    up = parse_args().command == "up"

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    install_stop_handler()  # Ctrl+C: the responders give up, gradio closes the server
    managed = None  # the llama-server `vaice up` started, if any
    try:
        if up:
            managed = ensure_server()
        check_server()
        if config.MODE == "assistant":
            if config.LLAMACPP_TOOLS:
                log.warning("VAICE_LLAMACPP_TOOLS is ignored in assistant mode: the skills run in vaice, "
                            "not on llama-server")
            responder = AssistantResponder()
            _announce_brain()
            if config.ASSISTANT_SHELL:
                log.warning("VAICE_ASSISTANT_SHELL is on: after you approve it, a shell command the assistant "
                            "proposes runs with vaice's own permissions (harmless ones by voice or click, "
                            "all others only by the Run button)")
            log.info("assistant mode: skills %s", ", ".join(responder.skills.names))
        else:
            if config.BRAIN_URL:
                log.warning("VAICE_BRAIN_URL is ignored in chat mode: the helper brain is for assistant mode only")
            responder = _chat_responder()
        log.info("using llama-server at %s", config.LLAMACPP_URL)

        if config.CONVERSATION_FILE is None:  # VAICE_DATA_DIR="": the conversation is in memory only
            log.info("VAICE_DATA_DIR is empty: the conversation is kept in memory only (nothing is persisted)")
            conversation = Conversation(None, window=config.HISTORY_TURNS)
        else:
            conversation = Conversation(config.CONVERSATION_FILE, window=config.HISTORY_TURNS)
            log.info("conversation history in %s (%d turns loaded)", config.CONVERSATION_FILE, len(conversation))

        stream = build_stream(responder, conversation)
        # gradio's index.html always links /manifest.json, and launch() re-derives pwa from
        # its own kwarg (None -> Spaces-only), so without pwa=True the route 404s and the
        # browser logs a manifest fetch error on every load.
        stream.ui.launch(server_port=config.PORT, pwa=True, inbrowser=up)
    except KeyboardInterrupt:
        # Ctrl+C while the model was loading: nothing to close, just leave
        log.info("stopped")
    finally:
        request_stop()  # the app is over: don't wait for a model call that may never return
        if managed is not None:
            managed.stop()
