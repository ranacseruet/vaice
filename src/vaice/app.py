"""Gradio/WebRTC voice UI: pause detection → responder (audio in, audio out)."""
import argparse
import json
import logging
import re
import shutil
import tempfile
import time
from collections.abc import Callable
from functools import partial
from pathlib import Path
from urllib.parse import urlsplit

import gradio as gr
import numpy as np
import requests
from fastrtc import AdditionalOutputs, Stream

from vaice import config, mcp_skills
from vaice.brain import ChatBrain
from vaice.browse import (
    API_KEY_ENV,
    GATE_POLICY,
    MIN_PI_VERSION,
    UNGATED_POLICY,
    PiAgent,
    default_browser_args,
    pi_version,
    write_pi_home,
)
from vaice.confirm import CONFIRM_SECONDS, Pending
from vaice.conversation import Conversation, Turn
from vaice.interruptible import InterruptSafeReplyOnPause, stop_playing_reply
from vaice.lifecycle import install_stop_handler, request_stop
from vaice.memory import Memory
from vaice.outbox import Outbox
from vaice.responders.assistant import AssistantResponder
from vaice.responders.base import Responder
from vaice.responders.llamacpp import LlamaCppResponder, check_server, fetch_tools, write_capable_tools
from vaice.server import ensure_server
from vaice.shell import RUN_SHELL
from vaice.skills import Skill
from vaice.vision import vision_skill

log = logging.getLogger(__name__)

UI_TRANSCRIPT_WAIT = 2.0  # seconds to hold the transcript update; the timer backstops one that lands late
VIEW_REFRESH_SECONDS = 1.0  # how often the chatbot re-checks the conversation (see refresh_view)
ERROR_TEXT_LIMIT = 200  # characters of a failure shown in the UI; the traceback goes to the log
CHATBOT_HEIGHT = 500  # pixels; the default (400) leaves the transcript cramped next to the mic
UI_TITLE = "vaice"  # the heading above the mic
BUTTON_ONLY = "Only the Run button can approve this one; saying yes will not."
NOTHING_REMEMBERED = "Nothing remembered."
BROWSE_ASK_EXTRA_SECONDS = 30.0  # a browser question waits this much longer than the confirmation's own expiry
_MARKDOWN_PUNCTUATION = re.compile(r"([!-/:-@\[-`{-~])")  # every ASCII punctuation character


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
    """The confirmation panel's components for `pending_view`: its visibility, its text, the request's id, and
    the visibility of its button row (*Run* / *Cancel*).

    The text holds the question and, in a code block, the skill and its arguments exactly as they will run
    (for a shell command the command line itself). A request a spoken yes cannot approve says so."""
    if view is None:
        return gr.update(visible=False), "", "", gr.update(visible=False)
    command = _shell_command(view) if view["name"] == RUN_SHELL else None
    if command is None:
        command = f"{view['name']} {view['arguments']}".rstrip()
    fence = _fence(command)
    notice = "" if view.get("voice", True) else f"{BUTTON_ONLY}\n\n"
    return (gr.update(visible=True), f"{view['question']}\n\n{notice}{fence}\n{command}\n{fence}", view["id"],
            gr.update(visible=True))


def memory_markdown(notes: list[str]) -> str:
    """The Memory panel's text: the notes as a list. They are model-written, so each is shown as plain text: every
    punctuation character is backslash-escaped (CommonMark), which leaves no link, image, HTML or emphasis."""
    if not notes:
        return NOTHING_REMEMBERED
    return "\n".join("- " + _MARKDOWN_PUNCTUATION.sub(lambda m: "\\" + m.group(1), note) for note in notes)


def refresh_view(conversation: Conversation, known: str, pending: Callable[[], Pending | None] | None = None,
                 memory: Callable[[], list[str]] | None = None,
                 browsing: Callable[[], bool] | None = None) -> tuple:
    """The timer's tick: the chatbot's and tool panel's new views when the conversation changed.

    The reply can finish before its parallel transcription session reports the transcript, so
    without this the "…" would sit in the chatbot until the next exchange or a page reload. Views
    are compared as JSON so an unchanged transcript is never re-rendered.

    With `pending` (a responder that asks for confirmation) the confirmation panel is part of the view:
    its visibility, text, the request's id (which the buttons pass back) and its button row's visibility
    come after the two views.
    With `browsing` (assistant mode with a browser: whether a browser task runs) the visibility of the task's *Stop*
    row comes after those.
    With `memory` (the notes, in assistant mode with memory on) the Memory panel's text comes after those.
    """
    view = (conversation.messages_for_ui(), conversation.tool_activity_for_ui())
    components = view
    if pending is not None:
        asked = pending_view(pending())
        view += (asked,)
        components += confirmation_panel(asked)
    if browsing is not None:
        running = browsing()
        view += (running,)
        components += (gr.update(visible=running),)
    if memory is not None:
        notes = memory()
        view += (notes,)
        components += (memory_markdown(notes),)
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
                # Assistant mode's spoken progress ("One moment.") is handed out here too, though it is not part of
                # the recorded reply: a barge-in then trims the text a little less than it should.
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
    # a responder that can speak with no reply in flight (assistant mode: the outcome of a click-approved call) gets the
    # outbox the audio handler drains; chat mode has no `attach_outbox`, so no outbox either
    attach_outbox = getattr(responder, "attach_outbox", None)
    outbox = Outbox() if attach_outbox is not None else None
    if attach_outbox is not None:
        attach_outbox(outbox)
    stream = Stream(
        handler=InterruptSafeReplyOnPause(make_handler(responder, conversation), can_interrupt=config.CAN_INTERRUPT,
                                          outbox=outbox),
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
                with gr.Row(visible=False) as run_row:
                    run_button = gr.Button("Run", variant="primary")
                    cancel_button = gr.Button("Cancel")
            pending_id = gr.State("")  # the request the panel shows: a click can only settle that one
            confirm_outputs = [confirm_panel, confirm_text, pending_id, run_row]
        # Assistant mode with a browser: a Stop button while a browser task runs (the 1 s timer shows and hides it)
        browsing = getattr(responder, "browsing", None) if hasattr(responder, "stop_task") else None
        browse_outputs: list = []
        if browsing is not None:
            with gr.Row(visible=False) as browse_row:
                browse_stop_button = gr.Button("Stop the browser task", variant="stop")
            browse_outputs = [browse_row]
        # Tool calls and their results, kept out of the conversation: collapsed by default, and the
        # 1 s timer (below) fills it when a tool round lands
        with gr.Accordion("Tool activity", open=False):
            tool_panel = gr.Markdown()
        # Assistant mode with memory on: the notes it keeps about you; "New conversation" leaves them alone
        memory = getattr(responder, "memory", None)
        memory_outputs: list = []
        if memory is not None:
            with gr.Accordion("Memory", open=False):
                memory_panel = gr.Markdown()
            memory_outputs = [memory_panel]

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

        if browsing is not None:
            def stop_browsing() -> tuple:
                responder.stop_task()
                return gr.update(visible=False), ""  # hide at once; the blank `seen` redraws on the next tick

            browse_stop_button.click(stop_browsing, outputs=[browse_row, seen])

        gr.Timer(VIEW_REFRESH_SECONDS).tick(
            partial(refresh_view, conversation, pending=pending, memory=None if memory is None else memory.notes,
                    browsing=browsing),
            inputs=[seen], outputs=[chatbot, tool_panel, *confirm_outputs, *browse_outputs, *memory_outputs, seen])
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
                "--tools (see docs/chat-mode.md)"
            )
    return LlamaCppResponder(tools=tools or None)


def _announce_vision() -> None:
    """Assistant mode: say which vision model looks at the screen. A host off this machine gets a warning that a
    screenshot is sent to it; nothing here touches the API key, which is never logged (not even its length)."""
    if not config.VISION_URL:
        return
    parts = urlsplit(config.VISION_URL)
    log.info("assistant vision (look_at_screen): %s, model %s", parts.netloc, config.VISION_MODEL or "(none set)")
    if not config.VISION_MODEL:
        log.warning("VAICE_VISION_MODEL is empty: most hosted endpoints require a model name")
    if parts.hostname not in config.LOCAL_HOSTS:
        log.warning(
            "the vision model at %s is not on this machine: each time you ask about your screen, a screenshot of "
            "your main display is sent to it (only then; it is deleted from this Mac right after)", parts.netloc)


def _vision_skill() -> Skill | None:
    """Assistant mode: `look_at_screen` over the configured vision endpoint (None when VAICE_VISION_URL is unset)."""
    if not config.VISION_URL:
        return None
    brain = ChatBrain.helper(config.VISION_URL, config.VISION_TIMEOUT, config.VISION_MODEL, config.VISION_API_KEY,
                             max_tokens=config.VISION_MAX_TOKENS or None)
    return vision_skill(brain, f"{urlsplit(config.VISION_URL).netloc}, model {config.VISION_MODEL or '(none set)'}")


def _open_memory() -> Memory | None:
    """Assistant mode: the notes it keeps (None when VAICE_ASSISTANT_MEMORY is off). The notes' text is never logged
    here, only how many there are and where."""
    if not config.ASSISTANT_MEMORY:
        log.info("assistant memory is off (VAICE_ASSISTANT_MEMORY): no notes are kept or used")
        return None
    memory = Memory(config.MEMORY_FILE)
    if config.MEMORY_FILE is None:
        log.info("VAICE_DATA_DIR is empty: the notes the assistant keeps stay in memory only (nothing is persisted)")
    else:
        log.info("assistant memory: %d notes loaded from %s", len(memory), config.MEMORY_FILE)
    return memory


def _open_browse() -> PiAgent | None:
    """Assistant mode: the browser agent for the `browse` skill (None when VAICE_BROWSE is off, or when it cannot be
    used: the reason is a warning, and the skill is then not offered). Needs its own endpoint (VAICE_BROWSE_URL: Pi
    uses that, with its key, which goes to the child's environment only and is never logged) and the `pi` command,
    which the user installs: its presence and version are checked, nothing is installed."""
    if config.BROWSE == "off":
        return None
    if not config.BROWSE_URL:
        log.warning("VAICE_BROWSE needs VAICE_BROWSE_URL (the browser agent has an endpoint of its own): "
                    "browse is not offered")
        return None
    if shutil.which("pi") is None:
        log.warning("VAICE_BROWSE=pi, but the `pi` command is not on PATH (install the Pi coding agent, "
                    "https://pi.dev): browse is not offered")
        return None
    wanted = ".".join(map(str, MIN_PI_VERSION))
    version = pi_version()
    if version is None:
        log.warning("could not read the version of `pi --version`; browse needs %s or newer and goes on", wanted)
    elif version < MIN_PI_VERSION:
        log.warning("`pi` is version %s, browse needs %s or newer: browse is not offered",
                    ".".join(map(str, version)), wanted)
        return None
    base = (config.DATA_DIR if config.DATA_DIR is not None else Path(tempfile.mkdtemp(prefix="vaice-"))) / "browse"
    profile = base / "chrome-profile"  # the browser's own profile: logins persist here, nothing of the user's
    args = default_browser_args(profile)
    if config.BROWSE_HEADLESS:
        args.append("--headless")
    try:
        write_pi_home(base / "pi", base_url=config.BROWSE_URL, model=config.BROWSE_MODEL,
                      api_key_env=API_KEY_ENV if config.BROWSE_API_KEY else None, browser_args=args)
    except OSError as exc:
        log.warning("could not write the browser agent's configuration under %s (%s): browse is not offered", base,
                    type(exc).__name__)
        return None
    if not config.BROWSE_MODEL:
        log.warning("VAICE_BROWSE_MODEL is empty: the browser agent needs a model name")
    log.info("browse: Pi in %s with %s, model %s, browser profile %s, %s window, up to %g s a task", base / "pi",
             urlsplit(config.BROWSE_URL).netloc, config.BROWSE_MODEL or "(none set)", profile,
             "no" if config.BROWSE_HEADLESS else "a visible", config.BROWSE_SECONDS)
    ungated = config.BROWSE_CONFIRM == "off"
    log.warning("browse: pages' text and what the browser agent does go to %s; %s",
                urlsplit(config.BROWSE_URL).netloc,
                "VAICE_BROWSE_CONFIRM=off: the agent clicks, types and submits forms without asking" if ungated else
                "a step that changes a page needs a spoken yes or the Run button, but navigation and reading do not")
    env = {API_KEY_ENV: config.BROWSE_API_KEY} if config.BROWSE_API_KEY else None
    return PiAgent(base / "pi", log_path=base / "pi.log", env=env,
                   ask_timeout_s=CONFIRM_SECONDS + BROWSE_ASK_EXTRA_SECONDS, chrome_profile=profile,
                   gate_policy=UNGATED_POLICY if ungated else GATE_POLICY)


def _open_mcp() -> mcp_skills.McpTools | None:
    """Assistant mode: start the MCP servers in VAICE_MCP_CONFIG (None when it is unset) and say what each offers:
    tool names only, never the file's `env` values."""
    if config.MCP_CONFIG is None:
        return None
    tools = mcp_skills.start(config.MCP_CONFIG, config.MCP_TIMEOUT)
    for name, offered in tools.summaries:
        log.info("mcp server %s: %d tools offered (%s)", name, len(offered), ", ".join(offered) or "none")
    return tools


def _warn_removed_settings() -> None:
    """One warning naming the settings that are set but no longer read, and why. Names only: a value may be a key."""
    removed = config.removed_settings()
    if removed:
        log.warning("these settings are no longer used and are ignored: %s",
                    "; ".join(f"{name} ({reason})" for name, reason in removed))


def main() -> None:
    up = parse_args().command == "up"

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    _warn_removed_settings()
    install_stop_handler()  # Ctrl+C: the responders give up, gradio closes the server
    managed = None  # the llama-server `vaice up` started, if any
    mcp_tools = None  # the MCP servers started for assistant mode, if any
    browse_agent = None  # the browser agent for assistant mode's `browse` skill, if any
    try:
        if up:
            managed = ensure_server()
        check_server()
        if config.MODE == "assistant":
            if config.LLAMACPP_TOOLS:
                log.warning("VAICE_LLAMACPP_TOOLS is ignored in assistant mode: the skills run in vaice, "
                            "not on llama-server")
            browse_agent = _open_browse()
            mcp_tools = _open_mcp()
            responder = AssistantResponder(memory=_open_memory(), vision=_vision_skill(),
                                           extra_skills=mcp_tools.skills if mcp_tools else (), browse=browse_agent)
            _announce_vision()
            if config.ASSISTANT_SHELL:
                log.warning("VAICE_ASSISTANT_SHELL is on: after you approve it, a shell command the assistant "
                            "proposes runs with vaice's own permissions (harmless ones by voice or click, "
                            "all others only by the Run button)")
            log.info("assistant mode: skills %s", ", ".join(responder.skills.names))
        else:
            if config.VISION_URL:
                log.warning("VAICE_VISION_URL is ignored in chat mode: the screen skill is for assistant mode only")
            if config.MCP_CONFIG is not None:
                log.warning("VAICE_MCP_CONFIG is ignored in chat mode: MCP tools are for assistant mode only")
            if config.BROWSE != "off":
                log.warning("VAICE_BROWSE is ignored in chat mode: the browser skill is for assistant mode only")
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
        if browse_agent is not None:
            browse_agent.close()  # ends the browser agent and the browser it started
        if mcp_tools is not None:
            mcp_tools.close()  # ends the servers' processes
        if managed is not None:
            managed.stop()
