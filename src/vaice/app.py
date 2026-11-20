"""Gradio/WebRTC voice UI: pause detection → assistant (audio in, audio out)."""
import argparse
import json
import logging
import re
import shlex
import shutil
import tempfile
import time
from collections.abc import Callable
from functools import partial
from pathlib import Path
from urllib.parse import urlsplit

import gradio as gr
import numpy as np
from fastrtc import AdditionalOutputs, Stream

from vaice import config, mcp_skills
from vaice.assistant import Assistant
from vaice.browse import (
    API_KEY_ENV,
    GATE_POLICY,
    MIN_PI_VERSION,
    UNGATED_POLICY,
    BrowseAgent,
    acp_agent,
    check_gate,
    default_browser_args,
    launcher_path,
    new_nonce,
    pi_agent,
    pi_environment,
    pi_version,
    write_launcher,
    write_pi_home,
)
from vaice.confirm import CONFIRM_SECONDS, Pending
from vaice.conversation import Conversation, Turn
from vaice.interruptible import InterruptSafeReplyOnPause, stop_playing_reply
from vaice.lifecycle import install_stop_handler, request_stop
from vaice.memory import Memory
from vaice.outbox import Outbox
from vaice.realtime import check_server
from vaice.server import ensure_server
from vaice.shell import RUN_SHELL
from vaice.skills import Skill
from vaice.vision import VisionClient, vision_skill

log = logging.getLogger(__name__)

UI_TRANSCRIPT_WAIT = 2.0  # seconds to hold the transcript update; the timer backstops one that lands late
VIEW_REFRESH_SECONDS = 1.0  # how often the chatbot re-checks the conversation (see refresh_view)
ERROR_TEXT_LIMIT = 200  # characters of a failure shown in the UI; the traceback goes to the log
CHATBOT_HEIGHT = 500  # pixels; the default (400) leaves the transcript cramped next to the mic
UI_TITLE = "vaice"  # the heading above the mic
UI_SUBTITLE = "llama-server · assistant"  # the line under it names the voice service
NOTHING_REMEMBERED = "Nothing remembered."
BROWSE_ASK_EXTRA_SECONDS = 30.0  # a browser question waits this much longer than the confirmation's own expiry
_MARKDOWN_PUNCTUATION = re.compile(r"([!-/:-@\[-`{-~])")  # every ASCII punctuation character


def _error_text(exc: Exception) -> str:
    return " ".join(f"{type(exc).__name__}: {exc}".split())[:ERROR_TEXT_LIMIT]


def pending_view(pending: Pending | None) -> dict | None:
    """What the confirmation panel shows of a waiting request (None: nothing waits)."""
    if pending is None:
        return None
    return {"id": pending.id, "question": pending.question, "name": pending.name, "arguments": pending.arguments}


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
    (for a shell command the command line itself). Only the buttons answer it."""
    if view is None:
        return gr.update(visible=False), "", "", gr.update(visible=False)
    command = _shell_command(view) if view["name"] == RUN_SHELL else None
    if command is None:
        command = f"{view['name']} {view['arguments']}".rstrip()
    fence = _fence(command)
    return (gr.update(visible=True), f"{view['question']}\n\n{fence}\n{command}\n{fence}", view["id"],
            gr.update(visible=True))


def memory_markdown(notes: list[str]) -> str:
    """The Memory panel's text: the notes as a list. They are model-written, so each is shown as plain text: every
    punctuation character is backslash-escaped (CommonMark), which leaves no link, image, HTML or emphasis."""
    if not notes:
        return NOTHING_REMEMBERED
    return "\n".join("- " + _MARKDOWN_PUNCTUATION.sub(lambda m: "\\" + m.group(1), note) for note in notes)


def refresh_view(conversation: Conversation, known: str, pending: Callable[[], Pending | None],
                 browsing: Callable[[], bool], memory: Callable[[], list[str]] | None = None) -> tuple:
    """The timer's tick: the chatbot's and tool panel's new views when the conversation changed.

    The reply can finish before its parallel transcription session reports the transcript, so
    without this the "…" would sit in the chatbot until the next exchange or a page reload. Views
    are compared as JSON so an unchanged transcript is never re-rendered.

    The confirmation panel (`pending`: the request waiting for the user) is part of the view: its visibility, text,
    the request's id (which the buttons pass back) and its button row's visibility come after the two views.
    `browsing` (whether a browser task runs) gives the visibility of the task's *Stop* row, after those.
    With `memory` (the notes, with memory on) the Memory panel's text comes after those.
    """
    asked = pending_view(pending())
    view = (conversation.messages_for_ui(), conversation.tool_activity_for_ui(), asked)
    components = (*view[:2], *confirmation_panel(asked))
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


def make_handler(responder: Assistant, conversation: Conversation):
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
                    messages.append({"role": "assistant", "content": f"(error: {_error_text(exc)})"})
            except Exception:
                log.exception("[%s] could not show the failure in the chatbot", responder.name)
                return
            yield AdditionalOutputs(messages)  # show the error, keep the connection; the timer fills the tool panel
            return
        turn.transcribed.wait(UI_TRANSCRIPT_WAIT)
        yield AdditionalOutputs(conversation.messages_for_ui())  # the timer fills the tool panel within a second

    return reply


def build_stream(responder: Assistant, conversation: Conversation) -> Stream:
    chatbot = gr.Chatbot(type="messages", label="Conversation", height=CHATBOT_HEIGHT, allow_tags=False)
    # what the responder says with no reply in flight (the outcome of a Run click, a browser task) goes to the outbox
    # the audio handler drains
    outbox = Outbox()
    responder.attach_outbox(outbox)
    stream = Stream(
        handler=InterruptSafeReplyOnPause(make_handler(responder, conversation), can_interrupt=config.CAN_INTERRUPT,
                                          outbox=outbox),
        modality="audio",
        mode="send-receive",
        additional_outputs=[chatbot],
        additional_outputs_handler=lambda old, new: new,
        ui_args={"title": UI_TITLE, "subtitle": UI_SUBTITLE},
    )
    with stream.ui:
        stream.ui.load(conversation.messages_for_ui, outputs=chatbot)  # show the stored conversation on every page load
        new_conversation = gr.Button("New conversation")
        # refresh_view's backstop (the timer below) catches conversation changes no reply update covered
        # (a late transcript, a barge-in's trimmed text, a tool round); seen holds the views it last rendered
        seen = gr.State("")
        # A request waiting for the user's click (the responder's `pending`) shows here, above the tool panel
        with gr.Column(visible=False) as confirm_panel:
            confirm_text = gr.Markdown()
            with gr.Row(visible=False) as run_row:
                run_button = gr.Button("Run", variant="primary")
                cancel_button = gr.Button("Cancel")
        pending_id = gr.State("")  # the request the panel shows: a click can only settle that one
        confirm_outputs = [confirm_panel, confirm_text, pending_id, run_row]
        # A Stop button while a browser task runs (the 1 s timer shows and hides it)
        with gr.Row(visible=False) as browse_row:
            browse_stop_button = gr.Button("Stop the browser task", variant="stop")
        # Tool calls and their results, kept out of the conversation: collapsed by default, and the
        # 1 s timer (below) fills it when a tool round lands
        with gr.Accordion("Tool activity", open=False):
            tool_panel = gr.Markdown()
        # With memory on: the notes it keeps about you; "New conversation" leaves them alone
        memory = responder.memory
        memory_outputs: list = []
        if memory is not None:
            with gr.Accordion("Memory", open=False):
                memory_panel = gr.Markdown()
            memory_outputs = [memory_panel]

        def reset() -> tuple[list, str]:
            for handler in list(getattr(stream, "handlers", {}).values()):  # per-connection handler copies
                stop_playing_reply(handler)  # silence a reply in flight; its turn is detached below
            conversation.reset()
            responder.clear_pending()  # a request from the old conversation must not be approvable
            return [], ""

        new_conversation.click(reset, outputs=[chatbot, tool_panel])
        chatbot.clear(reset)  # the chatbot's own trash icon would otherwise only clear the view

        def settle(settle_request: Callable[[str], object], request_id: str) -> tuple:
            settle_request(request_id)  # a stale id settles nothing
            # hide at once; the blank `seen` makes the next tick draw whatever is pending now
            return gr.update(visible=False), "", ""

        run_button.click(partial(settle, responder.approve), inputs=[pending_id],
                         outputs=[confirm_panel, pending_id, seen])
        cancel_button.click(partial(settle, responder.cancel), inputs=[pending_id],
                            outputs=[confirm_panel, pending_id, seen])

        def stop_browsing() -> tuple:
            responder.stop_task()
            return gr.update(visible=False), ""  # hide at once; the blank `seen` redraws on the next tick

        browse_stop_button.click(stop_browsing, outputs=[browse_row, seen])

        gr.Timer(VIEW_REFRESH_SECONDS).tick(
            partial(refresh_view, conversation, pending=responder.pending, browsing=responder.browsing,
                    memory=None if memory is None else memory.notes),
            inputs=[seen], outputs=[chatbot, tool_panel, *confirm_outputs, browse_row, *memory_outputs, seen])
    return stream


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="vaice", description="Talk to a local llama-server audio model.")
    parser.add_argument(
        "command", nargs="?", choices=["up"],
        help="up: start llama-server first if none is running (and stop it at exit), then open the UI")
    return parser.parse_args(argv)


def _announce_vision() -> None:
    """Say which vision model looks at the screen. A host off this machine gets a warning that a
    screenshot is sent to it; nothing here touches the API key, which is never logged (not even its length)."""
    if not config.VISION_URL:
        return
    parts = urlsplit(config.VISION_URL)
    log.info("vision (look_at_screen): %s, model %s", parts.netloc, config.VISION_MODEL or "(none set)")
    if not config.VISION_MODEL:
        log.warning("VAICE_VISION_MODEL is empty: most hosted endpoints require a model name")
    if parts.hostname not in config.LOCAL_HOSTS:
        log.warning(
            "the vision model at %s is not on this machine: each time you ask about your screen, a screenshot of "
            "your main display is sent to it (only then; it is deleted from this Mac right after)", parts.netloc)


def _vision_skill() -> Skill | None:
    """`look_at_screen` over the configured vision endpoint (None when VAICE_VISION_URL is unset)."""
    if not config.VISION_URL:
        return None
    client = VisionClient(config.VISION_URL, config.VISION_TIMEOUT, config.VISION_MODEL, config.VISION_API_KEY,
                          max_tokens=config.VISION_MAX_TOKENS or None)
    return vision_skill(client, f"{urlsplit(config.VISION_URL).netloc}, model {config.VISION_MODEL or '(none set)'}")


def _open_memory() -> Memory | None:
    """The notes the assistant keeps (None when VAICE_MEMORY is off). The notes' text is never logged here, only how
    many there are and where."""
    if not config.MEMORY:
        log.info("assistant memory is off (VAICE_MEMORY): no notes are kept or used")
        return None
    memory = Memory(config.MEMORY_FILE)
    if config.MEMORY_FILE is None:
        log.info("VAICE_DATA_DIR is empty: the notes the assistant keeps stay in memory only (nothing is persisted)")
    else:
        log.info("assistant memory: %d notes loaded from %s", len(memory), config.MEMORY_FILE)
    return memory


def _open_browse() -> BrowseAgent | None:
    """The browser agent for the `browse` skill (None when VAICE_BROWSE is off, or when it cannot be used: the reason
    is a warning, and the skill is then not offered). `pi` is the Pi coding agent through pi-acp, `acp` any ACP agent
    (VAICE_BROWSE_AGENT). Both need the ACP client SDK (the `browse` extra) and `npx` (pi-acp, the browser's
    server); what the user installs (`pi`, the agent) is checked, nothing is installed."""
    if config.BROWSE == "off":
        return None
    if config.BROWSE == "acp" and not config.BROWSE_AGENT:
        log.warning("VAICE_BROWSE=acp needs VAICE_BROWSE_AGENT (the agent's command line, e.g. `opencode acp`): "
                    "browse is not offered")
        return None
    if config.BROWSE == "pi" and not config.BROWSE_URL:
        log.warning("VAICE_BROWSE needs VAICE_BROWSE_URL (the browser agent has an endpoint of its own): "
                    "browse is not offered")
        return None
    argv: list[str] = []
    if config.BROWSE == "acp":
        try:
            argv = shlex.split(config.BROWSE_AGENT)
        except ValueError:
            argv = []
        if not argv:
            log.warning("VAICE_BROWSE_AGENT is not a command line (empty, or unbalanced quotes): browse is not "
                        "offered")
            return None
        if shutil.which(argv[0]) is None:
            log.warning("VAICE_BROWSE_AGENT: `%s` is not on PATH: browse is not offered", argv[0])
            return None
    elif shutil.which("pi") is None:
        log.warning("VAICE_BROWSE=pi, but the `pi` command is not on PATH (install the Pi coding agent, "
                    "https://pi.dev): browse is not offered")
        return None
    if shutil.which("npx") is None:
        log.warning("VAICE_BROWSE needs `npx` on PATH (Node.js: it starts %s and the browser's server): "
                    "browse is not offered", "pi-acp" if config.BROWSE == "pi" else "the browser's server")
        return None
    try:
        import acp  # noqa: F401 - checks the optional `browse` extra is installed
    except ImportError:
        log.warning("VAICE_BROWSE needs the ACP client SDK: run `uv sync --extra browse` (or `pip install "
                    "'vaice[browse]'`): browse is not offered")
        return None
    if config.BROWSE == "pi":
        return _open_pi_browse()
    return _open_acp_browse(argv)


def _browse_base() -> tuple[Path, Path, list[str]]:
    """Where the browser agent's files go, the browser's own profile (logins persist there, nothing of the user's)
    and chrome-devtools-mcp's arguments."""
    base = (config.DATA_DIR if config.DATA_DIR is not None else Path(tempfile.mkdtemp(prefix="vaice-"))) / "browse"
    profile = base / "chrome-profile"
    args = default_browser_args(profile)
    if config.BROWSE_HEADLESS:
        args.append("--headless")
    return base, profile, args


def _open_pi_browse() -> BrowseAgent | None:
    """Pi through pi-acp. Pi needs its own endpoint (VAICE_BROWSE_URL), with a key that goes to the child's
    environment only and is never logged, and a version this was measured against. The gate extension is checked
    against a real Pi before the skill is offered (`browse.check_gate`)."""
    wanted = ".".join(map(str, MIN_PI_VERSION))
    version = pi_version()
    if version is None:
        log.warning("could not read the version of `pi --version`; browse needs %s or newer and goes on", wanted)
    elif version < MIN_PI_VERSION:
        log.warning("`pi` is version %s, browse needs %s or newer: browse is not offered",
                    ".".join(map(str, version)), wanted)
        return None
    base, profile, args = _browse_base()
    home = base / "pi"
    try:
        write_pi_home(home, base_url=config.BROWSE_URL, model=config.BROWSE_MODEL,
                      api_key_env=API_KEY_ENV if config.BROWSE_API_KEY else None, browser_args=args)
        write_launcher(launcher_path(home))
    except OSError as exc:
        log.warning("could not write the browser agent's configuration under %s (%s): browse is not offered", base,
                    type(exc).__name__)
        return None
    if not config.BROWSE_MODEL:
        log.warning("VAICE_BROWSE_MODEL is empty: the browser agent needs a model name")
    ungated = config.BROWSE_CONFIRM == "off"
    ask_timeout_s = CONFIRM_SECONDS + BROWSE_ASK_EXTRA_SECONDS
    nonce = new_nonce()
    env = pi_environment(home, ask_timeout_s=ask_timeout_s, gate_policy=UNGATED_POLICY if ungated else GATE_POLICY,
                         nonce=nonce, api_key=config.BROWSE_API_KEY)
    if not check_gate(home, env, log_path=base / "pi.log"):
        log.warning("Pi did not report vaice's tool gate (see %s): without it the browser's tools would run "
                    "unasked, so browse is not offered", base / "pi.log")
        return None
    log.info("browse: Pi through pi-acp in %s with %s, model %s, browser profile %s, %s window, up to %g s a task",
             home, urlsplit(config.BROWSE_URL).netloc, config.BROWSE_MODEL or "(none set)", profile,
             "no" if config.BROWSE_HEADLESS else "a visible", config.BROWSE_SECONDS)
    log.warning("browse: pages' text and what the browser agent does go to %s; %s",
                urlsplit(config.BROWSE_URL).netloc,
                "VAICE_BROWSE_CONFIRM=off: the agent clicks, types and submits forms without asking" if ungated else
                "a step that changes a page needs the Run button, but navigation and reading do not")
    return pi_agent(home, env=env, nonce=nonce, log_path=base / "pi.log", ask_timeout_s=ask_timeout_s,
                    chrome_profile=profile)


def _open_acp_browse(argv: list[str]) -> BrowseAgent | None:
    """Any ACP agent (VAICE_BROWSE_AGENT). vaice can only gate what the agent asks about: its own tools and its
    permission mode are set in its own config, which is said at start-up. The Pi settings do not apply."""
    ignored = [name for name, value in (("VAICE_BROWSE_URL", config.BROWSE_URL), ("VAICE_BROWSE_MODEL",
               config.BROWSE_MODEL), ("VAICE_BROWSE_API_KEY", config.BROWSE_API_KEY)) if value]
    if ignored:
        log.warning("%s: set, but they configure Pi only and are ignored with VAICE_BROWSE=acp (the agent has its "
                    "own model and login)", ", ".join(ignored))
    base, profile, args = _browse_base()
    ungated = config.BROWSE_CONFIRM == "off"
    try:
        agent = acp_agent(argv, work=base / "acp-work", log_path=base / "acp.log", browser_args=args,
                          ask_timeout_s=CONFIRM_SECONDS + BROWSE_ASK_EXTRA_SECONDS, chrome_profile=profile,
                          confirm=not ungated)
    except OSError as exc:
        log.warning("could not make the browser agent's working directory under %s (%s): browse is not offered",
                    base, type(exc).__name__)
        return None
    log.info("browse: the ACP agent `%s`, browser profile %s, %s window, up to %g s a task", argv[0], profile,
             "no" if config.BROWSE_HEADLESS else "a visible", config.BROWSE_SECONDS)
    log.warning("browse: pages' text and what the agent does go to the agent's own model endpoint; vaice asks "
                "before every step the agent asks about%s, but the agent's own tools and permission mode are set "
                "in its config, not by vaice",
                " (VAICE_BROWSE_CONFIRM=off: it allows each without asking)" if ungated else "")
    return agent


def _open_mcp() -> mcp_skills.McpTools | None:
    """Start the MCP servers in VAICE_MCP_CONFIG (None when it is unset) and say what each offers:
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
    install_stop_handler()  # Ctrl+C: the assistant gives up, gradio closes the server
    managed = None  # the llama-server `vaice up` started, if any
    mcp_tools = None  # the MCP servers started, if any
    browse_agent = None  # the browser agent for the `browse` skill, if any
    try:
        if up:
            managed = ensure_server()
        check_server()
        browse_agent = _open_browse()
        mcp_tools = _open_mcp()
        responder = Assistant(memory=_open_memory(), vision=_vision_skill(),
                                       extra_skills=mcp_tools.skills if mcp_tools else (), browse=browse_agent)
        _announce_vision()
        if config.SHELL:
            log.warning("VAICE_SHELL is on: after you approve it with the Run button, a shell command the assistant "
                        "proposes runs with vaice's own permissions")
        log.info("assistant skills: %s", ", ".join(responder.skills.names))
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
