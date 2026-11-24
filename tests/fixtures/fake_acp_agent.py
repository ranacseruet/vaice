"""A scripted fake ACP agent for tests/test_acp_agent.py, built on the SDK's agent side (`acp.run_agent`): it speaks ACP
on stdin/stdout, plays a scenario for each prompt and records what it received, in order.

Environment:
  FAKE_ACP_LOG            file that gets one JSON line per fact: {"start": ...}, {"initialize": ...},
                          {"new_session": ...}, {"prompt": ...}, {"permission": ...} (the client's answer),
                          {"cancel": 1}, {"close_session": 1}
                          and {"wire": ...} (what arrived, in order: "permission_response" or "cancel")
  FAKE_ACP_SCENARIO       the scenario (JSON): {"steps": [...], "stop_reason": "end_turn"}; a prompt whose text holds
                          `scenario=NAME` plays FAKE_ACP_SCENARIOS[NAME] (a JSON object of scenarios) instead
  FAKE_ACP_CLOSE_SESSION  1: advertise `sessionCapabilities.close`
  FAKE_ACP_IGNORE_CANCEL  1: a cancel is recorded and then ignored: the prompt never ends (an agent that hangs)
  FAKE_ACP_INIT_DELAY     seconds before `initialize` is answered (default 0)
  FAKE_ACP_INIT_FAIL_ONCE path to a marker file: the first `initialize` creates it and raises RequestError
  FAKE_ACP_IGNORE_TERM    1: SIGTERM is ignored (only SIGKILL ends it)

Steps (each is an object):
  {"text": "..."}                       an agent_message_chunk
  {"tool": "name", "args": {...}}       a tool_call (title = name, rawInput = args)
  {"permission": {"title": "...", "raw": {...}}}   a session/request_permission with pi-acp's two options; the answer
                                        is recorded. Options: "kinds" overrides them (a list of kinds); "raw" may be
                                        any JSON value; "content" (texts) and "locations" (paths) are sent with it
  {"parallel_permissions": [ask1, ask2]} send multiple session/request_permission calls concurrently (asyncio.gather)
  {"sleep": 0.2}                        wait (a cancel received meanwhile ends the prompt as "cancelled")
  {"exit": 3}                           the process exits at once (mid-task)
  {"fail": "text"}                      the prompt fails with a JSON-RPC error carrying `text`
After every step a received cancel ends the prompt with stop_reason "cancelled" (unless FAKE_ACP_IGNORE_CANCEL).
"""
import asyncio
import contextlib
import json
import os
import signal
import sys

from acp import PROTOCOL_VERSION, RequestError, run_agent
from acp.schema import (
    AgentCapabilities,
    AgentMessageChunk,
    ContentToolCallContent,
    Implementation,
    InitializeResponse,
    NewSessionResponse,
    PermissionOption,
    PromptResponse,
    SessionCapabilities,
    SessionCloseCapabilities,
    TextContentBlock,
    ToolCallLocation,
    ToolCallStart,
    ToolCallUpdate,
)

LOG = os.environ.get("FAKE_ACP_LOG")
SCENARIO = json.loads(os.environ.get("FAKE_ACP_SCENARIO", "{}"))
SCENARIOS = json.loads(os.environ.get("FAKE_ACP_SCENARIOS", "{}"))
IGNORE_CANCEL = os.environ.get("FAKE_ACP_IGNORE_CANCEL") == "1"

if os.environ.get("FAKE_ACP_IGNORE_TERM") == "1":
    signal.signal(signal.SIGTERM, signal.SIG_IGN)


def note(**fact) -> None:
    if LOG:
        with open(LOG, "a") as f:
            f.write(json.dumps(fact) + "\n")


def observe(event) -> None:
    """Stream observer: the order in which a permission response and a cancel arrive."""
    message = event.message
    if event.direction.value != "incoming":
        return
    if isinstance(message.get("result"), dict) and "outcome" in message["result"]:
        note(wire="permission_response")
    elif message.get("method") == "session/cancel":
        note(wire="cancel")


class FakeAgent:
    def __init__(self) -> None:
        self.cancelled = asyncio.Event()
        self.sessions = 0

    def on_connect(self, conn) -> None:
        self.conn = conn

    async def initialize(self, protocol_version, client_capabilities=None, client_info=None, **kwargs):
        note(initialize={
            "caps": client_capabilities.model_dump(by_alias=True, exclude_none=True) if client_capabilities else None,
            "client": client_info.name if client_info else None})
        await asyncio.sleep(float(os.environ.get("FAKE_ACP_INIT_DELAY", "0")))
        fail_once = os.environ.get("FAKE_ACP_INIT_FAIL_ONCE")
        if fail_once and not os.path.exists(fail_once):
            with open(fail_once, "w") as f:
                f.write("failed\n")
            raise RequestError(-32603, "init failed")
        sessions = SessionCapabilities(close=SessionCloseCapabilities()) if os.environ.get(
            "FAKE_ACP_CLOSE_SESSION") == "1" else SessionCapabilities()
        return InitializeResponse(protocol_version=PROTOCOL_VERSION,
                                  agent_info=Implementation(name="fake-acp", version="0"),
                                  agent_capabilities=AgentCapabilities(session_capabilities=sessions))

    async def new_session(self, cwd, mcp_servers=None, additional_directories=None, **kwargs):
        self.sessions += 1
        note(new_session={"cwd": cwd, "mcp_servers": [s.model_dump(by_alias=True, exclude_none=True)
                                                       for s in (mcp_servers or [])]})
        return NewSessionResponse(session_id=f"s{self.sessions}")

    async def close_session(self, session_id, **kwargs):
        note(close_session=session_id)

    async def cancel(self, session_id, **kwargs):
        note(cancel=1)
        self.cancelled.set()

    async def prompt(self, prompt, session_id, **kwargs):
        text = prompt[0].text
        note(prompt=text)
        self.cancelled.clear()
        scenario = SCENARIO
        for name, candidate in SCENARIOS.items():
            if f"scenario={name}" in text:
                scenario = candidate
        for step in scenario.get("steps", []):
            await self.play(session_id, step)
            if self.cancelled.is_set() and not IGNORE_CANCEL:
                return PromptResponse(stop_reason="cancelled")
        if self.cancelled.is_set() and IGNORE_CANCEL:
            await asyncio.sleep(3600)  # hung: never answers the prompt
        return PromptResponse(stop_reason=scenario.get("stop_reason", "end_turn"))

    async def play(self, session_id, step) -> None:
        if "text" in step:
            await self.conn.session_update(session_id, AgentMessageChunk(
                content=TextContentBlock(type="text", text=step["text"])))
        elif "tool" in step:
            await self.conn.session_update(session_id, ToolCallStart(
                tool_call_id=step.get("id", "t1"), title=step["tool"], raw_input=step.get("args"),
                status="in_progress"))
        elif "permission" in step:
            await self._ask_permission(session_id, step["permission"])
        elif "parallel_permissions" in step:
            await asyncio.gather(*(self._ask_permission(session_id, ask) for ask in step["parallel_permissions"]))
        elif "sleep" in step:
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(self.cancelled.wait(), step["sleep"])
        elif "exit" in step:
            os._exit(step["exit"])
        elif "fail" in step:
            raise RequestError(-32603, step["fail"])

    async def _ask_permission(self, session_id, ask) -> None:
        kinds = ask.get("kinds", ["allow_once", "reject_once"])
        options = [PermissionOption(option_id=kind.split("_")[0], name=kind, kind=kind) for kind in kinds]
        raw = ask.get("raw", {"method": "confirm", "title": ask.get("title"), "message": ask.get("message")})
        response = await self.conn.request_permission(
            session_id=session_id, options=options,
            tool_call=ToolCallUpdate(
                tool_call_id="pi-ui-1", title=ask.get("title", "tool"), kind="other", status="pending",
                raw_input=raw,
                content=[ContentToolCallContent(type="content", content=TextContentBlock(type="text", text=text))
                         for text in ask.get("content", [])] or None,
                locations=[ToolCallLocation(path=path) for path in ask.get("locations", [])] or None))
        outcome = response.outcome
        note(permission={"outcome": outcome.outcome, "option": getattr(outcome, "option_id", None),
                         "message": raw.get("message") if isinstance(raw, dict) else None})
        if outcome.outcome == "cancelled":  # a cancelled question ends the turn (an agent may do that)
            self.cancelled.set()


async def main() -> None:
    note(start={"pid": os.getpid(), "cwd": os.getcwd(), "env_names": sorted(os.environ),
                "env": {k: v for k, v in os.environ.items() if not k.startswith("FAKE_ACP_")}})
    print("fake acp agent stderr line", file=sys.stderr, flush=True)
    await run_agent(FakeAgent(), observers=[observe])


asyncio.run(main())
