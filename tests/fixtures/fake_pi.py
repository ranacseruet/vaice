"""A fake `pi --mode rpc` for tests/test_browse.py: the same JSONL protocol (as measured against Pi 1.5.2), with a
behaviour chosen by environment variables, and a record of everything it received.

Environment:
  FAKE_PI_LOG            file that gets one JSON line per fact: {"start": ...}, {"cmd": ...}, {"orphaned": ...}, ...
  FAKE_PI_SCENARIO       what a prompt does (see run_prompt); a prompt whose text holds `scenario=NAME` overrides it
  FAKE_PI_NEW_SESSION_DELAY  seconds before new_session is answered (default 0.15, so a prompt sent too early is seen)
  FAKE_PI_NO_GATE        1: get_commands does not list vaice-gate
  FAKE_PI_IGNORE_TERM    1: SIGTERM is ignored (only SIGKILL ends it)

Like Pi 1.5.2, a prompt that arrives before new_session's response was sent starts on the old session, is replaced
and never settles: the fake records {"orphaned": true} and ignores that prompt.
"""
import json
import os
import queue
import re
import signal
import sys
import threading
import time

LOG = os.environ.get("FAKE_PI_LOG")
SCENARIO = os.environ.get("FAKE_PI_SCENARIO", "normal")
NEW_SESSION_DELAY = float(os.environ.get("FAKE_PI_NEW_SESSION_DELAY", "0.15"))

if os.environ.get("FAKE_PI_IGNORE_TERM") == "1":
    signal.signal(signal.SIGTERM, signal.SIG_IGN)

commands: queue.Queue = queue.Queue()
aborted = threading.Event()
answers: dict = {}  # extension_ui_response by id
out_lock = threading.Lock()


def note(**fact) -> None:
    if LOG:
        with open(LOG, "a") as f:
            f.write(json.dumps(fact) + "\n")


def emit(record: dict) -> None:
    with out_lock:
        sys.stdout.buffer.write(json.dumps(record).encode() + b"\n")
        sys.stdout.buffer.flush()


def reader() -> None:
    for line in iter(sys.stdin.buffer.readline, b""):
        commands.put(json.loads(line))
    commands.put(None)  # stdin closed: an orderly shutdown


def respond(cmd: dict, data=None, success=True, error=None) -> None:
    rec = {"type": "response", "id": cmd.get("id"), "command": cmd["type"], "success": success}
    if data is not None:
        rec["data"] = data
    if error:
        rec["error"] = error
    emit(rec)


def handle(cmd: dict) -> None:
    """A command that arrives while a prompt is running (or between commands)."""
    note(cmd=cmd["type"], **{k: v for k, v in cmd.items() if k in ("message", "confirmed", "cancelled", "id")})
    kind = cmd["type"]
    if kind == "abort":
        if SCENARIO == "abort_ignored":
            note(abort="ignored")
            return
        aborted.set()
        respond(cmd)
    elif kind == "extension_ui_response":
        answers[cmd["id"]] = cmd
    elif kind == "get_commands":
        names = [] if os.environ.get("FAKE_PI_NO_GATE") == "1" else [{"name": "vaice-gate", "source": "extension"}]
        respond(cmd, {"commands": names})
    elif kind == "new_session":
        time.sleep(NEW_SESSION_DELAY)
        if any(c is not None and c.get("type") == "prompt" for c in list(commands.queue)):
            note(orphaned=True)  # the prompt was sent before this reply: Pi would start it on the old session
            drop_prompts()
        respond(cmd, {"cancelled": False})
    elif kind == "prompt":
        pass  # handled by main()


def drop_prompts() -> None:
    kept = [c for c in list(commands.queue) if c is None or c.get("type") != "prompt"]
    with commands.mutex:
        commands.queue.clear()
        commands.queue.extend(kept)


def wait(seconds: float, until=None) -> bool:
    """Process commands for `seconds` (or until `until()` is true or an abort arrives). True when aborted."""
    end = time.monotonic() + seconds
    while time.monotonic() < end and not aborted.is_set() and not (until and until()):
        try:
            cmd = commands.get(timeout=0.02)
        except queue.Empty:
            continue
        if cmd is None:
            os._exit(0)
        handle(cmd)
    return aborted.is_set()


def assistant(text: str | None, tool: dict | None = None, stop: str = "stop", extra: list | None = None) -> None:
    content = list(extra or [])
    if text is not None:
        content.append({"type": "text", "text": text})
    if tool:
        content.append({"type": "toolCall", "id": "t1", "name": tool["name"], "arguments": tool["args"]})
    emit({"type": "message_start", "message": {"role": "assistant", "content": [], "stopReason": "pending"}})
    if text:
        emit({"type": "message_update", "usage": {},
              "assistantMessageEvent": {"type": "text_delta", "contentIndex": 0, "delta": text}})
    error = f"401 bad key {os.environ.get('VAICE_PI_API_KEY')}" if stop == "error" else None
    emit({"type": "message_end", "message": {"role": "assistant", "content": content, "stopReason": stop,
                                             "errorMessage": error}})


def tool(name: str, args: dict) -> None:
    emit({"type": "tool_execution_start", "toolCallId": "t1", "toolName": name, "args": args})
    emit({"type": "tool_execution_end", "toolCallId": "t1", "toolName": name, "result": {}, "isError": False,
          "durationMs": 5})


def settle(was_aborted: bool = False) -> None:
    emit({"type": "agent_end", "messages": [], "willRetry": False})
    emit({"type": "agent_settled", "aborted": was_aborted})


def run_prompt(cmd: dict) -> None:
    emit({"type": "agent_start"})
    emit({"type": "message_start", "message": {"role": "user", "content": cmd["message"]}})
    emit({"type": "message_end", "message": {"role": "user", "content": cmd["message"]}})
    if SCENARIO in ("normal", "ui_noise", "error", "unparsed"):
        assistant("I'll open the page.", {"name": "mcp__chrome__navigate_page", "args": {"url": "https://x.test"}},
                  stop="toolUse")
        tool("mcp__chrome__navigate_page", {"url": "https://x.test"})
        if SCENARIO == "ui_noise":
            emit({"type": "extension_ui_request", "id": "n1", "method": "notify", "message": "hello"})
            emit({"type": "extension_ui_request", "id": "s1", "method": "select", "title": "pick",
                  "options": ["a", "b"]})
            emit({"type": "extension_ui_request", "id": "i1", "method": "input", "title": "type"})
            wait(5, until=lambda: {"s1", "i1"} <= set(answers))
        if SCENARIO == "unparsed":
            sys.stdout.buffer.write(b"this is not json\n")
            sys.stdout.buffer.flush()
        if SCENARIO == "error":
            assistant(None, stop="error")
        else:
            assistant("The title is Python.", extra=[{"type": "thinking", "thinking": "hmm"}])
            assistant(None)  # a last message with no text: the answer is the one before it
        settle()
    elif SCENARIO == "confirm":
        assistant("I'll click it.", {"name": "mcp__chrome__click", "args": {"uid": "1_2"}}, stop="toolUse")
        emit({"type": "extension_ui_request", "id": "q1", "method": "confirm", "title": "mcp__chrome__click",
              "message": '{"uid":"1_2"}', "timeout": 60000})
        wait(10, until=lambda: "q1" in answers)
        if aborted.is_set():
            settle(True)
        elif answers.get("q1", {}).get("confirmed"):
            tool("mcp__chrome__click", {"uid": "1_2"})
            assistant("Clicked it.")
            settle()
        else:  # a refused step: the model routes around it unless the client aborts
            if wait(2):
                settle(True)
            else:
                note(routed_around=True)
                tool("mcp__chrome__navigate_page", {"url": "https://other.test"})
                assistant("I went another way.")
                settle()
    elif SCENARIO == "ext_error":
        emit({"type": "extension_error", "extensionPath": "/x/gate.ts", "event": "tool_call", "error": "boom"})
        if wait(5):
            settle(True)
    elif SCENARIO == "exit_midtask":
        tool("mcp__chrome__navigate_page", {"url": "https://x.test"})
        os._exit(3)
    elif SCENARIO == "handled":
        pass  # disposition "handled": no run, so no events
    elif SCENARIO in ("slow", "abort_ignored"):
        tool("mcp__chrome__navigate_page", {"url": "https://x.test"})
        if wait(60):
            settle(True)


def main() -> None:
    global SCENARIO
    note(start={"pid": os.getpid(), "argv": sys.argv[1:], "cwd": os.getcwd(),
                "env": {k: os.environ.get(k) for k in ("PI_CODING_AGENT_DIR", "HOME", "VAICE_GATE_POLICY",
                                                       "VAICE_GATE_TIMEOUT_MS", "VAICE_PI_API_KEY")},
                "env_names": sorted(os.environ)})
    print("fake pi stderr line", file=sys.stderr, flush=True)
    threading.Thread(target=reader, daemon=True).start()
    while True:
        cmd = commands.get()
        if cmd is None:
            return
        if cmd["type"] == "prompt":
            note(cmd="prompt", message=cmd["message"], id=cmd.get("id"))
            aborted.clear()
            named = re.search(r"scenario=(\w+)", cmd["message"])
            SCENARIO = named.group(1) if named else os.environ.get("FAKE_PI_SCENARIO", "normal")
            if SCENARIO == "handled":
                respond(cmd, {"disposition": "handled"})
            else:
                respond(cmd, {"disposition": "started"})
            run_prompt(cmd)
        else:
            if cmd["type"] == "new_session":
                aborted.clear()
            handle(cmd)


if __name__ == "__main__":
    main()
