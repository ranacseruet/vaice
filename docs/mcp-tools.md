# MCP tools

`VAICE_MCP_CONFIG` (assistant mode, off by default) lets the assistant use tools from [MCP](https://modelcontextprotocol.io)
servers: vaice starts each server you list, offers the tools you name as skills called `<server>__<tool>`, and runs a
call when the brain asks for one, subject to the approval policy you gave that tool. It needs the optional extra
(`uv sync --extra mcp`); with the setting on and the package missing, vaice exits with that command.

The file has Claude Desktop's `mcpServers` shape, **stdio servers only** (`command`, `args`, optional `env` and
`cwd`), plus one key of vaice's own, `tools`:

```json
{
  "mcpServers": {
    "notes": {
      "command": "uvx",
      "args": ["some-notes-mcp"],
      "env": {"NOTES_DIR": "~/Notes"},
      "tools": {"search_notes": "run", "add_note": "auto", "delete_note": "click"}
    }
  }
}
```

```bash
uv sync --extra mcp
VAICE_MODE=assistant VAICE_MCP_CONFIG=~/.vaice/mcp.json .venv/bin/vaice
```

- **Only the tools you list are offered.** A server can offer anything, and vaice cannot tell what a tool does, so
  an unlisted tool does not exist for the assistant (startup logs the names of the ones left out; a listed tool the
  server does not offer is a warning). A server's own annotations (`readOnlyHint` and the like) are claims, not facts,
  and change nothing.
- **Approval is a per-tool policy.** `"run"` runs at once, like the read skills; `"voice"` asks first and a spoken
  yes or the **Run** button approves; `"auto"` announces the call and runs it by itself after a few seconds unless
  you stop it (next bullet); `"click"` asks first and only the **Run** button approves (a spoken yes is
  answered with a note and the request stays pending). **`click` is the default**, and a plain list
  (`"tools": ["search_notes"]`) means all click: a tool you did not think about gets the strictest treatment, and
  you loosen it tool by tool. A tool that reads and cannot be turned against you is a fair `run`; anything that
  writes, sends, deletes or spends is not. The policy comes from the file alone: nothing the model says can make a
  tool looser.
- **`auto` is a stop button with a timer.** The assistant says "Using navigate page from chrome in 5 seconds.";
  the panel shows the exact call and
  "Runs by itself in N s; press **Stop** to cancel it.", with the red **Stop** button first and **Run now** beside it. The countdown (`VAICE_ASSISTANT_AUTO_SECONDS`, default 5)
  runs from the **end of the announcement**, as vaice estimates it from the audio handed out to your speaker (the
  speech arrives faster than it plays, so the panel starts above 5 and counts down to 0); it is an estimate, not a
  measured playback position. A cut-off or failed announcement never starts it, so it never runs. **Stop**, or saying "stop", "cancel", "no" (any of the phrases in
  [Confirmation](assistant-mode.md#confirmation)) cancels it; saying anything else drops it and is handled as a new
  question; a spoken yes or **Run now** runs it at once. "New conversation" and stopping vaice end it too. Limit: speech
  still in progress when the timer fires has not been transcribed yet, so "stop" or "anything else" only cancels
  speech that has already ended; the **Stop** button always works. Whatever
  settles it first wins and it runs at most once. With `VAICE_ASSISTANT_AUTO_SECONDS=0` an `auto` tool behaves
  exactly like `click`. Use it for calls you would not mind being made while you look away (opening a page), not
  for anything that writes, sends or spends.
- **`auto` never follows another skill in the same reply.** If any other skill ran earlier in the reply, the
  assistant cannot know whether what it reported is trustworthy text (a page or a file can tell the model to call a
  tool), so that call is asked as a `click` one: no timer, only the **Run** button. Limit: content read in an
  **earlier turn** is not tracked, so a call the model makes in a later turn on the strength of it still counts as
  `auto`. That is the same limit the other policies have, and a reason to keep `auto` for harmless tools.
- **The question names the tool and the server, never the arguments** (they come from the model and can be long or
  misleading); the panel shows the exact call, as for `run_shell`.
- **What a server sends is untrusted.** A tool's description (which the brain reads) is cleaned to one line, capped
  at 300 characters and prefixed `(from <server>)`; a result reaches the brain only as a tool result, is spoken up
  to 2000 characters (`(more on screen)` after that) and kept whole in the *Tool activity* detail; and `remember`
  refuses once such a skill has run in the reply, so a result cannot become a lasting note. None of this makes a
  malicious server safe: only list servers you would run anyway.
- **Startup.** One INFO line per server (`mcp server notes: 3 tools offered (...)`, names only). A server that does
  not start within `VAICE_MCP_TIMEOUT` is skipped with a warning and the others load; a mistake in the file (bad
  JSON, a `url` entry, a bad policy, two skills with the same name) exits with `vaice: VAICE_MCP_CONFIG: ...` naming
  the server and the problem, never an `env` value. A server's stderr goes to `<VAICE_DATA_DIR>/mcp-<name>.log`
  (mode 0600; a temporary directory when `VAICE_DATA_DIR` is empty). Servers are ended when vaice stops.
- **Calls** stop on barge-in, *New conversation* (a click-approved one) and app stop, and end after
  `VAICE_MCP_TIMEOUT`; a failure is spoken, not raised. A server that dies mid-session is not restarted: its tools
  fail until you restart vaice.
- Not supported: remote (`url`) servers, resources and prompts (tools only), and reconnecting.
  Not verified: a real-world server other than the test fixture (`tests/fixtures/mcp_server.py`), and how well the
  2B brain chooses among many MCP tools (no eval covers them); with the helper model off, keep the list short.
