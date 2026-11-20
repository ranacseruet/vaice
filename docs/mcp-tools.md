# MCP tools

`VAICE_MCP_CONFIG` (off by default) lets the assistant use tools from [MCP](https://modelcontextprotocol.io)
servers: vaice starts each server you list, offers the tools you name as skills called `<server>__<tool>`, and runs a
call when the model asks for one, subject to the approval policy you gave that tool. It needs the optional extra
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
      "tools": {"search_notes": "run", "delete_note": "click"}
    }
  }
}
```

```bash
uv sync --extra mcp
VAICE_MCP_CONFIG=~/.vaice/mcp.json .venv/bin/vaice
```

- **Only the tools you list are offered.** A server can offer anything, and vaice cannot tell what a tool does, so
  an unlisted tool does not exist for the assistant (startup logs the names of the ones left out; a listed tool the
  server does not offer is a warning). A server's own annotations (`readOnlyHint` and the like) are claims, not facts,
  and change nothing.
- **Approval is a per-tool policy, one of two.** `"run"` runs at once, like the read skills; `"click"` asks first
  and only the **Run** button approves (nothing you say does).
  **`click` is the default**, and a plain list (`"tools": ["search_notes"]`) means all click: a tool you did not
  think about gets the strictest treatment, and you loosen it tool by tool. A tool that reads and cannot be turned
  against you is a fair `run`; anything that writes, sends, deletes or spends is not. The policy comes from the
  file alone: nothing the model says can make a tool looser. The policies `"voice"` and `"auto"` (a spoken yes, a
  timer) were removed: a config that still uses one exits at startup naming the server and the tool. After **Run**
  the stored call runs and the assistant says how it went a moment later, with no new utterance from you; starting to
  talk over it stops it.
- **Do not configure a browser server here.** For the web, use the `browse` skill
  ([Browser tasks](browser-tasks.md)): Pi drives the browser in the background and asks before it changes a page.
  A browser MCP server such as `chrome-devtools-mcp` in this file would give the model a second, competing
  browser path (a larger model mixed the two sets of tools), so leave it out.
- **The question names the tool and the server, never the arguments** (they come from the model and can be long or
  misleading); the panel shows the exact call, as for `run_shell`.
- **What a server sends is untrusted.** A tool's description (which the model reads) is cleaned to one line, capped
  at 300 characters and prefixed `(from <server>)`; a result reaches the model only as a tool result, is spoken up
  to 2000 characters (`(more on screen)` after that) and kept whole in the *Tool activity* detail; and `remember`
  refuses once such a skill has run in the reply, so a result cannot become a lasting note. None of this makes a
  malicious server safe: only list servers you would run anyway.
- **Startup.** One INFO line per server (`mcp server notes: 3 tools offered (...)`, names only). A server that does
  not start within `VAICE_MCP_TIMEOUT` is skipped with a warning and the others load; a mistake in the file (bad
  JSON, a `url` entry, a bad policy, two skills with the same name) exits with `vaice: VAICE_MCP_CONFIG: ...` naming
  the server and the problem, never an `env` value. A server's stderr goes to `<VAICE_DATA_DIR>/mcp-<name>.log`
  (mode 0600; a temporary directory when `VAICE_DATA_DIR` is empty). Servers are ended when vaice stops.
- **Calls** stop on barge-in, *New conversation* (a Run-approved one) and app stop, and end after
  `VAICE_MCP_TIMEOUT`; a failure is spoken, not raised. A server that dies mid-session is not restarted: its tools
  fail until you restart vaice.
- Not supported: remote (`url`) servers, resources and prompts (tools only), and reconnecting.
  Not verified: a real-world server other than the test fixture (`tests/fixtures/mcp_server.py`), and how well the
  2B model chooses among many MCP tools (no eval covers them): keep the list short.
