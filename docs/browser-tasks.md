# Browser tasks

`VAICE_BROWSE=pi` (off by default) adds one skill, `browse(goal)`: when you ask for something to be done on the web
("Open python.org and tell me the page title"), the assistant says it is working in the browser and vaice hands the
goal to the [Pi coding agent](https://pi.dev), which drives a Chrome window through chrome-devtools-mcp. The task runs
**in the background and quietly**: you can keep talking (a new question is answered as usual and does not stop the
task), vaice says nothing about progress, asks before any step that changes a page, and when the task ends a fresh
realtime session says the outcome in a sentence or two ("The page title is Welcome to Python.org"). Ask "How is the
browser task going?" and the assistant reports where it is.

You install Pi yourself (version 1.5.2 or newer, `pi` on `PATH`); vaice never installs it. At startup vaice checks the
version: a missing `pi` or an older one is a warning and `browse` is simply not offered; a version it cannot parse
only warns.

```bash
VAICE_BROWSE=pi VAICE_BROWSE_URL=http://localhost:1234/v1 VAICE_BROWSE_MODEL=lfm2.5-24b-a2b .venv/bin/vaice up
```

- **Pi has a model of its own.** It uses the `VAICE_BROWSE_URL` endpoint (with `VAICE_BROWSE_MODEL` and
  `VAICE_BROWSE_API_KEY`), so `browse` is not offered without `VAICE_BROWSE_URL` (a warning says so). The local
  llama-server decides when to call `browse`. The key reaches Pi only through its process environment, never a file or a log
  line. Whatever the pages show, and your goal, goes to the browse endpoint (startup warns).
- **Chrome gets a profile of its own**, `<VAICE_DATA_DIR>/browse/chrome-profile` (a temporary directory when
  `VAICE_DATA_DIR` is empty), in a visible window. Logins you make there persist between tasks; your own Chrome is
  never touched. `VAICE_BROWSE_HEADLESS=1` hides the window. Navigation is unrestricted: the agent may open any site.
- **Steps that change a page ask first, and only the Run button approves.** Reading and navigating run at once.
  Clicking, filling, typing, pressing a key, running a script, uploading, dragging and answering a dialog ask: the panel
  shows the tool and its exact arguments with **Run** and **Cancel**, and a fresh realtime session says a fixed
  sentence naming the action and the site ("The browser task wants to click something on example.com. The details are
  on the screen. Press Run to allow it, or Cancel."). **Run** lets the task go on; **Cancel** stops it and says "Okay,
  I stopped there." Nothing you say answers it, and silence expires after 60 s as a no. Pi's own words are never
  spoken: only fixed sentences and the final answer. `VAICE_BROWSE_CONFIRM=off` turns these questions off: every
  browser step then runs unasked, forms and purchases included (startup warns).
- **Status.** `browser_status` answers "How is the browser task going?", "What is it doing?" and "Any update?" from
  what vaice already receives from Pi: the goal, how long it has run, the tools it called with the pages it visited,
  the page it is on, a step that waits for approval, and Pi's latest words between tools (cleaned and capped; the
  screen shows more). It says "no browser task is running" when there is none. No new command is sent to Pi.
- **Stop.** Say "Stop the browser task" (`stop_browsing`) or press the **Stop the browser task** button. *New
  conversation*, closing vaice and `VAICE_BROWSE_SECONDS` (default 180) stop it too; a new utterance does not. Only one
  task runs at a time: a second `browse` request while one works is refused, and one made while the previous task's
  outcome is still being said replaces it (the old outcome is dropped). A stop is quiet.
- **Quiet.** Nothing is spoken while the task works except the step heads-up and the outcome (or, when it fails or is
  declined, a fixed sentence). The outcome is added to the turn that asked for the task even if you spoke meanwhile.
  The *Tool activity* panel shows a single `browse` entry per task.
- **One browser path.** While `browse` is offered, an MCP server in `VAICE_MCP_CONFIG` that runs `chrome-devtools-mcp`
  is not started (a warning names it): with both, the model mixed the two browsers' tools. Set `VAICE_BROWSE=off` to
  use such a server directly.
- **No Google.** Pi is told to search with Bing or DuckDuckGo: Google shows an automated browser a bot check instead
  of results.

Not verified: speaker echo (the microphone hearing the heads-up and the VAD cutting a reply, #36; headphones assumed),
any model other than the one named above driving Pi, and a task that outlives a page reload of the UI. Chrome and
chrome-devtools-mcp are **not** in Pi's process group, so vaice also ends processes that were started with its own
profile directory when a task ends and at exit (nothing else is signalled).

See [Privacy](privacy.md) for what leaves the machine and [Architecture](architecture.md) for how the task is wired.
