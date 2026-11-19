# Browser tasks

`VAICE_BROWSE=pi` (assistant mode, off by default) adds one skill, `browse(goal)`: when you ask for something to be
done on the web ("Open python.org and tell me the page title"), vaice says "Working in the browser." and hands the
goal to the [Pi coding agent](https://pi.dev), which drives a Chrome window through chrome-devtools-mcp. The task runs
**in the background and quietly**: you can keep talking (a new question is answered as usual and does not stop the
task), vaice says nothing about progress, asks before any step that changes a page, and when the task ends the brain
phrases the outcome ("The page title is Welcome to Python.org"). Ask "How is the browser task going?" and the brain
reports where it is.

You install Pi yourself (version 1.5.2 or newer, `pi` on `PATH`); vaice never installs it. At startup vaice checks the
version: a missing `pi` or an older one is a warning and `browse` is simply not offered; a version it cannot parse
only warns.

```bash
VAICE_MODE=assistant VAICE_BRAIN_URL=http://localhost:1234/v1 VAICE_BRAIN_MODEL=lfm2.5-24b-a2b VAICE_BROWSE=pi .venv/bin/vaice up
```

- **It needs a helper brain.** Pi uses the `VAICE_BRAIN_URL` endpoint (with `VAICE_BRAIN_MODEL` and
  `VAICE_BRAIN_API_KEY`) as its model, so `browse` is not offered without `VAICE_BRAIN_URL` (a warning says so). The key
  reaches Pi only through its process environment, never a file or a log line. Whatever the pages show, and your goal,
  goes to that endpoint (startup warns when it is not local).
- **Chrome gets a profile of its own**, `<VAICE_DATA_DIR>/browse/chrome-profile` (a temporary directory when
  `VAICE_DATA_DIR` is empty), in a visible window. Logins you make there persist between tasks; your own Chrome is
  never touched. `VAICE_BROWSE_HEADLESS=1` hides the window. Navigation is unrestricted: the agent may open any site.
- **Steps that change a page ask first.** Reading and navigating run at once. Clicking, filling, typing, pressing a
  key, running a script, uploading, dragging and answering a dialog ask: vaice speaks a fixed sentence ("Shall I click
  something on example.com? The details are on the screen."), the panel shows the tool and its exact arguments, and a
  spoken yes or the **Run** button lets it go on; a spoken no or **Cancel** stops the task ("Okay, I stopped there.").
  Silence expires after 60 s as a no. A yes is acknowledged silently, so the speaker cannot echo into the
  microphone. Anything else you say while a step waits is an ordinary question: it is answered as usual and the step
  keeps waiting (until its 60 s run out). Pi's own words are never spoken: only fixed sentences and the final answer.
- **Follow-up tasks.** After a task's answer the brain may start another ("now open the second one"), up to three
  browser tasks for one request; a fourth is refused with a spoken sentence. A skill you approve with a button (for
  example quitting an app) may likewise lead to a browser task.
- **Status.** `browser_status` answers "How is the browser task going?", "What is it doing?" and "Any update?" from
  what vaice already receives from Pi: the goal, how long it has run, the tools it called with the pages it visited,
  the page it is on, a step that waits for approval, and Pi's latest words between tools (cleaned and capped; the
  screen shows more). It says "no browser task is running" when there is none. No new command is sent to Pi.
- **Stop.** Say "Stop the browser task" (`stop_browsing`) or press the **Stop the browser task** button. *New
  conversation*, closing vaice and `VAICE_BROWSE_SECONDS` (default 180) stop it too; a new utterance does not. Only one
  task runs at a time: a second `browse` request while one works is refused with a spoken sentence, and one made
  while the previous task's outcome is still being phrased replaces it (the old outcome is dropped).
- **Quiet.** Nothing is spoken while the task works except its opening line ("Working in the browser."), the step
  questions and the outcome. Step questions and the outcome are not recorded in the conversation, apart from the
  outcome, which is added to the turn that asked for the task even if you spoke meanwhile. The *Tool activity* panel
  shows a single `browse` entry per task.
- **One browser path.** While `browse` is offered, an MCP server in `VAICE_MCP_CONFIG` that runs `chrome-devtools-mcp`
  is not started (a warning names it): with both, the model mixed the two browsers' tools. Set `VAICE_BROWSE=off` to
  use such a server directly.
- **No Google.** Pi is told to search with Bing or DuckDuckGo: Google shows an automated browser a bot check instead
  of results.

Not verified: speaker echo (the microphone hearing a question and the VAD cutting the reply, #36;
headphones assumed), any model other than the one named above driving Pi, and a task that outlives a page reload of
the UI. Chrome and chrome-devtools-mcp are **not** in Pi's process group, so vaice also ends processes that were
started with its own profile directory when a task ends and at exit (nothing else is signalled).

See [Privacy](privacy.md) for what leaves the machine and [Architecture](architecture.md) for how the task is wired.
