# Browser tasks

`VAICE_BROWSE=pi` (assistant mode, off by default) adds one skill, `browse(goal)`: when you ask for something to be
done on the web ("Open python.org and tell me the page title"), vaice says "Working in the browser." and hands the
goal to the [Pi coding agent](https://pi.dev), which drives a Chrome window through chrome-devtools-mcp. The task runs
**in the background**: you can keep talking (a new question is answered as usual), vaice speaks a short line when a page
opens, asks before any step that changes a page, and when the task ends the brain phrases the outcome ("The page title
is Welcome to Python.org").

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
  microphone. Anything else you say while a step waits cancels the task and is handled as a new question. Pi's
  own words are never spoken: only fixed sentences and the final answer.
- **Follow-up tasks.** After a task's answer the brain may start another ("now open the second one"), up to three
  browser tasks for one request; a fourth is refused with a spoken sentence. A skill you approve with a button (for
  example quitting an app) may likewise lead to a browser task.
- **Stop.** A **Stop the browser task** button shows while a task runs. *New conversation*, a new `browse` request and
  closing vaice stop it too (and so does `VAICE_BROWSE_SECONDS`, default 180). Only one task runs at a time.
- **Progress.** "Opening <host>." is spoken as a page opens, at most once every 8 seconds, never between a question
  and the moment it can be answered. Progress and step questions are not recorded in the conversation; the start line
  and the outcome are. The *Tool activity* panel shows a single `browse` entry per task.

Not verified: speaker echo (the microphone hearing a question or progress line and the VAD cutting the reply, #36;
headphones assumed), any model other than the one named above driving Pi, and a task that outlives a page reload of
the UI. Chrome and chrome-devtools-mcp are **not** in Pi's process group, so vaice also ends processes that were
started with its own profile directory when a task ends and at exit (nothing else is signalled).

See [Privacy](privacy.md) for what leaves the machine and [Architecture](architecture.md) for how the task is wired.
