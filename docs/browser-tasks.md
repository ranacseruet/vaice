# Browser tasks

`VAICE_BROWSE=pi` or `VAICE_BROWSE=acp` (off by default) adds one skill, `browse(goal)`: when you ask for something to
be done or looked up on the web ("Open python.org and tell me the page title", "Research some coffee makers online"), vaice says it is working in the browser and
hands the goal to a browser agent, which drives a Chrome window through chrome-devtools-mcp. The task runs
**in the background and quietly**: you can keep talking (a new question is answered as usual and does not stop the
task), vaice says nothing about progress, asks before any step that changes a page, and when the task ends a fresh
realtime session says the outcome in a sentence or two ("The page title is Welcome to Python.org"). The whole answer
is added to the chat as its own message. Ask "How is the
browser task going?" and the assistant reports where it is.

vaice speaks [ACP](https://agentclientprotocol.com), the Agent Client Protocol, to the agent: one agent process runs
as long as vaice does and every task is a fresh session in it. Two ways to pick the agent:

- **`VAICE_BROWSE=pi`**: the [Pi coding agent](https://pi.dev), reached through `pi-acp` (pinned, started with
  `npx`). Pi has a model of its own, set with `VAICE_BROWSE_URL` / `VAICE_BROWSE_MODEL` / `VAICE_BROWSE_API_KEY`.
- **`VAICE_BROWSE=acp`**: any ACP agent, started with the command line in `VAICE_BROWSE_AGENT`. The agent has its own
  model and login, configured in the agent.

Both need Node.js (`npx` on `PATH`: it starts `pi-acp` and chrome-devtools-mcp) and the ACP client SDK, an optional
extra: `uv sync --extra browse` (or `pip install 'vaice[browse]'`). With either missing, startup warns and `browse` is
simply not offered; vaice never installs Pi or the agent.

```bash
# Pi: install it yourself (version 1.5.2 or newer, `pi` on PATH)
VAICE_BROWSE=pi VAICE_BROWSE_URL=http://localhost:1234/v1 VAICE_BROWSE_MODEL=lfm2.5-24b-a2b .venv/bin/vaice up

# any other ACP agent, for example OpenCode (read "Other agents" below first)
VAICE_BROWSE=acp VAICE_BROWSE_AGENT="opencode acp" .venv/bin/vaice up
```

At startup vaice checks the version of `pi` (a missing `pi` or an older one is a warning and `browse` is not offered;
a version it cannot parse only warns), and for `acp` that the agent's program is on `PATH`.

- **Pi has a model of its own.** It uses the `VAICE_BROWSE_URL` endpoint (with `VAICE_BROWSE_MODEL` and
  `VAICE_BROWSE_API_KEY`), so `browse` is not offered without `VAICE_BROWSE_URL` (a warning says so). The local
  llama-server decides when to call `browse`. The key reaches Pi only through its process environment, never a file or a
  log line, and Pi runs with an environment of its own (only what it needs to find Node and the network, not vaice's
  other keys) and a configuration directory of its own: your Pi skills, extensions and context files are not loaded.
  Whatever the pages show, and your goal, goes to the browse endpoint (startup warns).
- **`acp` agents run as you configured them.** vaice starts the command, in an empty directory of its own, with your
  environment minus every `VAICE_*` variable (the agent needs its own `HOME` and provider keys; vaice's keys stay out),
  and gives it the same pinned chrome-devtools server. The `VAICE_BROWSE_URL`, `_MODEL` and `_API_KEY` settings
  configure Pi only: with `acp` they are ignored and startup names them. Pages' text and your goal go to whatever
  endpoint the agent uses. **vaice can only gate what the agent asks about**: it offers no file system or terminal to
  the agent, but the agent's own tools (a shell, file edits) and its permission mode are set in the agent's config, not
  by vaice, so pick an agent and a config that you trust with the browser. Every permission request the agent makes
  goes to the Run button, whatever it is for.
- **Other agents: checked with OpenCode.** OpenCode 2.3.2 (`opencode acp`, 2026-11-20, a local LFM2.5-24B model in LM
  Studio, `"permission": {"*": "ask", "edit": "deny", "bash": "deny", "webfetch": "deny"}` in its project config)
  used the chrome-devtools server vaice passes, and Stop and the time limit worked. But it calls the browser tools
  from code it runs in its own `execute` tool, so a request covers a whole code block, not one step, and in one task
  11 such calls ran without any request at all: **with OpenCode the Run button does not cover every browser step**.
  It was also slow with that model (3 of 6 tasks hit the 180 s limit). Pi with vaice's gate asks before each step
  that changes a page.
- **Chrome gets a profile of its own**, `<VAICE_DATA_DIR>/browse/chrome-profile` (a temporary directory when
  `VAICE_DATA_DIR` is empty), in a visible window. Logins you make there persist between tasks; your own Chrome is
  never touched. `VAICE_BROWSE_HEADLESS=1` hides the window. Navigation is unrestricted: the agent may open any site.
- **Steps that change a page ask first, and only the Run button approves.** With Pi, reading and navigating run at once;
  clicking, filling, typing, pressing a key, running a script, uploading, dragging and answering a dialog ask (the gate
  that decides is a Pi extension vaice ships). With `acp`, a step asks when the agent asks. The panel
  shows the tool and its exact arguments with **Run** and **Cancel**, and a fresh realtime session is asked to say a
  fixed sentence naming the action and the site ("The browser task wants to click something on example.com. The details are
  on the screen. Press Run to allow it, or Cancel."). **Run** lets the task go on; **Cancel** stops it, and the agent is told to stop as well, and says "Okay,
  I stopped there." Nothing you say answers it, and silence expires after 60 s as a no. Asking for something else that
  needs Run meanwhile (quitting an app, a shell command) does not replace the step: the model says a step is waiting,
  and you answer the step or stop the task first. The agent's own words are
  never spoken: only fixed sentences and the final answer, which is the text after the agent's last tool call.
  `VAICE_BROWSE_CONFIRM=off` turns these questions off: every browser step then runs unasked, forms and purchases
  included (startup warns).
- **The gate is checked twice (Pi).** A gate that is not working would leave Pi's browser tools ungated (checked
  with Pi 1.5.2: a gate that throws while loading stops Pi from starting, but one that loads without hooking in lets
  the tools run). So at startup vaice runs Pi once and checks that the gate is loaded (otherwise `browse` is
  not offered), and every task must start with the gate's own word (a random value per run) before the agent's first
  tool call; a task that does not is ended and fails.
- **Status.** `browser_status` answers "How is the browser task going?", "What is it doing?" and "Any update?" from
  what vaice already receives from Pi: the goal, how long it has run, the tools it called with the pages it visited,
  the page it is on, a step that waits for approval, and Pi's latest words between tools (cleaned and capped; the
  screen shows more). No new command is sent to Pi.
- **Follow-ups after the task ended.** The model does not see a finished task's whole result: it sees the summary that
  was said and, in the system block of every reply while the task is kept, one line with the task's goal, how long ago
  it ended and its spoken summary, the agent's `Summary:` line or the start of its report ("You know what the last browser task found. It was asked: ...", cleaned
  of chat markers and capped at 400 characters; nothing for a stopped, declined or failed task), so "What did you
  find?" or "Which one was the cheapest?" can be answered from it; the model can also call `browser_status`, which
  reports the **last ended task**: its goal, how long ago it ended ("42 seconds ago", then minutes) and how it ended. A
  finished task adds its spoken summary and then more of the answer, cleaned for speech and capped at 700 characters
  as a whole (the summary comes first); a task that was stopped, declined at a step or failed is one sentence
  ("the last browser task, ..., was stopped 2 minutes ago", no error text: the panel has it). A task
  cancelled by the Stop button, `stop_browsing` or exit counts as stopped even if its agent still returned an answer:
  that answer is not kept. The last result lives in memory only (not in `conversation.json`, never in the model's history; only the summary line above is in its instructions): a new `browse` task replaces
  it at once (while that runs, the status is the running one), *New conversation* forgets it, and so does
  restarting vaice. With no task yet, it says "no browser task is running".
- **Stop.** Say "Stop the browser task" (`stop_browsing`) or press the **Stop the browser task** button. While
  a task runs, a line beside the button shows what it is doing: **Browser task running** · the goal (cut at 80
  characters) · the time run against the limit (`2:14 of 10:00`) · the tool calls so far (`7 steps`, left out at 0) ·
  the latest host it opened (`on bing.com`) · `waiting for Run` while a step waits for your approval (the
  confirmation panel has the details). The goal and the host are shown as plain text. The row, and so the line, goes
  when the agent is done: there is no "finished" line, the outcome is in the chat. *New
  conversation*, closing vaice and `VAICE_BROWSE_SECONDS` (default 600) stop it too; a new utterance does not. Only one
  task runs at a time: a second `browse` request while one works is refused, and one made while the previous task's
  outcome is still being said replaces it (the old outcome is dropped). A stop is quiet.
- **What you hear.** The reply that calls `browse` ends there: vaice has the voice model say "I'm working on it in the browser. I'll
  tell you when it's done." (`SAY`; it may reword it slightly), and the model is not asked to continue, so it cannot invent results. Other calls the model
  made in the same reply still run, but their results are not spoken. When the task ends you hear a short summary: the agent is
  asked to start its answer with a `Summary:` line of one or two plain sentences, and vaice speaks that line (without
  one, the answer with its narration before the first rule or heading dropped, cleaned for speech and capped).
- **The report in the chat.** The agent's whole answer appears in the chat as its own message headed "From the browser
  task:", with markdown links kept and images reduced to their alt text. It is stored with the turn (capped at 12000
  characters, `"reports"` in `conversation.json`) but is not part of what the model sees later. The *Tool activity*
  panel shows two rows per task: the start (`WORKING`) and the outcome (`finished: ...`, `stopped`, `declined`, `failed: ...`).
- **Quiet.** Nothing is spoken while the task works except the step heads-up and the outcome (or, when it fails or is
  declined, a fixed sentence). The outcome is added to the turn that asked for the task even if you spoke meanwhile; a step's
  heads-up is not (the panel shows the step), so it never becomes part of what the model sees later.
- **One browser path.** Do not also configure a browser MCP server (such as `chrome-devtools-mcp`) in
  `VAICE_MCP_CONFIG`: with both, the model mixed the two browsers' tools. vaice does not filter such a server out; it
  is up to the config file (see [MCP tools](mcp-tools.md)).
- **The window closes between tasks (Pi).** The browser belongs to the agent's session: when the next task starts,
  `pi-acp` closes the previous session's Pi and Chrome with it. The profile is kept, so logins persist, but the page
  you were on does not. Ask for a follow-up inside one goal. Another agent may keep its session, and so its browser,
  open: that is up to the agent.
- **No Google.** The agent is told to search with Bing or DuckDuckGo: Google shows an automated browser a bot check instead
  of results.
- **Today's date.** Each task's prompt carries the date, so "this week" or "the latest" means now: without it the
  agent's model researched a month of its own.
- **Research, not only sites.** The `browse` description names looking something up, researching a product and
  recent news or prices, as well as opening a website. With the earlier site-only wording the voice model answered
  "I can't browse the web" to most research requests: 13 of 30 calls without the screen skill, 0 of 30 with it, on
  `evals/browse_cases.json` (5 repeats, LFM3-Omni-2B Q8_0); 30 of 30 both ways with the new one, and no `browse` call on
  a question it should answer itself. `evals/browse_live.py` checks the agent's side: Pi 1.5.2 with a local Sage-A1.5 model
  answered 3 of 3 research goals in each of two runs (79 to 302 s, 16 to 43 tool calls), with nothing left holding
  the profile. Two of those six took longer than 180 s, the former default of `VAICE_BROWSE_SECONDS`, which is why the default is now 600; lower it
  if you want tasks cut off sooner.
- **Measured end to end.** `evals/browse_flow.py` (8 cases x 3, the real voice model, a fake agent replaying six
  recorded Pi answers; screen skill on / off) before and after this behaviour: the start reply invented results in
  14/24 and 16/24 runs before, 0/24 and 0/24 after; the spoken outcome named a finding in 13/24 and 13/24 before (in
  10/23 and 11/24 it was the agent's narration, "Let me compile my findings ..."), 24/24 and 24/24 after with no
  narration; the report was in the chat in 0/24 before, 24/24 after. The start sentence goes through the voice model (`SAY`): in all 72 runs it said "I'm working on it in the
  browser. I'll let you know when it's done." and nothing more. On `evals/browse_cases.json` x 5 no reply followed
  `browse` with `look_at_screen` (`browse-python-title` with the screen skill on did in 5 of 5 before). A second `browse`
  with the same goal in the starting reply now starts nothing and adds no event (before, about half of these replies
  made one and it was answered "busy"). `REPORT` and `SAY` as the outcome's wording
  scored the same (`--outcome say`: named 24/24, narration 0/24), so `REPORT` stayed. With the `Summary:` line in its
  prompt, Pi with the local model started 3 of 3 answers with one (`browse_live.py`, 29 to 259 s).
- **Follow-ups after the task.** Before `browser_status` kept the last result, a follow-up such as "What did you
  find?" could not see it and often started a new task. On `evals/browse_flow.py` (12 cases x 3, four of them added
  for this: "Which one was the cheapest?", "What were the results of the browser task?" and two new requests that
  should start a new task; screen skill on / off), the follow-up did what its case expects in 21/36 and 22/36 runs
  before, 26/36 and 27/36 after. "What were the results of the browser task?" called `browser_status` 3 of 3 times
  both before and after, but it answered "no browser task is running" before, so it named a finding 0/3 before and
  3/3 after. A new request after the task still started one in 6/6 runs both ways. Still open: a follow-up that
  should be answered from the result started a new task in 8/24 and 8/24 runs (11/24 and 10/24 before), mostly "What
  did you find?". The new `browser_status` description carries part of this: with the
  new code and the old description, the follow-up was right in 22/36 and 25/36 runs. On `evals/browse_cases.json` x 5,
  "What did the browser task find?" called `browser_status` 5/5 and 5/5 with the new description (3/5 and 0/5
  before, 2/5 and 2/5 on a rerun with the old one); "Tell me what you found in the browser." 0/5 both ways. All 29
  cases: 132/145 and 133/145 (132 and 129 before). The no-browse cases were 51/55 and 52/55 (53 and 54 before). A
  rerun of the old description, which for these single-turn cases is the same setup as before, gave 52 and 51, so
  that difference is noise.

Not verified: speaker echo (the microphone hearing the heads-up and the VAD cutting a reply, #36; headphones assumed),
any model other than the one named above driving Pi, ACP agents other than Pi (the protocol is generic, vaice's tests
use a scripted one), and a task that outlives a page reload of the UI. Chrome and chrome-devtools-mcp are **not** in
the agent's process group, so vaice also ends processes that were started with its own profile directory before a new
agent process starts and at exit (nothing else is signalled).

See [Privacy](privacy.md) for what leaves the machine and [Architecture](architecture.md) for how the task is wired.
