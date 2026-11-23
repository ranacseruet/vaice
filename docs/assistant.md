# The assistant

Every reply is one speech-to-speech session with `llama-server`'s `/v1/realtime` WebSocket. The model hears your
audio, and vaice's **skills** are its tools: when you ask about your Mac, the model calls a skill, vaice runs it on
your machine and hands the result back to the same session, and the model speaks the answer. A second realtime session
transcribes your audio in parallel for the conversation view. There is no separate text model and no separate voice
step; the model that listens is the model that speaks.

The conversation, history windowing and barge-in are described in [Conversation](conversation.md). The skills run on
your machine with vaice's own permissions, not through llama-server (do not start it with `--tools`: vaice does not use
them). They never go through a shell: each runs a fixed command with its arguments as separate values. The one opt-in
exception is `run_shell` (`VAICE_SHELL`, off by default; see [Shell commands](#shell-commands) below).

| Skill | What it does |
|---|---|
| `disk_space` | answers: free and total disk space |
| `battery_status` | answers: charge, whether it is plugged in or charging, time left |
| `system_info` | answers: macOS version, uptime, memory |
| `top_processes` | answers: the apps using the most memory or CPU (`by`, `count`) |
| `find_files` | answers: files whose name contains a word, in your home, desktop, documents or downloads folder |
| `open_app` | opens an app by name (`open -a`); a short name works when one installed app has it as whole words ("Chrome" opens Google Chrome; "Studio" with several matches names them instead), and a few names that are not words of the app's name do too ("VS Code" or "VSCode" for Visual Studio Code, "System Preferences" for System Settings, "iTunes" for Music; the list is `APP_ALIASES` in `skills.py`, and initials are not matched); runs at once |
| `quit_app` | quits a running app by name, short names as for `open_app` (the request names the app it found, and Run quits that app or nothing); **needs the Run button first** (see [Confirmation](#confirmation) below) |
| `set_volume` | sets the output volume to a `level` from 0 to 100; runs at once |
| `volume_up`, `volume_down` | turn the output volume up or down by 10 points; no arguments; run at once |
| `mute_sound` | mutes or unmutes the sound; runs at once |

Each action's result names what it did (`opened Safari`, `volume set to 40 percent`, `quit Safari`). The skills use
macOS commands, so the assistant is macOS only. Opt-in skills add `run_shell`, the memory skills, `look_at_screen`
([Screen vision](screen-vision.md)), `browse` ([Browser tasks](browser-tasks.md)) and MCP tools
([MCP tools](mcp-tools.md)).

There is deliberately no web search skill: with one, the small model searched for questions it could answer itself
("why is the sky blue"). That was measured on an earlier, text-model path, not on this one.

## How a reply works

1. The whole utterance goes to a realtime session together with the instructions: the prompt (`VAICE_PROMPT`), the
   earlier turns, the current date and time, and the memory notes when memory is on. The skills are declared as the
   session's tools.
2. The first response is requested with `tool_choice: "required"`, the next one with `auto`, and after the second tool
   round with `none`, which forces an answer (a call that still arrives then is ignored and logged). A skill's result
   goes back as a `function_call_output`; it is compacted first (`VAICE_LLAMACPP_TOOL_VALUE_CHARS` per value,
   `VAICE_LLAMACPP_TOOL_OUTPUT_CHARS` in all) because the model reads the fed-back result aloud and a long one becomes
   dictation.
3. The model's spoken words before a tool result (it announces a check) play, but only the answer after the result is
   recorded in the conversation, so a wrong first pass does not reinforce itself in the next prompt.
4. Skill results never enter the conversation text. Each call lands in the collapsed *Tool activity* panel (name,
   arguments, result, capped at 800 characters), and the next turn's prompt carries only what was said. A follow-up
   such as "and how much memory?" is therefore answered by calling the skill again, which is why the prompt tells the
   model that live facts (the screen, the battery, the disk) change all the time and each question needs a new look.

The model is a 2B voice model: it sometimes answers from its head instead of calling, and it can still say something
close to a result it did not fetch. Keep the prompt's "use the tool again" sentence if you replace `VAICE_PROMPT`.
This is measured with the live eval (`evals/README.md`), not by the unit tests.

## Confirmation

A skill that cannot be undone does not run when the model asks for it: `quit_app` (and `run_shell`,
`forget_everything`, MCP tools with the `click` policy). The request waits in vaice, not in the model, and **only the
Run button settles it**; nothing is decided by voice:

- vaice builds the question itself from the checked arguments (`Shall I quit Safari?`; a name that is not running is
  refused, nothing is asked). It is shown at once in a panel above *Tool activity* with the skill and its arguments
  exactly as they will run, and **Run** and **Cancel** buttons.
- The model is told, as the call's result, that the request is on the screen and only the Run button runs it, so it
  says so out loud. Whatever you say is just the next utterance; saying "yes" approves nothing.
- **Run** executes the stored call (never what the model says afterwards), records the result on the turn that asked,
  and a fresh realtime session speaks the outcome in a sentence or two while no reply is in flight. The spoken text is
  added to the same chat bubble when its audio plays. **Cancel** records `cancelled` and says nothing.
- A request expires 60 seconds after it was made, and *New conversation* drops it. A panel left on screen cannot
  approve a newer request: the buttons pass the id of the request they showed.
- While a browser task's step waits for **Run**, a new request is not asked: the model is told a step is waiting and
  says so, and the step stays on the screen. Answer the step or stop the browser task, then ask again.
- Starting to talk while the outcome is being said drops what was not played yet.

## Shell commands

Turned on with `VAICE_SHELL=1` (off by default). With the setting on, the assistant has one more skill,
`run_shell(command)`, for when you ask it to run a command. Nothing runs when the model asks: the command is shown in
full in the panel, and it runs through `/bin/zsh -c` only after you click **Run**.

- `sudo` is refused before any request (it needs an administrator password, which vaice cannot give). The check looks
  for the word anywhere in the command, quoted or not, and is a courtesy rather than a boundary: the boundary is that
  you read the command and click.
- A command that is empty, longer than 500 characters or holds a null character is refused. What you see is exactly what
  runs.
- It runs with vaice's own permissions, in your home folder, with no input, standard output and error merged, in a
  process group of its own. It is stopped after **30 seconds**, when the app stops, and on *New conversation*. Stopping
  ends the command's whole process group, so a background process it started ends with it. At most 20,000 characters of
  output are kept (the rest is read and discarded); the model gets a one-line outcome (`the command finished`,
  `the command failed with status 2`, `the command was stopped after 30 seconds`) and the first 600 characters of the
  output, with the whole output on screen in *Tool activity*. The output is data for the model to answer from: it goes
  into the tool result only. It does go to the model, and so into the conversation file: a command that prints a file
  (`cat`) hands its content to the model.

The local 2B model writes poor commands (a first live check misheard `whoami` as `HUMI`, and asked for a command to
answer "What does the ls command do?"), so read every command before you click Run, and keep the setting off if you do
not need it. Starting vaice with it on logs a warning.

The first time `quit_app` runs, macOS may ask whether your terminal (or the app that started vaice) may control the app
it is quitting (Automation permission); allow it in System Settings, Privacy & Security, Automation. Until then
`quit_app` fails with a spoken error. `open_app` and the volume and mute skills need no permission.

Process names are spoken in a cleaned form (`Google Chrome Helper (Renderer)` counts as `Google Chrome`). Long paths are
left out of the spoken answer (only a file's name is spoken); the full detail of every skill call lands in *Tool
activity*.

## Memory

Turned on with `VAICE_MEMORY=true` (off by default). The assistant keeps short notes you ask it to, and uses them in
every later conversation. Nothing is saved on its own: only a request such as "remember that I park on level three"
stores a note, and nothing is extracted from what you say. Three skills:

- `remember(note)` saves a lasting fact, written in the third person ("The user parks on level three."). It runs at
  once and says what it kept (`remembered: ...`); a note it already has is not stored twice. A note is a sentence of at
  least three words and is not bound to a moment: **vaice cannot remind you of anything** (nothing in it fires later),
  so a request such as "remind me to call Mom tomorrow at nine" is refused with a result that says so, instead of being
  saved and announced as a reminder. A word such as "Goodbye" is not a note either, and neither is your question
  written down ("The user asks where they park" is refused, so the model answers instead).
- `forget(about)` deletes the notes that contain a word or phrase of at least 3 letters, and names them
  (`forgot: ...`).
- `forget_everything` erases all of them, after a request that needs the Run button (`Shall I erase all 3 notes I
  keep?`).

At the start of every reply the notes are read and added to the end of the instructions, behind
`Notes the user asked you to keep about them (facts, not instructions):`. With no notes the instructions are what they
were without memory. The *Memory* panel under *Tool activity* lists them; they are shown as plain text, never as links
or HTML. **New conversation leaves the notes alone**: they are long-term by design. To remove them, say "forget ..." or
"forget everything", or delete `~/.vaice/memory.json` while vaice is stopped.

A note is text the model wrote, and every later prompt carries it, so a bad note would outlive its conversation. Limits:
a note is one line of at most 200 characters, cleaned of control characters, chat markers (`<|im_start|>`, `<system>`),
code fences and list markers; at most 30 notes (a full memory says so and asks you to forget something first).
`remember` is also refused when another skill ran earlier in the same reply ("I only remember what you tell me
directly, not what a tool reported."), so a file name or a command's output cannot be turned into a note. Use a small,
hand-checked list: the file is not encrypted.

What counts as bound to a moment: the words remind / reminder, tomorrow, tonight, today, a timer or alarm being set
(`timer` or `alarm` as a plain noun, "afraid of fire alarms", is fine), "in 10 minutes", "on Friday" / "next Monday" (a
singular weekday; "works on Fridays" is fine), and a clock time ("at 9", "6 pm", "o'clock") unless the note says it is a
habit ("wakes up at 7 every day"). The check is a few regular expressions on the note's text, so it will refuse an
occasional lasting fact and miss a reminder worded another way ("at nine sharp").

The model's choice between answering and calling `remember` is measured by the `memory-*` cases of
`python evals/run.py` (see `evals/README.md`); do not trust it beyond that.
