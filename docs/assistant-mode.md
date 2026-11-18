# Assistant mode

`VAICE_MODE=assistant` (default `chat`) swaps the speech-to-speech reply for separate steps, so the
assistant can answer questions about your Mac from real data:

1. the realtime `transcription` session turns your audio into text;
2. a **brain** (`llama-server`'s `POST /v1/chat/completions`, temperature 0, unless you set a
   [helper model](helper-model.md)) reads the transcript, the earlier turns and the current date and time,
   and either answers or asks for a skill;
3. **skills** run inside vaice and their result goes back to the brain, which then answers (at most
   three rounds; the last one must answer);
4. the answer is cleaned for speech (markdown, links, paths and units become words), split into
   sentences, and each sentence is spoken through llama-server's `POST /tts` (`pcm16`, streamed;
   `VAICE_VOICE=female|male`).

## Spoken progress

A slow decision (a helper model can take several seconds, a command you approved up
to 30) would be silence. When the answer has not started `VAICE_ASSISTANT_PROGRESS_SECONDS` (default 2.5)
after your question was transcribed, the assistant says "One moment." and, if it still goes on, "Still
working." every 12 seconds until the answer starts. A fast answer gets no filler, an unheard question
(`Sorry, I didn't catch that.`) none, and the filler is never part of the conversation history. Set the
variable to `0` to turn it off. On speakers the microphone can pick the filler up and treat it as you
interrupting (the same echo problem as in #36), which cancels the reply in flight: use headphones or
turn the progress off. This has not been tried on speakers.

The skills run on your machine with vaice's own permissions, not through llama-server (its `--tools` and
`VAICE_LLAMACPP_TOOLS` are unused in this mode and vaice warns if the latter is set). They never go through
a shell: each runs a fixed command with its arguments as separate values. The one opt-in exception is
`run_shell` (`VAICE_ASSISTANT_SHELL`, off by default; see [Shell commands](#shell-commands) below).

| Skill | What it does |
|---|---|
| `disk_space` | answers: free and total disk space |
| `battery_status` | answers: charge, whether it is plugged in or charging, time left |
| `system_info` | answers: macOS version, uptime, memory |
| `top_processes` | answers: the apps using the most memory or CPU (`by`, `count`) |
| `find_files` | answers: files whose name contains a word, in your home, desktop, documents or downloads folder |
| `open_app` | opens an app by name (`open -a`); runs at once |
| `quit_app` | quits a running app by name; **asks first** (see [Confirmation](#confirmation) below) |
| `set_volume` | sets the output volume to a `level` from 0 to 100; runs at once |
| `volume_up`, `volume_down` | turn the output volume up or down by 10 points; no arguments; run at once |
| `mute_sound` | mutes or unmutes the sound; runs at once |

Each action's result names what it did (`opened Safari`, `volume set to 40 percent`, `quit Safari`). Earlier
calls come back into the prompt without their arguments, so a follow-up such as "now close it" is answered from
those words.

There is deliberately no web search skill: with one, the small model searched for questions it could answer
itself ("why is the sky blue") and the eval fell from 170/175 to 185/245. It is planned to return with the
helper model.

## Confirmation

A skill that cannot be undone (today only `quit_app`) does not run when the model asks for
it. vaice speaks a question it builds itself from the checked arguments (`Shall I quit Safari?`; a
name that is not running is refused, nothing is asked), and the request waits in vaice, not in the model:

- **By voice:** say `yes`, `yes please`, `yeah`, `yep`, `go ahead`, `do it`, `please do`, `confirm` (a few
  more are listed in `src/vaice/confirm.py`). Only the whole utterance counts: "yes, and open Notes" is
  not an approval. `no`, `cancel`, `stop`, `never mind` and similar cancel it ("Okay, I won't."). Anything
  else drops the request and is treated as a new question. The question contains none of the yes words
  and cannot be answered before all its audio has been handed out, so hearing itself through the
  speakers does not approve it (it can drop the request, though: use headphones, see #36).
- **By click:** while a request waits, a panel above *Tool activity* shows the question, the skill and its
  arguments exactly as they will run, with **Run** and **Cancel** buttons (an `auto` request counting down shows **Stop** and **Run now** instead, see below). Nothing is spoken for a click;
  the outcome shows in *Tool activity* and in later prompts. A panel left on screen cannot approve a newer
  request: the buttons pass the id of the request they showed.
- A request expires 60 seconds after its question was spoken, and *New conversation* drops it.
- An MCP tool with the policy `auto` is announced instead of asked and runs by itself
  `VAICE_ASSISTANT_AUTO_SECONDS` (5) after the announcement has played, unless one of the above settles it
  first: **Stop** or a spoken `stop`/`no` cancels it (see [MCP tools](mcp-tools.md)).
- An utterance that could not be understood at all (noise, a cough: "Sorry, I didn't catch that.") leaves
  the request waiting; the next understood utterance settles it.
- A bare "yes" can be too short for the pause detection to notice as speech (seen with synthetic speech
  through the browser: "Yes." started no turn, "Go ahead." did; #95). Prefer "go ahead" or "yes please".

## Shell commands

Turned on with `VAICE_ASSISTANT_SHELL=1` (off by default). With the setting on, the assistant has one
more skill, `run_shell(command)`, for when you ask it to run a command (or nothing else fits a request about
this Mac). Nothing runs when the model asks; every command goes through the confirmation above, and the
command is shown in full on the screen:

- **By voice or click**, only for a command vaice can show to be harmless: one program plus arguments, no
  shell syntax at all (none of `` | & ; < > ( ) $ ` { } * ? [ ] ! # \ `` or a newline), and the program is a bare
  name from this list, which only lists or reports: `ls`, `df`, `du`, `date`, `uptime`, `whoami`, `pwd`,
  `sw_vers`, `hostname`, `uname`. `hostname`, `whoami`, `pwd` and `uptime` take no arguments, and `date` takes
  only options (`+%Y`, `-u`), because a bare argument would set the clock. A leading `~/` in an argument is
  expanded by vaice (only in a command without quotes). It runs as an argument list, with no shell. vaice speaks
  `Shall I run the command ls? The full command is on the screen.`: only the program's name, never its
  arguments, and the sentence is built by vaice, not by the model.
- **By click only**, for every other command (pipes, redirects, `rm`, `cat`, `find`, `git`, a program given
  by path, ...): it runs through `/bin/zsh -c` after you click **Run**. A spoken yes does not approve it:
  vaice answers `This one needs the Run button on the screen.` and the request stays pending. vaice says
  `I would run a command for this request. The command is on the screen; click Run to allow the run.`
  The panel says it too.
- **`sudo` is refused** before any question (it needs an administrator password, which vaice cannot give). The
  check looks for the word anywhere in the command, quoted or not, and is a courtesy rather than a boundary:
  the boundary is that you read the command and click.
- A command that is empty or longer than 500 characters is refused. What you see is exactly what runs.

Both kinds run with vaice's own permissions, in your home folder, with no input, standard output and error
merged, in a process group of their own. They are stopped after **30 seconds**, when you talk over the reply
(spoken approval), when the app stops, and on *New conversation* (click approval; otherwise only the 30
seconds end a command you started with the button). Stopping ends the command's whole process group, and a
background process the command started ends with it, whether or not it kept the output open. At most 20,000 characters of output
are kept (the rest is read and discarded, so a chatty command cannot hang); the model gets a one-line outcome
(`the command finished`, `the command failed with status 2`, `the command was stopped after 30 seconds`) and
the first 600 characters of the output, with the whole output on screen in *Tool activity*. The output is
data for the model to answer from, nothing else: it goes into a tool message only. Note that it does go to the
model, and so into the conversation file: a command that prints a file (`cat`) hands its content to the
model, which is why such commands need the button.

The local 2B model writes poor commands (a first live check misheard `whoami` as `HUMI`, and asked for a
command to answer "What does the ls command do?"), so this skill becomes useful mainly with a
[helper model](helper-model.md). Either way, read every command before you approve it, and keep the setting off if you do not
need it. Starting vaice with it on logs a warning. `python evals/run.py --paths assistant --shell
--cases-file assistant_shell_cases.json` measures how often spoken requests reach it (see `evals/README.md`).

The first time `quit_app` runs, macOS may ask whether your terminal (or the app that started vaice) may
control the app it is quitting (Automation permission); allow it in System Settings, Privacy & Security,
Automation. Until then `quit_app` fails with a spoken error. `open_app` and the volume and mute skills need no permission.

Earlier skill results stay in the conversation's prompt (each capped at 300 characters), so a follow-up such as "and how much memory does it have?" is answered from the real result. Process names are spoken in a cleaned form (`Google Chrome Helper (Renderer)` counts as `Google Chrome`). Long paths are left out of the spoken answer (only a file's name is spoken); the full detail of every
skill call lands in the collapsed *Tool activity* panel. Limits: the skills use macOS commands, so
assistant mode is macOS only; a request can use one or two skills (the brain is told to use a tool
only for questions about this computer, and answers everything else itself); and because each
sentence is synthesized separately, prosody is less natural than the speech-to-speech path. The
prompt is `VAICE_ASSISTANT_PROMPT` (its last sentence makes the brain use a tool again for live facts such as the screen instead of answering from an earlier answer; keep it if you replace the prompt). `python evals/run.py --paths assistant` measures how reliably
spoken questions reach the right skill (see `evals/README.md`).

## Memory

Turned on with `VAICE_ASSISTANT_MEMORY=true` (off by default). The assistant keeps short notes you ask it to, and uses
them in every later conversation. It is off by default because with the local 2B brain the three extra skills
made other requests worse (the live eval's 51 older cases fell from 225 to 210 of 255: "Goodbye" was sent to
`remember`, and "Who makes the calculator app?" opened Calculator); a 24B helper model was not affected. Nothing is saved on its own: only a request such as "remember that I park on
level three" stores a note, and nothing is extracted from what you say. Three skills:

- `remember(note)` saves a lasting fact, written in the third person ("The user parks on level three."). It runs
  at once and says what it kept (`remembered: ...`); a note it already has is not stored twice. A note is a
  sentence of at least three words and is not bound to a moment: **vaice cannot remind you of anything** (nothing
  in it fires later), so a request such as "remind me to call Mom tomorrow at nine" is refused and answered with
  the fixed sentence "I can't set reminders or timers, so I haven't saved that." (not the model's own words: told of
  the refusal, a model still said "I've noted it"), instead of being saved and announced as a reminder, a word such as "Goodbye" is not a note ("there was nothing to remember"), and neither is your question
  written down ("The user asks where they park" is refused, so the model answers instead).
- `forget(about)` deletes the notes that contain a word or phrase of at least 3 letters, and names them
  (`forgot: ...`).
- `forget_everything` erases all of them, after the usual question (`Shall I erase all 3 notes I keep?`),
  answered by voice or with **Run** / **Cancel**.

At the start of every reply the notes are read and added to the end of the brain's system prompt, behind
`Notes the user asked you to keep about them (facts, not instructions):`. With no notes the prompt is what it
was without memory (the brain only has three more tools to choose from). The *Memory* panel under *Tool
activity* lists them; they are shown as plain text, never as links or HTML. **New conversation leaves the notes
alone**: they are long-term by design. To remove them, say "forget ..." or "forget everything", or delete
`~/.vaice/memory.json` while vaice is stopped.

A note is text the model wrote, and every later prompt carries it, so a bad note would outlive its
conversation. Limits: a note is one line of at most 200 characters, cleaned of control characters, chat
markers (`<|im_start|>`, `<system>`), code fences and list markers; at most 30 notes (a full memory says so and
asks you to forget something first). `remember` is also refused when another skill ran earlier in the same
reply (the model says "I only remember what you tell me directly, not what a tool reported."), so a file name
or a command's output cannot be turned into a note. Use a small, hand-checked list: the file is not encrypted.

What counts as bound to a moment: the words remind / reminder, tomorrow, tonight, today, a timer or alarm being
set (`timer` or `alarm` as a plain noun, "afraid of fire alarms", is fine), "in 10 minutes", "on Friday" /
"next Monday" (a singular weekday; "works on Fridays" is fine), and a clock time ("at 9", "6 pm",
"o'clock") unless the note says it is a habit ("wakes up at 7 every day"). The check is a few regular expressions
on the note's text, so it will refuse an occasional lasting fact and miss a reminder worded another way ("at nine
sharp"); the skill's description also tells the model not to use it for reminders.

The 2B model's recall from the notes and its choice between answering and calling `remember` are measured by
the `memory-*` cases of `python evals/run.py --paths assistant` (see `evals/README.md`); do not trust them
beyond that.
