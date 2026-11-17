# Vision

## North star

**A real-life Jarvis: an assistant you talk to, naturally and out loud, that gets things done on your computer.**

You say "find the report I was editing yesterday and open it", "what's eating my battery?" or "turn it down and quit
Slack", and it does it, tells you in a sentence what happened, and asks before anything it cannot take back. It
remembers what you told it, can see what you see when you ask, and reaches into your other tools. It is yours: it
runs on your machine, with models you choose, and nothing leaves it unless you decide so.

This is a direction, not a promise of a date. `vaice` is a single-user demo on the way there, and the rest of this page
says honestly how far it has come.

## What "Jarvis-style" means here

Five capabilities, each of which has to work before the experience does:

| Capability | What it means | Today |
|---|---|---|
| **Talk like a person** | Low latency, talk over it and it stops, short spoken answers (never paths or raw output read aloud), no push-to-talk ritual | Streaming replies, first audio in about 0.6–1.7 s, barge-in, spoken progress for slow answers. Still needs a browser tab and headphones |
| **Get things done** | Run real actions on the computer, from a growing set of skills, with free-form commands as the escape hatch | A handful of read and action skills for macOS, approved shell commands, any [MCP](https://modelcontextprotocol.io) tool |
| **Know the context** | Remember you, see your screen, keep the thread of a conversation | Multi-turn history with tool results, explicit notes, on-demand screen vision |
| **Be trustworthy** | You can see what it is about to do, say no, and trust that it will not do more | Confirmation by voice or click, exact calls shown, off-by-default opt-ins, cancellable everything |
| **Be yours** | Local first, models swappable, your data on your disk | A local 2B model by default, any OpenAI-compatible endpoint as an optional brain or vision model |

## Principles

These decide trade-offs, and they are why the architecture looks the way it does ([architecture](architecture.md)).

1. **Trust is the product.** An assistant that sometimes does the wrong irreversible thing is worse than none. Every
   action has a policy (at once, ask, click only), the user can see the exact call, and the gate is code, not a
   prompt. A feature that cannot be made safe stays off.
2. **Models propose, vaice disposes.** The model decides what to try; vaice decides whether and how it runs, enforces
   timeouts and cancellation, and keeps untrusted text (files, web pages, tool output) away from instruction-bearing
   roles.
3. **Spoken, not dictated.** The voice says one or two sentences; the screen holds the paths, output and commands.
4. **Local first, remote by choice.** The default needs no account and sends nothing out. A stronger helper model, a
   vision model or a cloud MCP server is a choice you make, and startup says what it will receive.
5. **Measure, don't assume.** Claims about what a model does come from the live eval with repeats, per build and
   quantization. A web-search skill was removed, and the shell and memory skills kept off by default, after the eval showed they
   made a small model worse; that is the process working.
6. **Small, replaceable parts.** Ears, brain, hands and mouth meet at text, so a better model drops in without a rewrite.
7. **Opt-in power, honest limits.** Each capability documents what it can do, what it sends where, and what is not verified.

## Where we are

The roadmap ([#87](https://github.com/ranacseruet/vaice/issues/87)) was built in phases, all landed:

| Phase | What it added |
|---|---|
| 0 | A live eval harness, and a measured comparison of the interleaved and text-mode paths |
| 1 | `vaice up`: one command starts the model server and the UI |
| 2 | Assistant mode: skills run by vaice, not the model server, read-only first |
| 3 | Action skills, the confirmation policy, and opt-in shell commands |
| 4 | An optional stronger helper brain (any OpenAI-compatible endpoint) and spoken progress for slow answers |
| 5 | Long-term memory, screen vision, and MCP tools |

On the local 2B model the assistant routes about 225 of 255 scored requests correctly across 51 spoken cases (five
runs each); a 24B helper model does better, and the cases where the small model over-calls tools are why the shell,
memory and web search are off or absent by default.

## What is missing

Roughly in the order it limits the north star. These are directions to refine in issues, not committed plans.

- **A better always-available form factor.** Today you open a browser tab and wear headphones. A menu-bar app or a
  global hotkey would make "just ask" real, and how it is invoked day to day is still open. Speaker echo (the
  microphone hearing the assistant, [#36](https://github.com/ranacseruet/vaice/issues/36)) must be solved first for
  anything but headphones.
- **A smarter local brain.** A 2B model picks skills reasonably but writes poor shell commands and over-calls tools. The
  helper-model seam exists; the open work is a local model that is good enough that the helper is optional.
- **More to do.** Skills for the things people actually ask: calendar and mail, files, notes, the browser. Much of this
  can arrive as MCP tools rather than as built-ins. Reminders and timers need something that runs later, which vaice
  does not have yet (it refuses to pretend).
- **Multi-step tasks.** Today one reply is at most a few skill calls. A real assistant plans, carries on across turns,
  recovers from a failed step, and reports at the end.
- **Context without ceremony.** Memory is explicit notes only; screen vision is a screenshot on request. More ambient,
  proactive context is valuable and is a privacy question first.
- **Beyond one Mac.** Skills assume macOS today; other platforms and a second device are open.

## Not goals (for now)

- A hosted, multi-user service. `vaice` is one person's assistant on their own computer.
- Autonomy without consent. Anything that changes the machine asks, by design, however good the models get.
- Replacing the model providers. `vaice` runs models; it does not train or bundle them.

## How we will know it is working

- **Latency:** time to first audio under about a second for a plain answer, and spoken progress when a task is slow.
- **Accuracy:** the live eval's per-case pass rate on the assistant path, tracked per model and build, with no
  regression when a skill is added.
- **Safety:** no skill that can change the machine runs without its policy, and every refusal and failure is spoken, not
  silent. Held by unit tests.
- **Usefulness:** the one that matters and the hardest to measure: does it save you a trip to the keyboard often
  enough that you keep it running?
