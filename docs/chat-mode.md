# Chat mode

Chat mode is the default (`VAICE_MODE=chat`). One speech-to-speech model hears your audio and streams its spoken
reply back over `llama-server`'s `/v1/realtime` WebSocket, with no separate transcription, brain or voice step.
It is the fastest path to the first word of a reply and the right choice for plain conversation. To get things
done on your computer, use [assistant mode](assistant-mode.md) instead.

```
browser mic ─WebRTC─▶ fastrtc (VAD: detects you stopped talking)
                         │  audio
                         ▼
                  LlamaCppResponder  llama-server :8001 /v1/realtime (WebSocket): speech streams back;
                   │                  a second session transcribes your audio in parallel
                   │ records the transcript and spoken text on `turn`
                   ▼
                  Conversation (~/.vaice/conversation.json) ──▶ transcript in the UI
```

The conversation, history windowing and barge-in are described in [Conversation](conversation.md). The one extra
chat mode has is the model's own tools, below.

## Tools

`vaice` can speak the answers of llama-server's **built-in tools** through the voice path. Enable the
tools on the server (read-only subset recommended — they run with the server's own permissions, so
never enable `exec_shell_command` or `write_file` for a voice demo):

```bash
llama-server -hf LiquidAI/LFM3-Omni-2B-GGUF:Q8_0 --temp 0 --tools get_info,read_file
```

then start `vaice` with `VAICE_LLAMACPP_TOOLS=builtin`. vaice takes **every tool the server
advertises** — the set of tools is the server operator's decision, made once on the
`llama-server` command line (`--tools get_info,read_file`), and it is also the gate: the server
refuses to execute anything it did not advertise. If the server advertises a tool marked
`permissions.write` (`exec_shell_command`, `write_file`, `edit_file` in llama-server's built-ins),
vaice warns at startup. Tool calls run with an overall bound of `VAICE_LLAMACPP_TIMEOUT` (120 s by
default); when it passes, or when the reply is stopped (barge-in, **New conversation**, Ctrl+C),
`vaice` stops waiting for the call and discards its result — it does **not** cancel the HTTP
request, so the server may still run (and finish) the tool, which is one more reason to keep the
set read-only. Stopping returns the session to listening instead of stalling the thread for the
full timeout. The definitions come from `GET /tools`, are declared in each realtime
`session.update`, and a spoken question like *"What operating system is this?"* makes the model
emit a function call; `vaice` executes it through the server's `POST /tools` and feeds the result
back, and the model speaks the answer. A call for a tool the server did not advertise is refused
by the server and spoken as a failure.

The first response is requested with `tool_choice: "required"`; after a tool result vaice asks
for the next response with `auto` (the model may answer or chain one more call), and after the
second tool round it asks for the final answer with `none`. That closes a live failure mode: a
follow-up that inherited `required` was forced to call again, so it repeated the raw result and
could loop until the token cap instead of answering. A call that still arrives in the forced
final round is not executed.

Only the answer the model gives **after** a tool result is recorded in the conversation: whatever
it says before the call (the working-out-loud preamble) is spoken, but it is not written to
history or shown in the transcript, so a wrong first pass cannot reinforce itself in the next
turn's prompt.

LFM3-Audio notes from live testing: **tell the model to use the tools** — with the default prompt
("Reply naturally and concisely…") the model announces a check and then stops without calling in
most runs; a prompt like `Use the available tools to answer questions about this system.` made it
call and speak the real answer in 4 of 4 runs (vs 1 of 6 with the default). Keep the tool subset
small (one or two; with three or more tools it tends to answer from its head and get it wrong).
`tool_choice: "required"` is reliable in text-only mode, but it is not enforced with interleaved
text+audio: with the voice output the model can answer without calling, so in practice the prompt
and the tool subset decide whether it calls.

The model **reads the fed-back tool result aloud** (a 2B voice model limitation; prompt lines
forbidding it have no measurable effect), so `vaice` compacts every result before feeding it back:
JSON becomes short `key: value` pairs, values are truncated at 200 characters and the whole output
at 800 by default (`VAICE_LLAMACPP_TOOL_VALUE_CHARS`, `VAICE_LLAMACPP_TOOL_OUTPUT_CHARS`). A
`read_file` payload is plain text, so it is bounded by the 800-character output cap only — a
210-character file lost its last word when the 200-character per-value cap applied to it. The
caps have to be large enough to keep the answer: at 40/120 a `get_info` result lost the timestamp
before the model could speak it, and a "what time" question became "I can't tell the time". Larger
caps mean more dictation; the defaults keep the answer while holding the result to a couple of
sentences. The transcript can also show the raw result and stray tags around a tool round; whether
the audio includes them varies, so keep the caps conservative.

Tool calls and their results are kept **out of the conversation**: the chatbot shows only what was
said, and each call lands in the collapsed *Tool activity* panel under it (name, arguments, result;
the last 10 across the conversation, each result capped at 800 characters with a `... (truncated)`
marker). The transcript that is recorded and shown is also cleaned of the model's scaffolding
(`<tool_call>…</tool_call>` blocks, `</think` fragments, stray `<|…|>` control tokens). There is
no weather tool in llama-server — a question it cannot answer with the configured tools is refused
(or, with `exec_shell_command` enabled, the model improvises a shell command, which is one more
reason to keep that tool out).
