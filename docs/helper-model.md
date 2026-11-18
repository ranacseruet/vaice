# Helper model

The 2B model routes questions to the right skill well, but it writes poor shell commands, sometimes calls a
skill when it should just answer, and sometimes answers past the result. A **helper model** takes over only
the brain (step 2 of [assistant mode](assistant-mode.md)): any OpenAI-compatible chat-completions endpoint, whether a server on this machine
(LM Studio, Ollama, another `llama-server`) or a hosted API. vaice does not tell them apart; it posts to
`<VAICE_BRAIN_URL>/chat/completions` and reads normal `tool_calls`. Speech recognition and speech output stay on
the local `llama-server`.

```bash
# A server on this machine (LM Studio; Ollama is http://localhost:11434/v1), no key needed
VAICE_MODE=assistant VAICE_BRAIN_URL=http://localhost:1234/v1 VAICE_BRAIN_MODEL=<model id> .venv/bin/vaice

# A hosted endpoint: keep the key in .env or the environment (VAICE_BRAIN_API_KEY=your-key-here)
VAICE_MODE=assistant VAICE_BRAIN_URL=https://openrouter.ai/api/v1 VAICE_BRAIN_MODEL=<model id> .venv/bin/vaice
```

- **What is sent to the endpoint**, on every brain call: the transcript of what you say, the earlier turns of
  the conversation (your words, the spoken replies, and earlier skill results), the system prompt, the
  current date and time, the skills' definitions, and the results of the skills it asks for: file names,
  process names, the computer's name (`system_info`), and, with `VAICE_ASSISTANT_SHELL` on, the output of any
  command you approved. **Audio is never sent.** Startup logs the endpoint's host and model, and a warning
  when the host is not `localhost`, `127.0.0.1` or `::1`.
- **The request holds only `messages`, `tools` and `model`.** vaice sends no `temperature` and no token cap to
  a helper: hosted models differ in which they accept, and a cap would cut a reasoning model's answer short.
  Its replies may carry a separate `reasoning_content`; that is never spoken and never sent back.
- **Latency is seconds per decision**: a question that needs a skill is two brain calls. A 24B model on this
  machine took 1.2 to 5.6 s per call in a live check.
- **Failures are spoken** (and shown in the chatbot as `(error: …)`): a refused key, a busy endpoint (HTTP
  429), an unreachable or too slow one each get a short fixed sentence. After you approved an action, a brain
  failure does not hide that it ran: vaice speaks the skill's own result. Nothing checks the endpoint at
  startup; an unreachable helper is found out at the first question.
- Results of the live eval vary from run to run without temperature 0, so measure a helper with repeats
  (`--brain-url`, `--brain-model`; see `evals/README.md`).
