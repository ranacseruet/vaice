# vaice

[![CI](https://github.com/ranacseruet/vaice/actions/workflows/ci.yml/badge.svg)](https://github.com/ranacseruet/vaice/actions/workflows/ci.yml)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Talk to your computer and get things done.** `vaice` is a voice assistant that runs in your browser on local
models: you speak, it answers out loud as it thinks, and it can check on your Mac, open apps, change the volume and
more, asking before it does anything it cannot undo. The goal is a real-life Jarvis, the voice-controlled AI from *Iron Man* (see the [vision](docs/vision.md) for what that means here).

> **Status:** a single-user demo in active development, tested on macOS with a 2B speech model. It is useful today
> for questions and a handful of actions, and it is not yet an assistant you would trust with your whole computer.

## What it does

- **Natural conversation.** Streaming spoken replies, barge-in (talk over it and it stops), and a conversation that
  carries over between questions and survives a restart.
- **Gets things done:** disk space, battery, running processes, finding files, opening and quitting apps, volume.
  Anything that cannot be undone waits for you to press **Run**.
- **Opt-in power:** approved shell commands, long-term memory of facts you ask it to keep, looking at your screen,
  tools from any [MCP](https://modelcontextprotocol.io) server, and web
  tasks in a browser window that asks before it changes a page.
- **Local first.** Audio, transcripts and tool results stay on your machine unless you point an optional feature at
  a remote endpoint, and startup warns when you do. See [Privacy](docs/privacy.md).
- **Measured, not assumed.** A live eval harness scores how reliably spoken requests become the right action, and
  prompt and skill changes are checked against it.

## How it works

```
browser mic ─WebRTC─▶ fastrtc (VAD) ─▶ one speech-to-speech session ◀──▶ skills (run in vaice, gated by Run)
                                              │
                                              ▼
                                  streamed speech ─▶ speaker
```

One model hears you and answers out loud. When you ask about your computer, it calls a skill that vaice itself runs
and speaks the result. Details: [architecture](docs/architecture.md).

## Quickstart

You need macOS, [uv](https://docs.astral.sh/uv/), Chrome and headphones (speaker audio feeds back into the
microphone). `vaice` does not bundle a model; it runs [llama.cpp](https://github.com/ggml-org/llama.cpp)
`llama-server` **b11960 or newer** with a speech model, and `vaice up` starts it for you.

```bash
brew install llama.cpp
git clone https://github.com/ranacseruet/vaice.git && cd vaice
uv sync --extra dev
.venv/bin/vaice up        # starts llama-server if needed, then opens http://127.0.0.1:7860
```

The first run downloads about 2.4 GB of model weights (LFM3-Omni-2B, 8-bit; about 4 GB of memory once loaded).
In the browser, allow the microphone, click **Record**, ask a question and pause. Then try "How much battery do I have left?" or "Open Safari". More options (running `llama-server` yourself,
requirements, every setting): [Installation](docs/installation.md) and [Configuration](docs/configuration.md).

## Documentation

Start at the [documentation index](docs/README.md). The main pages:

| | |
|---|---|
| [Vision](docs/vision.md) | Where this is going, and the principles it is built on |
| [Architecture](docs/architecture.md) | How the pieces fit, and the code layout |
| [Installation](docs/installation.md) · [Configuration](docs/configuration.md) | Setup, requirements, every `VAICE_*` setting |
| [The assistant](docs/assistant.md) | Skills, confirmation, shell commands, memory |
| [Screen vision](docs/screen-vision.md) · [MCP tools](docs/mcp-tools.md) · [Browser tasks](docs/browser-tasks.md) | The opt-in extensions |
| [Privacy](docs/privacy.md) · [Troubleshooting](docs/troubleshooting.md) | What leaves your machine; fixing common problems |

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) and [Development](docs/development.md). Coding agents also read
[AGENTS.md](AGENTS.md), which holds the design decisions and the gotchas that have bitten this repo. Security issues:
[SECURITY.md](SECURITY.md).

## License

Licensed under the [MIT License](LICENSE).
