# vaice documentation

Start with the [project README](../README.md) for what `vaice` is and a five-minute quickstart. This directory holds
everything else.

## Where it is going, and how it is built

| Page | What it covers |
|---|---|
| [Vision](vision.md) | The north star (a Jarvis-style assistant that gets things done on your computer), the principles, where the project is today and what comes next |
| [Architecture](architecture.md) | The processes, how a reply works, the safety model, threading, state on disk, and the code map |

## Setting up and running

| Page | What it covers |
|---|---|
| [Installation](installation.md) | Install `llama-server`, run `vaice up`, talk in the browser; requirements and compatibility |
| [Configuration](configuration.md) | Every `VAICE_*` environment variable, with defaults |
| [Conversation](conversation.md) | Startup and shutdown, how history is kept, barge-in |
| [Troubleshooting](troubleshooting.md) | Off-topic or echoing replies, a server without `/v1/realtime`, and other known problems |
| [Privacy](privacy.md) | What is stored on disk and what can leave your machine |

## Using it

| Page | What it covers |
|---|---|
| [The assistant](assistant-mode.md) | Skills, confirmation, shell commands and memory |
| [Screen vision](screen-vision.md) | The `look_at_screen` skill and its vision model |
| [MCP tools](mcp-tools.md) | Offer tools from MCP servers as skills, with a per-tool approval policy |
| [Browser tasks](browser-tasks.md) | The `browse` skill: a Chrome window driven by the Pi agent in the background, asking before it changes a page |

## Working on vaice

| Page | What it covers |
|---|---|
| [Development](development.md) | Dependency updates, the test suite, the live eval and code quality |
| [CONTRIBUTING.md](../CONTRIBUTING.md) | Setup, pull request flow, keeping docs in sync |
| [AGENTS.md](../AGENTS.md) | The design decisions and gotchas for coding agents and new contributors |
| [evals/README.md](../evals/README.md) | The live tool-calling eval |
| [CHANGELOG.md](../CHANGELOG.md) | What changed and when |

## Keeping the docs in sync

A change to behavior, a setting or a contract updates the page that describes it in the same pull request:
[Configuration](configuration.md) for a `VAICE_*` variable (and `.env.example`), [Architecture](architecture.md)
for a change to how the pieces fit or to the code layout, and [Vision](vision.md) when a roadmap item lands or the
direction changes.
