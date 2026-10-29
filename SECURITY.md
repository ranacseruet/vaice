# Security Policy

## Supported Versions

| Version | Supported |
|---|---|
| `0.1.x` (`main`) | Yes |

## Reporting a Vulnerability

Please **do not** report security vulnerabilities through public GitHub issues.

Instead, please report vulnerabilities privately through [GitHub Private Vulnerability Reporting](https://github.com/ranacseruet/vaice/security/advisories/new) on this repository.

When reporting a vulnerability, please include:
- A description of the issue and its potential impact
- Steps or a minimal reproduction to trigger the vulnerability
- Affected versions and configuration (`VAICE_*` settings, Python version, OS)

We will review reports promptly and coordinate a fix before public disclosure.

## Operational & Local Security Notes

- **Local-only binding by default:** `vaice` launches Gradio on `127.0.0.1:7860` (`VAICE_PORT`), and connects to `llama-server` on `http://localhost:8001` (`VAICE_LLAMACPP_URL`). Neither service is authenticated by default; do not expose `vaice` or `llama-server` directly to untrusted networks without TLS and authentication in front.
- **Plain-text transcript rendering:** `vaice` explicitly sets `gr.Chatbot(..., allow_tags=False)` so model-generated text is never rendered as raw HTML/JavaScript in the browser UI.
- **Local conversation storage:** Conversation transcripts are persisted as plain JSON in `~/.vaice/conversation.json` (or `VAICE_DATA_DIR`).
