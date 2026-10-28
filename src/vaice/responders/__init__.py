"""Responder backends and the factory that picks one."""
import logging

from vaice import config
from vaice.responders.base import AudioFrame, Responder

log = logging.getLogger(__name__)

BACKENDS = ("auto", "llamacpp", "native")
# Old names that still work, wherever a backend is named (--backend, VAICE_BACKEND)
BACKEND_ALIASES = {"cascade": "llamacpp"}
BACKEND_CHOICES = (*BACKENDS, *BACKEND_ALIASES)


def _build_llamacpp(*, strict: bool) -> Responder:
    from vaice.responders.llamacpp import LlamaCppResponder, check_server

    check_server(strict=strict)
    return LlamaCppResponder()


def _build_native() -> Responder:
    if not config.NATIVE_MODEL:
        raise RuntimeError("no native model configured (VAICE_NATIVE_MODEL is empty)")
    from vaice.responders.native import NativeResponder

    return NativeResponder.load(config.NATIVE_MODEL)


def create_responder(backend: str = config.BACKEND) -> Responder:
    """`auto` prefers llama-server's own speech and falls back to the native speech-to-speech model."""
    backend = BACKEND_ALIASES.get(backend, backend)
    if backend not in BACKENDS:
        raise ValueError(f"unknown backend {backend!r}; expected one of {BACKEND_CHOICES}")
    if backend == "native":
        return _build_native()
    if backend == "llamacpp":
        return _build_llamacpp(strict=False)
    try:
        return _build_llamacpp(strict=True)
    except Exception as exc:
        # Traceback included: this also catches bugs and API changes, not just "no server here"
        log.warning(
            "llama.cpp backend unavailable (%s: %s); falling back to native",
            type(exc).__name__,
            exc,
            exc_info=True,
        )
    return _build_native()


__all__ = ["AudioFrame", "Responder", "create_responder"]
