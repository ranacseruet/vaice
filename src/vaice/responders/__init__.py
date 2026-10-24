"""Responder backends and the factory that picks one."""
import logging

from vaice import config
from vaice.responders.base import AudioFrame, Responder

log = logging.getLogger(__name__)

BACKENDS = ("auto", "cascade", "native")


def _build_cascade() -> Responder:
    from vaice import tts
    from vaice.responders.cascade import CascadeResponder

    tts.load_model()  # load once at startup so the first reply isn't slow
    return CascadeResponder()


def _build_native() -> Responder:
    if not config.NATIVE_MODEL:
        raise RuntimeError("no native model configured (VAICE_NATIVE_MODEL is empty)")
    from vaice.responders.native import NativeResponder

    return NativeResponder.load(config.NATIVE_MODEL)


def create_responder(backend: str = config.BACKEND) -> Responder:
    """`auto` prefers the native speech-to-speech model and falls back to the cascade."""
    if backend not in BACKENDS:
        raise ValueError(f"unknown backend {backend!r}; expected one of {BACKENDS}")
    if backend in ("auto", "native"):
        try:
            return _build_native()
        except Exception as exc:
            if backend == "native":
                raise
            log.warning("native speech backend unavailable (%s: %s); falling back to cascade", type(exc).__name__, exc)
    return _build_cascade()


__all__ = ["AudioFrame", "Responder", "create_responder"]
