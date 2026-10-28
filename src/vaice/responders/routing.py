"""Capability-based selection for the verified responder paths."""
from dataclasses import dataclass


@dataclass(frozen=True)
class ModelCapabilities:
    """The modalities a server explicitly advertises for one model.

    ``None`` means the server did not provide a usable value. It is deliberately distinct from an
    empty set: unknown metadata is not evidence that a modality is unsupported.
    """

    model_id: str | None
    input_modalities: frozenset[str] | None
    output_modalities: frozenset[str] | None


def resolve_pipeline(
    requested: str,
    capabilities: ModelCapabilities | None,
    *,
    realtime_protocol: bool = False,
) -> str:
    """Resolve a configured pipeline without inferring unsupported adapters.

    An unavailable server preserves the configured explicit path. Unknown capabilities do not
    become unsupported claims or route guesses. Auto mode selects realtime only when both audio
    directions and the websocket protocol are verified; text mode stays explicit because model
    metadata does not prove chat endpoint support.
    """
    if requested not in {"realtime", "text", "auto"}:
        raise ValueError("pipeline must be realtime, text, or auto")

    if requested != "auto":
        if capabilities is not None:
            if (
                requested == "text"
                and capabilities.input_modalities is not None
                and "text" not in capabilities.input_modalities
            ):
                raise RuntimeError(f"model {capabilities.model_id or '(unknown)'} does not accept text input")
            if capabilities.output_modalities is not None:
                required = "audio" if requested == "realtime" else "text"
                if required not in capabilities.output_modalities:
                    raise RuntimeError(
                        f"model {capabilities.model_id or '(unknown)'} does not output {required}; "
                        f"it advertises {sorted(capabilities.output_modalities)}"
                    )
            if (
                requested == "realtime"
                and capabilities.input_modalities is not None
                and "audio" not in capabilities.input_modalities
            ):
                raise RuntimeError(
                    f"model {capabilities.model_id or '(unknown)'} does not accept input audio; "
                    f"it advertises {sorted(capabilities.input_modalities)}"
                )
        return requested

    if capabilities is None:  # server down or still loading: auto cannot select without evidence
        raise RuntimeError(
            "llama-server is unavailable or still loading, so auto mode cannot verify a pipeline; "
            "set VAICE_PIPELINE=realtime or VAICE_PIPELINE=text to select a configured adapter explicitly"
        )
    inputs = capabilities.input_modalities
    outputs = capabilities.output_modalities
    if inputs is None or outputs is None:
        raise RuntimeError(
            "model capabilities are unknown; set VAICE_PIPELINE=realtime or VAICE_PIPELINE=text "
            "to select a configured adapter explicitly"
        )
    if "audio" in inputs and "audio" in outputs:
        if realtime_protocol:
            return "realtime"
        raise RuntimeError(
            "the model advertises input and output audio, but the /v1/realtime protocol could not be verified; "
            "set VAICE_PIPELINE=realtime to select that adapter explicitly"
        )
    if "audio" in inputs or "audio" in outputs:
        raise RuntimeError(
            "the realtime adapter requires verified input and output audio modalities; "
            "set VAICE_PIPELINE=realtime or VAICE_PIPELINE=text to select a configured adapter explicitly"
        )
    if "text" in inputs and "text" in outputs:
        raise RuntimeError(
            "the model advertises text input and output, but model metadata does not prove that an "
            "OpenAI-compatible chat endpoint is available; set VAICE_PIPELINE=text to select that adapter explicitly"
        )
    raise RuntimeError(
        f"no verified pipeline matches model {capabilities.model_id or '(unknown)'} "
        f"(input={sorted(inputs)}, output={sorted(outputs)}); set VAICE_PIPELINE=realtime or "
        "VAICE_PIPELINE=text only if the configured adapter supports it"
    )
