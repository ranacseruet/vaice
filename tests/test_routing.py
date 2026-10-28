import pytest

from vaice.responders.routing import ModelCapabilities, resolve_pipeline


def capabilities(inputs, outputs, model_id="model"):
    return ModelCapabilities(model_id, frozenset(inputs) if inputs is not None else None,
                             frozenset(outputs) if outputs is not None else None)


def test_auto_uses_realtime_only_when_both_audio_modalities_and_protocol_are_verified():
    caps = capabilities(["text", "audio"], ["text", "audio"])
    assert resolve_pipeline("auto", caps, realtime_protocol=True) == "realtime"


def test_auto_does_not_infer_audio_input_from_audio_output():
    caps = capabilities(["text"], ["text", "audio"])
    with pytest.raises(RuntimeError, match="input and output audio"):
        resolve_pipeline("auto", caps, realtime_protocol=True)


def test_auto_requires_a_verified_realtime_protocol():
    caps = capabilities(["text", "audio"], ["text", "audio"])
    with pytest.raises(RuntimeError, match="realtime protocol"):
        resolve_pipeline("auto", caps, realtime_protocol=False)


def test_auto_reports_unknown_instead_of_calling_it_unsupported():
    caps = capabilities(None, None)
    with pytest.raises(RuntimeError, match=r"unknown.*VAICE_PIPELINE=realtime or VAICE_PIPELINE=text"):
        resolve_pipeline("auto", caps)


@pytest.mark.parametrize("override", ["realtime", "text"])
def test_explicit_pipeline_overrides_unknown_capability_metadata(override):
    assert resolve_pipeline(override, capabilities(None, None)) == override


def test_auto_does_not_assume_that_advertised_text_means_openai_chat_is_available():
    caps = capabilities(["text"], ["text"])
    with pytest.raises(RuntimeError, match=r"OpenAI-compatible chat endpoint.*VAICE_PIPELINE=text"):
        resolve_pipeline("auto", caps)


def test_explicit_modes_reject_known_incompatible_modalities():
    with pytest.raises(RuntimeError, match="does not accept text input"):
        resolve_pipeline("text", capabilities(["audio"], ["text"]))
    with pytest.raises(RuntimeError, match="does not output audio"):
        resolve_pipeline("realtime", capabilities(["text", "audio"], ["text"]))


def test_explicit_paths_keep_the_startup_tolerance_when_the_server_is_unavailable():
    assert resolve_pipeline("realtime", None) == "realtime"
    assert resolve_pipeline("text", None) == "text"


def test_auto_reports_an_unavailable_server_without_guessing():
    with pytest.raises(
        RuntimeError,
        match=r"llama-server is unavailable.*set VAICE_PIPELINE=realtime or VAICE_PIPELINE=text",
    ):
        resolve_pipeline("auto", None)


def test_unknown_pipeline_name_is_rejected():
    with pytest.raises(ValueError, match="realtime, text, or auto"):
        resolve_pipeline("hybrid", None)
