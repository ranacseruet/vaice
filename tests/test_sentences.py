from vaice.sentences import clean_for_speech, split_sentences


def test_splits_on_sentence_boundaries_across_fragments():
    out = list(split_sentences(["Hello", " there.", " How", " are you?", " Fine"]))
    assert out == ["Hello there.", "How are you?", "Fine"]


def test_does_not_split_decimals():
    assert list(split_sentences(["Pi is 3", ".14 roughly."])) == ["Pi is 3.14 roughly."]


def test_empty_stream_yields_only_empty_remainder():
    assert [s for s in split_sentences([]) if s] == []


def test_clean_for_speech_strips_markdown():
    assert clean_for_speech("**Hi** _there_ `x`") == "Hi there x"
