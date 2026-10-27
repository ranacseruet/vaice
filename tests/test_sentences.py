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


def test_abbreviations_do_not_end_a_sentence():
    out = list(split_sentences(["Ask Dr. Smith, e.g.", " on Monday. Thanks."]))
    assert out == ["Ask Dr. Smith, e.g. on Monday.", "Thanks."]


def test_list_items_are_split_on_lines_not_after_their_numbers():
    out = list(split_sentences(["Steps:\n1. Boil", " water\n2. Add tea"]))
    assert out == ["Steps:", "1. Boil water", "2. Add tea"]


def test_numbers_ending_a_sentence_still_split():
    assert list(split_sentences(["I am 30. What", " about you?"])) == ["I am 30.", "What about you?"]


def test_clean_for_speech_strips_bullets_links_and_strikethrough():
    assert clean_for_speech("- See [the docs](http://x.y) ~~now~~, well-known") == "See the docs now, well-known"


def test_etc_ends_a_sentence_only_before_a_capital():
    assert list(split_sentences(["I like dogs, etc. What about you?"])) == ["I like dogs, etc.", "What about you?"]
    assert list(split_sentences(["Bring snacks, etc. and drinks."])) == ["Bring snacks, etc. and drinks."]


def test_etc_waits_for_the_next_fragment_to_decide():
    assert list(split_sentences(["I like dogs, etc. ", "What about you?"])) == ["I like dogs, etc.", "What about you?"]


def test_initials_do_not_end_a_sentence():
    assert list(split_sentences(["The U.S. Army came. Yes."])) == ["The U.S. Army came.", "Yes."]


def test_a_number_is_never_spoken_on_its_own():
    assert list(split_sentences(["See fig. 3. Then go home."])) == ["See fig.", "3. Then go home."]


def test_am_pm_ends_a_sentence_before_a_capital():
    assert list(split_sentences(["Let's meet at 3 p.m. Tomorrow is Tuesday."])) == [
        "Let's meet at 3 p.m.",
        "Tomorrow is Tuesday.",
    ]
    assert list(split_sentences(["It is 5 a.m. What time is it?"])) == ["It is 5 a.m.", "What time is it?"]


def test_am_pm_does_not_end_a_sentence_before_a_lower_case_word():
    assert list(split_sentences(["The flight leaves at 7 a.m. and arrives at 9 p.m."])) == [
        "The flight leaves at 7 a.m. and arrives at 9 p.m."
    ]
    # the lower-case word can also arrive later: the deferred decision covers that too
    assert list(split_sentences(["Meet at 3 p.m. ", "and then leave."])) == ["Meet at 3 p.m. and then leave."]


def test_am_pm_waits_for_the_next_fragment_to_decide():
    assert list(split_sentences(["Let's meet at 3 p.m. ", "Tomorrow is Tuesday."])) == [
        "Let's meet at 3 p.m.",
        "Tomorrow is Tuesday.",
    ]


def test_clean_for_speech_folds_line_breaks():
    assert clean_for_speech("1.\nApples") == "1. Apples"
