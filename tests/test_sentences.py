from vaice import sentences
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


def test_every_fragment_cut_gives_the_same_split():
    # the scanner resumes where it left off; only a deferred decision (a.m./etc. at the buffer end)
    # is re-checked, so the split must not depend on where the fragments are cut
    whole = "Call me at 7 a.m. Maybe. Dogs, etc. Fine. The U.S. is big. I said 3. It counts. Done"
    as_one = list(split_sentences([whole]))
    for cut in range(1, len(whole)):
        assert list(split_sentences([whole[:cut], whole[cut:]])) == as_one, f"cut at {cut}"


def test_a_long_stream_keeps_finding_every_boundary():
    # the resume must skip decided text, not boundaries: 400 sentences word by word, none lost
    reply = "".join(f"Sentence {i} looks fine. " for i in range(400))
    fragments = [word + " " for word in reply.split()]  # the space rides with the word, as in a real stream
    out = [s for s in split_sentences(fragments) if s]
    assert len(out) == 400
    assert out[0] == "Sentence 0 looks fine." and out[-1] == "Sentence 399 looks fine."


class CountingEnd:
    """The sentence-end pattern, counting the characters each finditer call scans."""

    def __init__(self, pattern):
        self.pattern = pattern
        self.scanned = 0

    def finditer(self, text, pos=0):
        self.scanned += len(text) - pos
        return self.pattern.finditer(text, pos)

    def search(self, text):
        return self.pattern.search(text)


def test_the_resume_scans_only_the_new_fragment(monkeypatch):
    # guard for the resume: reverting scan_from to 0 re-scans the whole buffer per fragment, which
    # this counts; wall-clock timing would be flaky, a counted span is deterministic
    reply = "".join(f"Sentence {i} looks fine. " for i in range(400))
    fragments = [word + " " for word in reply.split()]
    counting = CountingEnd(sentences._SENTENCE_END)
    monkeypatch.setattr(sentences, "_SENTENCE_END", counting)
    out = [s for s in sentences.split_sentences(fragments) if s]
    assert len(out) == 400
    assert counting.scanned <= 2 * len(reply)  # resume scans each new fragment, plus a deferred boundary


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
