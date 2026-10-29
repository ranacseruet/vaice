from vaice.responders.base import spoken_part


def test_spoken_part_splits_the_text_by_the_share_of_audio():
    assert spoken_part("one two three four", 40, 10) == "one"
    assert spoken_part("one two three four", 40, 20) == "one two"
    assert spoken_part("one two three four", 40, 40) == "one two three four"
    assert spoken_part("one two", 0, 0) == ""  # no audio, nothing said
    # truncation, like Turn.close(): 3.5 words' worth of audio never credits the 4th word
    assert spoken_part("one two three four", 8, 7) == "one two three"
    # ... and like close, never empty: some audio was handed out, so one word stays
    assert spoken_part("one two three", 40, 10) == "one"
    assert spoken_part("one two three", 40, 0) == ""  # no audio handed out at all
