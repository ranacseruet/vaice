import json
import logging
import os
import threading

import pytest

from vaice import conversation as conv
from vaice.conversation import INAUDIBLE, Conversation, Turn, window_start


@pytest.fixture(autouse=True)
def short_wait(monkeypatch):
    monkeypatch.setattr(conv, "TRANSCRIPT_WAIT", 0.05)


def exchange(c, user, assistant=None, finish=True):
    turn = c.start_turn()
    turn.set_user_text(user)
    if assistant is not None:
        turn.set_assistant_text(assistant)
    if finish:
        turn.finish()
    return turn


def test_starting_a_turn_interrupts_an_unfinished_previous_one():
    c = Conversation()
    first = c.start_turn()
    first.set_assistant_text("Paris is")
    second = c.start_turn()

    assert first.interrupted and first.closed.is_set() and not first.complete
    assert not second.closed.is_set()


def test_a_finished_turn_is_not_marked_interrupted():
    c = Conversation()
    first = exchange(c, "hi", "hello")
    c.start_turn()

    assert first.complete and not first.interrupted


def test_late_assistant_text_on_a_closed_turn_is_ignored_but_the_transcript_is_not():
    c = Conversation()
    first = c.start_turn()
    first.set_assistant_text("Paris is")
    c.start_turn()

    first.set_assistant_text("Paris is the capital of France.")  # a dropped generator finishing late
    first.set_user_text("capital of France?")
    first.finish()  # ...and so is a late finish

    assert first.assistant_text == "Paris is"
    assert first.user_text == "capital of France?"
    assert first.interrupted and not first.complete


def test_standalone_turn_works_without_a_conversation():
    turn = Turn()
    turn.set_assistant_text("hi")
    turn.set_user_text("hello")
    turn.finish()
    assert (turn.user_text, turn.assistant_text, turn.complete) == ("hello", "hi", True)
    assert turn.closed.is_set() and turn.transcribed.is_set()


@pytest.mark.parametrize(
    "count, window, expected",
    [(0, 10, 0), (10, 10, 0), (11, 10, 5), (15, 10, 5), (16, 10, 10), (21, 10, 15), (3, 1, 2), (7, 0, 7), (5, 3, 2), (9, 4, 6)],
)
def test_window_start_drops_whole_blocks_of_the_oldest_turns(count, window, expected):
    assert window_start(count, window) == expected


def test_window_start_never_keeps_more_than_the_window_and_moves_only_every_block():
    window = 10
    starts = [window_start(n, window) for n in range(40)]
    assert all(n - s <= window for n, s in zip(range(40), starts))
    # stable prefix: between drops the start index does not move
    assert starts[11:16] == [5] * 5
    assert starts[16:21] == [10] * 5
    assert starts == sorted(starts) and all(s % (window // 2) == 0 for s in starts)


def test_history_is_windowed_and_the_start_is_stable_between_drops():
    c = Conversation(window=4)  # blocks of 2
    starts = []
    for i in range(8):
        turn = exchange(c, f"q{i}", f"a{i}")
        probe = c.start_turn()
        history = c.history(before=probe)
        starts.append(history[0][1])
        c._turns.remove(probe)  # only looking at the history, not extending the conversation
    # q0..q3 fit; then the start moves by whole blocks of two turns
    assert starts == ["q0", "q0", "q0", "q0", "q2", "q2", "q4", "q4"]


def test_window_zero_keeps_no_history():
    c = Conversation(window=0)
    exchange(c, "q", "a")
    assert c.history(before=c.start_turn()) == []


def test_history_marks_interrupted_replies_and_skips_empty_ones():
    c = Conversation()
    exchange(c, "one", "partial")  # finished normally...
    c._turns[-1].interrupted = True  # ...but stored as interrupted
    exchange(c, "two", "")  # nothing was handed out before the interrupt
    exchange(c, "three", "done")

    assert c.history(before=c.start_turn()) == [
        ("user", "one"),
        ("assistant", "partial (interrupted)"),
        ("user", "two"),
        ("user", "three"),
        ("assistant", "done"),
    ]


def test_history_only_includes_turns_before_the_given_one():
    c = Conversation()
    first = exchange(c, "one", "a")
    second = c.start_turn()
    c.start_turn()
    assert c.history(before=first) == []
    assert c.history(before=second) == [("user", "one"), ("assistant", "a")]


def test_history_uses_a_transcript_that_arrives_late(monkeypatch):
    monkeypatch.setattr(conv, "TRANSCRIPT_WAIT", 5)
    c = Conversation()
    first = c.start_turn()
    first.set_assistant_text("answer")
    second = c.start_turn()
    threading.Timer(0.1, first.set_user_text, args=["late question"]).start()

    assert c.history(before=second) == [("user", "late question"), ("assistant", "answer (interrupted)")]


def test_history_falls_back_to_inaudible_when_the_transcript_never_arrives():
    c = Conversation()
    first = c.start_turn()
    first.set_assistant_text("answer")
    assert c.history(before=c.start_turn())[0] == ("user", INAUDIBLE)
    assert exchange(c, "")  # an empty transcript is inaudible too
    assert c.history(before=c.start_turn())[-1] == ("user", INAUDIBLE)


def test_messages_for_ui_include_pending_turns():
    c = Conversation()
    exchange(c, "hello", "hi there")
    pending = c.start_turn()
    assert c.messages_for_ui() == [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi there"},
        {"role": "user", "content": "…"},
    ]
    pending.set_user_text("again")
    pending.set_assistant_text("yes")
    assert c.messages_for_ui()[-2:] == [{"role": "user", "content": "again"}, {"role": "assistant", "content": "yes"}]


def test_persistence_round_trip(tmp_path):
    path = tmp_path / "nested" / "conversation.json"
    c = Conversation(path)
    exchange(c, "q1", "a1")
    exchange(c, "q2", "a2 partial", finish=False)
    c.start_turn()  # interrupts the unfinished one

    loaded = Conversation(path)

    assert len(loaded) == 3
    assert json.loads(path.read_text())["version"] == 1
    assert loaded.history(before=loaded.start_turn()) == [
        ("user", "q1"),
        ("assistant", "a1"),
        ("user", "q2"),
        ("assistant", "a2 partial (interrupted)"),
        ("user", INAUDIBLE),  # the third turn was never transcribed
    ]


def test_loaded_turn_without_transcript_gets_empty_text_and_is_complete(tmp_path):
    path = tmp_path / "c.json"
    path.write_text(json.dumps({"version": 1, "turns": [{"user": None, "assistant": "x", "interrupted": False, "complete": False}]}))

    (turn,) = Conversation(path)._turns

    assert turn.user_text == "" and turn.complete and turn.closed.is_set() and turn.interrupted


def test_every_change_is_saved_with_no_temp_files_left(tmp_path):
    path = tmp_path / "c.json"
    c = Conversation(path)
    turn = c.start_turn()
    assert json.loads(path.read_text())["turns"][0]["user"] is None
    turn.set_user_text("q")
    turn.set_assistant_text("a")
    turn.finish()

    assert json.loads(path.read_text())["turns"] == [{"user": "q", "assistant": "a", "interrupted": False, "complete": True}]
    assert os.listdir(tmp_path) == ["c.json"]


def test_a_failed_write_leaves_the_previous_file_intact(tmp_path, monkeypatch):
    path = tmp_path / "c.json"
    c = Conversation(path)
    exchange(c, "q", "a")
    before = path.read_text()

    def boom(src, dst):
        raise OSError("disk full")

    monkeypatch.setattr(conv.os, "replace", boom)
    with pytest.raises(OSError, match="disk full"):
        c.start_turn()

    assert path.read_text() == before
    assert os.listdir(tmp_path) == ["c.json"]


@pytest.mark.parametrize("content", ["{not json", '{"version": 1}', '{"version": 2, "turns": []}', '{"version": 1, "turns": [1]}'])
def test_corrupt_file_is_moved_aside_with_a_warning(tmp_path, caplog, content):
    path = tmp_path / "conversation.json"
    path.write_text(content)

    with caplog.at_level(logging.WARNING, logger=conv.__name__):
        c = Conversation(path)

    assert len(c) == 0
    assert not path.exists()
    assert (tmp_path / "conversation.json.corrupt").read_text() == content
    assert str(path) in caplog.text and "conversation.json.corrupt" in caplog.text
    exchange(c, "fresh", "start")  # and a new file is written
    assert len(Conversation(path)) == 1


def test_missing_file_starts_empty_without_warning(tmp_path, caplog):
    with caplog.at_level(logging.WARNING, logger=conv.__name__):
        assert len(Conversation(tmp_path / "nope.json")) == 0
    assert caplog.text == ""


def test_reset_clears_and_persists_and_detaches_an_in_flight_turn(tmp_path):
    path = tmp_path / "c.json"
    c = Conversation(path)
    exchange(c, "old", "reply")
    in_flight = c.start_turn()

    c.reset()
    in_flight.set_user_text("late transcript")
    in_flight.set_assistant_text("late reply")
    in_flight.finish()

    assert c.messages_for_ui() == []
    assert json.loads(path.read_text())["turns"] == []
    assert len(Conversation(path)) == 0
    fresh = c.start_turn()
    assert c.history(before=fresh) == []
    assert in_flight.closed.is_set()
