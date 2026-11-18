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
    [
        (0, 10, 0), (10, 10, 0), (11, 10, 5), (15, 10, 5), (16, 10, 10),
        (21, 10, 15), (3, 1, 2), (7, 0, 7), (5, 3, 2), (9, 4, 6),
    ],
)
def test_window_start_drops_whole_blocks_of_the_oldest_turns(count, window, expected):
    assert window_start(count, window) == expected


def test_window_start_never_keeps_more_than_the_window_and_moves_only_every_block():
    window = 10
    starts = [window_start(n, window) for n in range(40)]
    assert all(n - s <= window for n, s in zip(range(40), starts, strict=True))
    # stable prefix: between drops the start index does not move
    assert starts[11:16] == [5] * 5
    assert starts[16:21] == [10] * 5
    assert starts == sorted(starts) and all(s % (window // 2) == 0 for s in starts)


def test_history_is_windowed_and_the_start_is_stable_between_drops():
    c = Conversation(window=4)  # blocks of 2
    starts = []
    for i in range(8):
        exchange(c, f"q{i}", f"a{i}")
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


def test_window_zero_does_not_wait_for_a_transcript_it_would_discard():
    c = Conversation(window=0)
    previous = c.start_turn()  # never transcribed

    class RecordingEvent:
        waited = False

        def wait(self, timeout=None):
            self.waited = True
            return False

    previous.transcribed = RecordingEvent()

    assert c.history(before=c.start_turn()) == []
    assert not previous.transcribed.waited


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


def test_history_leaves_a_pending_transcript_out_of_the_prompt():
    c = Conversation()
    pending = c.start_turn()
    pending.set_assistant_text("an answer to a question the model cannot see")

    # The transcript has not arrived yet: the whole exchange is left out, so the prompt
    # never holds a reply to a question that is missing (and never says the user was
    # unintelligible while they are still being transcribed).
    assert c.history(before=c.start_turn()) == []


def test_history_shows_a_transcript_that_arrived_empty_as_inaudible():
    c = Conversation()
    assert exchange(c, "")  # an empty transcript is inaudible
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
    path.write_text(json.dumps(
        {"version": 1, "turns": [{"user": None, "assistant": "x", "interrupted": False, "complete": False}]}
    ))

    (turn,) = Conversation(path)._turns

    assert turn.user_text == "" and turn.complete and turn.closed.is_set() and turn.interrupted


def test_a_saved_error_round_trips_and_older_files_without_it_still_load(tmp_path):
    path = tmp_path / "c.json"
    c = Conversation(path)
    failed = c.start_turn()
    failed.set_user_text("q")
    failed.fail("boom")
    loaded = Conversation(path)

    assert loaded._turns[0].error == "boom"

    path.write_text(json.dumps(
        {"version": 1, "turns": [{"user": "q", "assistant": "a", "interrupted": False, "complete": True}]}
    ))
    (turn,) = Conversation(path)._turns
    assert turn.error == ""


def test_every_change_is_saved_with_no_temp_files_left(tmp_path):
    path = tmp_path / "c.json"
    c = Conversation(path)
    turn = c.start_turn()
    assert json.loads(path.read_text())["turns"][0]["user"] is None
    turn.set_user_text("q")
    turn.set_assistant_text("a")
    turn.finish()

    assert json.loads(path.read_text())["turns"] == [
        {"user": "q", "assistant": "a", "tools": [], "error": "", "interrupted": False, "complete": True}
    ]
    assert os.listdir(tmp_path) == ["c.json"]


def test_a_failed_write_is_a_warning_and_the_conversation_continues_in_memory(tmp_path, monkeypatch, caplog):
    path = tmp_path / "c.json"
    c = Conversation(path)
    exchange(c, "q", "a")
    before = path.read_text()

    def boom(src, dst):
        raise OSError("disk full")

    monkeypatch.setattr(conv.os, "replace", boom)
    with caplog.at_level(logging.WARNING, logger=conv.__name__):
        c.start_turn()  # no raise: a full disk must not end the conversation

    assert path.read_text() == before  # the old file is intact
    assert os.listdir(tmp_path) == ["c.json"]  # and no temp file is left behind
    assert "disk full" in caplog.text
    assert len(c) == 2  # the new turn is in memory even though it could not be saved

    # and the conversation keeps working in memory
    turn = c.start_turn()
    turn.set_user_text("still works")
    assert turn.user_text == "still works" and len(c) == 3


@pytest.mark.parametrize(
    "content",
    ["{not json", '{"version": 1}', '{"version": 2, "turns": []}', '{"version": 1, "turns": [1]}'],
)
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


def test_a_corrupt_file_that_cannot_be_moved_aside_still_starts_empty(tmp_path, caplog, monkeypatch):
    path = tmp_path / "conversation.json"
    path.write_text("{not json")

    def no_move(src, dst):  # e.g. the file or its directory belongs to another user
        raise PermissionError(13, "Read-only file system")

    monkeypatch.setattr(conv.os, "replace", no_move)
    with caplog.at_level(logging.WARNING, logger=conv.__name__):
        c = Conversation(path)  # startup must not die on the failed quarantine

    assert len(c) == 0
    assert path.exists()  # the unreadable file is left where it is
    assert "could not move it aside" in caplog.text and "starting empty" in caplog.text
    exchange(c, "works", "in memory")  # the conversation continues in memory (the saves warn)
    assert len(c) == 1


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


def test_a_file_that_cant_be_read_is_an_error_not_corruption(tmp_path):
    path = tmp_path / "conversation.json"
    path.mkdir()  # reading it raises IsADirectoryError, an OSError like a permission or I/O error

    with pytest.raises(IsADirectoryError):
        Conversation(path)
    assert path.is_dir() and not (tmp_path / "conversation.json.corrupt").exists()


def test_setting_the_same_assistant_text_again_does_not_save():
    saves = []
    turn = Turn(on_change=lambda: saves.append(1))
    turn.set_assistant_text("Paris.")
    turn.set_assistant_text("Paris.")
    assert len(saves) == 1


def test_fail_records_the_error_and_the_missing_transcript_and_finishes():
    saves = []
    turn = Turn(on_change=lambda: saves.append(1))

    turn.set_assistant_text("Half an")
    turn.fail("RuntimeError: llama-server fell over")

    assert (turn.error, turn.user_text, turn.complete) == ("RuntimeError: llama-server fell over", "", True)
    assert turn.transcribed.is_set() and turn.closed.is_set()
    assert turn.assistant_text == "Half an" and not turn.interrupted
    assert len(saves) == 3  # text, error, finish

    turn.fail("again")  # a late failure (e.g. a dropped generator) still records the error
    assert turn.error == "again"
    assert len(saves) == 4


def test_fail_keeps_a_transcript_the_responder_did_report():
    turn = Turn()
    turn.set_user_text("what was asked")
    turn.fail("boom")

    assert turn.user_text == "what was asked" and turn.error == "boom"


def test_fail_closes_the_turn_even_when_recording_the_error_fails():
    saves = []

    def on_change():
        saves.append(1)
        if len(saves) > 1:  # the error itself cannot be saved
            raise OSError("disk full")

    turn = Turn(on_change=on_change)

    with pytest.raises(OSError, match="disk full"):
        turn.fail("boom")

    assert turn.closed.is_set() and turn.complete  # the turn is closed all the same
    assert turn.error == "boom"


def test_close_keeps_the_share_of_the_reply_that_had_played():
    turn = Turn()
    turn.audio_handed_out(4.0, now=10.0)
    turn.audio_handed_out(4.0, now=10.5)  # 8 s in all, playing from t=10
    turn.set_assistant_text("a b c d e f g h")
    turn.finish()  # generation ends long before playback does

    turn.close(now=13.0)  # 3 of 8 s played

    assert turn.interrupted and turn.assistant_text == "a b c"


def test_an_interrupted_reply_keeps_at_least_one_word():
    turn = Turn()
    turn.audio_handed_out(8.0, now=10.0)
    turn.set_assistant_text("a b c d e f g h")
    turn.finish()

    turn.close(now=10.1)  # 0.1 of 8 s played: a fraction of a word

    assert turn.interrupted and turn.assistant_text == "a"


def test_an_interrupted_reply_with_no_text_stays_empty():
    turn = Turn()
    turn.audio_handed_out(8.0, now=10.0)
    turn.finish()

    turn.close(now=10.1)

    assert turn.interrupted and turn.assistant_text == ""


def test_close_after_the_audio_finished_playing_keeps_everything():
    turn = Turn()
    turn.audio_handed_out(8.0, now=10.0)
    turn.set_assistant_text("a b c d e f g h")
    turn.finish()

    turn.close(now=19.0)

    assert not turn.interrupted and turn.assistant_text == "a b c d e f g h"


def test_tool_events_are_recorded_and_persisted():
    turn = Turn()
    turn.add_tool_event("get_info", "{}", "os: Darwin")
    turn.add_tool_event("read_file", '{"path": "a"}', "text")

    assert turn.tool_events == [{"name": "get_info", "arguments": "{}", "output": "os: Darwin"},
                                {"name": "read_file", "arguments": '{"path": "a"}', "output": "text"}]
    stored = Turn.from_dict(turn.to_dict(), threading.RLock(), None)
    assert stored.tool_events == turn.tool_events


def test_tool_events_from_older_conversations_are_empty():
    turn = Turn()
    data = turn.to_dict()
    del data["tools"]  # a file written before tool events existed
    assert Turn.from_dict(data, threading.RLock(), None).tool_events == []


def test_tool_activity_for_ui_shows_the_last_calls_with_cut_arguments():
    turns = [Turn(), Turn()]
    turns[0].add_tool_event("get_info", "{}", "os: Darwin")
    turns[1].add_tool_event("exec_shell_command", json.dumps({"command": "curl " + "x" * 200}), "done")

    panel = conv.tool_activity_for_ui(turns)
    assert "**get_info** `{}`" in panel and "os: Darwin" in panel
    assert "**exec_shell_command**" in panel and panel.count("curl") == 1  # the arguments are cut
    assert len(panel) < 400


def test_tool_activity_for_ui_is_empty_without_tool_calls():
    turn = Turn()
    turn.set_assistant_text("no tools here")
    assert conv.tool_activity_for_ui([turn]) == ""


def test_tool_activity_for_ui_is_bounded_to_the_last_events():
    turns = [Turn() for _ in range(20)]
    for i, turn in enumerate(turns):
        turn.add_tool_event(f"tool_{i}", "{}", "ok")

    panel = conv.tool_activity_for_ui(turns)
    assert "tool_19" in panel and "tool_0" not in panel


def test_add_tool_event_caps_long_output_with_a_marker():
    turn = Turn()
    turn.add_tool_event("read_file", "{}", "x" * 5000)

    output = turn.tool_events[0]["output"]
    assert len(output) == conv.TOOL_EVENT_OUTPUT_CHARS
    assert output.endswith("... (truncated)")


def test_tool_activity_for_ui_marks_truncated_output():
    turn = Turn()
    turn.tool_events.append({"name": "read_file", "arguments": "{}", "output": "y" * 5000})

    panel = conv.tool_activity_for_ui([turn])
    assert "... (truncated)" in panel
    assert len(panel) < conv.TOOL_EVENT_OUTPUT_CHARS + 100


def test_a_capped_output_is_capped_only_once():
    turn = Turn()
    turn.add_tool_event("read_file", "{}", "x" * 5000)
    once = turn.tool_events[0]["output"]
    turn.add_tool_event("again", "{}", turn.tool_events[0]["output"])

    assert turn.tool_events[1]["output"] == once  # the marker is not doubled


# --- tool results in the history (responders with `tool_history`)


def exchange_with_tools(c, user, assistant, *events, finish=True):
    turn = c.start_turn()
    turn.set_user_text(user)
    for name, output in events:
        turn.add_tool_event(name, "{}", output)
    turn.set_assistant_text(assistant)
    if finish:
        turn.finish()
    return turn


def test_history_without_tools_is_what_it_always_was_even_when_turns_have_tool_events():
    c = Conversation()
    exchange_with_tools(c, "memory?", "48 gigabytes.", ("system_info", "48 gigabytes of memory"))

    expected = [("user", "memory?"), ("assistant", "48 gigabytes.")]
    assert c.history(before=c.start_turn()) == expected
    assert c.history(before=c.start_turn(), tools=False) == expected


def test_history_with_tools_puts_each_result_between_the_user_and_assistant_messages():
    c = Conversation()
    exchange_with_tools(c, "up and memory?", "Three days, 48 gigabytes.",
                        ("system_info", "up 3 days"), ("system_info", "48 gigabytes of memory"))
    exchange(c, "thanks", "You are welcome.")

    assert c.history(before=c.start_turn(), tools=True) == [
        ("user", "up and memory?"),
        ("tool", "system_info: up 3 days"),
        ("tool", "system_info: 48 gigabytes of memory"),
        ("assistant", "Three days, 48 gigabytes."),
        ("user", "thanks"),
        ("assistant", "You are welcome."),
    ]


def test_a_tool_result_in_history_is_capped_at_300_characters():
    c = Conversation()
    exchange_with_tools(c, "q", "a", ("find_files", "x" * 500))

    history = c.history(before=c.start_turn(), tools=True)

    role, text = history[1]
    assert role == "tool"
    assert len(text) == len("find_files: ") + 300
    assert text.endswith("... (truncated)")


def test_tool_results_follow_the_same_windowing_and_dropping_rules_as_the_turns():
    c = Conversation(window=2)  # blocks of 2 turns
    exchange_with_tools(c, "q0", "a0", ("disk_space", "r0"))
    exchange_with_tools(c, "q1", "a1", ("disk_space", "r1"))
    exchange_with_tools(c, "q2", "a2", ("disk_space", "r2"))
    c.start_turn()  # its transcript never arrives: dropped
    probe = c.start_turn()

    assert c.history(before=probe, tools=True) == [
        ("user", "q2"), ("tool", "disk_space: r2"), ("assistant", "a2")]


def test_an_interrupted_turn_with_no_spoken_text_carries_no_tool_messages():
    c = Conversation()
    exchange_with_tools(c, "q", "", ("disk_space", "160 gigabytes free"), finish=False)

    assert c.history(before=c.start_turn(), tools=True) == [("user", "q")]


def test_tool_results_survive_a_reload_from_disk(tmp_path):
    path = tmp_path / "c.json"
    c = Conversation(path)
    exchange_with_tools(c, "q", "a", ("battery_status", "80 percent"))
    c.start_turn()

    loaded = Conversation(path)

    assert loaded.history(before=loaded.start_turn(), tools=True)[:3] == [
        ("user", "q"), ("tool", "battery_status: 80 percent"), ("assistant", "a")]


# --- a replayed call carries its real arguments (carrier: a JSON object `{"name", "arguments", "output"}`)


def history_of(*events):
    """The tool entries `Conversation.history(tools=True)` gives for one answered turn with these events."""
    c = Conversation()
    turn = c.start_turn()
    turn.set_user_text("q")
    for name, arguments, output in events:
        turn.add_tool_event(name, arguments, output)
    turn.set_assistant_text("a")
    turn.finish()
    return [text for role, text in c.history(before=c.start_turn(), tools=True) if role == "tool"]


def replayed(*events):
    """(name, arguments, content) of every call `build_messages` makes from these stored events."""
    from vaice.responders.assistant import build_messages

    c = Conversation()
    turn = c.start_turn()
    turn.set_user_text("q")
    for name, arguments, output in events:
        turn.add_tool_event(name, arguments, output)
    turn.set_assistant_text("a")
    turn.finish()
    messages = build_messages("p", c.history(before=c.start_turn(), tools=True), "next")
    calls = [call for m in messages for call in m.get("tool_calls", [])]
    contents = [m["content"] for m in messages if m["role"] == "tool"]
    return [(call["function"]["name"], call["function"]["arguments"], content)
            for call, content in zip(calls, contents, strict=True)]


def test_a_replayed_call_carries_the_arguments_it_was_made_with():
    snapshot = ("chrome__take_snapshot", '{"pageId": 1, "verbose": true}', "uid=1_0 RootWebArea")

    assert replayed(snapshot) == [("chrome__take_snapshot", '{"pageId":1,"verbose":true}', "uid=1_0 RootWebArea")]


def test_arguments_with_spaces_after_colons_and_an_output_holding_colons_still_split_correctly():
    out = replayed(("notes__add", '{"title": "a: b", "text": "x"}', 'said: "hello": world'))

    assert out == [("notes__add", '{"title":"a: b","text":"x"}', 'said: "hello": world')]


def test_arguments_that_are_not_a_json_object_replay_as_an_empty_object():
    for bad in ('{"pageId": 1', "[1, 2]", '"text"', "null", "42", ""):
        assert replayed(("chrome__take_snapshot", bad, "ok")) == [("chrome__take_snapshot", "{}", "ok")], bad


def test_arguments_longer_than_the_cap_replay_as_an_empty_object_and_are_never_truncated():
    from vaice.conversation import TOOL_HISTORY_ARGUMENT_CHARS

    assert TOOL_HISTORY_ARGUMENT_CHARS == 300
    at_cap = json.dumps({"q": "x" * (TOOL_HISTORY_ARGUMENT_CHARS - len('{"q":""}'))}, separators=(",", ":"))
    assert len(at_cap) == TOOL_HISTORY_ARGUMENT_CHARS

    assert replayed(("look_at_screen", at_cap, "r")) == [("look_at_screen", at_cap, "r")]
    assert replayed(("look_at_screen", json.dumps({"q": "x" * 400}), "r")) == [("look_at_screen", "{}", "r")]


def test_the_cap_counts_the_compact_form_not_the_stored_spacing():
    spaced = '{ "q" :  "' + "x" * 290 + '"   }'  # over 300 as stored, 298 compact
    assert len(spaced) > 300

    assert replayed(("look_at_screen", spaced, "r"))[0][1] == json.dumps({"q": "x" * 290}, separators=(",", ":"))


def test_unicode_argument_values_survive_unescaped():
    assert replayed(("notes__add", '{"text": "caf\\u00e9 \\u65e5\\u672c"}', "ok"))[0][1] == '{"text":"café 日本"}'


def test_an_event_without_arguments_or_with_empty_ones_stays_a_plain_entry():
    assert history_of(("battery_status", "{}", "80 percent"), ("disk_space", "", "160 free")) == [
        "battery_status: 80 percent", "disk_space: 160 free"]
    assert replayed(("disk_space", "", "160 free")) == [("disk_space", "{}", "160 free")]


def test_a_legacy_entry_replays_as_an_empty_object():
    from vaice.responders.assistant import build_messages

    messages = build_messages("p", [("user", "q"), ("tool", "disk_space: a: b"), ("assistant", "a")], "next")

    call = messages[2]["tool_calls"][0]["function"]
    assert call == {"name": "disk_space", "arguments": "{}"}
    assert messages[3]["content"] == "a: b"


def test_a_brace_entry_that_is_not_a_valid_carrier_is_read_as_a_plain_entry():
    from vaice.conversation import parse_tool_entry

    assert parse_tool_entry('{not json: out') == ("{not json", "{}", "out")
    assert parse_tool_entry('{"name": "x"}') == ('{"name"', "{}", '"x"}')  # no output key: not a carrier


# --- a call that waited for confirmation is replayed once, with its final event

SAFARI = '{"name":"Safari"}'


def test_a_call_that_waited_and_then_finished_replays_once_with_its_result():
    assert replayed(("quit_app", SAFARI, "waiting for confirmation"),
                    ("quit_app", SAFARI, "quit Safari")) == [("quit_app", SAFARI, "quit Safari")]
    for final in ("cancelled", "not confirmed", "expired"):
        assert replayed(("quit_app", SAFARI, "waiting for confirmation"),
                        ("quit_app", SAFARI, final)) == [("quit_app", SAFARI, final)]


def test_a_call_still_waiting_keeps_its_single_entry():
    assert replayed(("quit_app", SAFARI, "waiting for confirmation")) == [
        ("quit_app", SAFARI, "waiting for confirmation")]


def test_a_later_event_with_other_arguments_or_another_name_does_not_hide_the_waiting_one():
    waiting = ("quit_app", SAFARI, "waiting for confirmation")

    assert replayed(waiting, ("quit_app", '{"name":"Mail"}', "quit Mail")) == [
        ("quit_app", SAFARI, "waiting for confirmation"), ("quit_app", '{"name":"Mail"}', "quit Mail")]
    assert replayed(waiting, ("open_app", SAFARI, "opened Safari")) == [
        ("quit_app", SAFARI, "waiting for confirmation"), ("open_app", SAFARI, "opened Safari")]


def test_an_earlier_event_does_not_hide_a_later_waiting_one():
    assert replayed(("quit_app", SAFARI, "quit Safari"), ("quit_app", SAFARI, "waiting for confirmation")) == [
        ("quit_app", SAFARI, "quit Safari"), ("quit_app", SAFARI, "waiting for confirmation")]


def test_two_identical_calls_in_one_turn_each_replay_their_result_only():
    assert replayed(("quit_app", SAFARI, "waiting for confirmation"), ("quit_app", SAFARI, "quit Safari"),
                    ("quit_app", SAFARI, "waiting for confirmation"), ("quit_app", SAFARI, "cancelled")) == [
        ("quit_app", SAFARI, "quit Safari"), ("quit_app", SAFARI, "cancelled")]
    # the second one is still waiting: only the first has a later match
    assert replayed(("quit_app", SAFARI, "waiting for confirmation"), ("quit_app", SAFARI, "quit Safari"),
                    ("quit_app", SAFARI, "waiting for confirmation")) == [
        ("quit_app", SAFARI, "quit Safari"), ("quit_app", SAFARI, "waiting for confirmation")]


def test_the_stored_events_and_the_activity_panel_still_hold_both_events():
    c = Conversation()
    turn = c.start_turn()
    turn.set_user_text("q")
    turn.add_tool_event("quit_app", SAFARI, "waiting for confirmation")
    turn.add_tool_event("quit_app", SAFARI, "quit Safari")
    turn.set_assistant_text("a")
    turn.finish()

    assert [e["output"] for e in turn.tool_events] == ["waiting for confirmation", "quit Safari"]
    assert [e["output"] for e in turn.to_dict()["tools"]] == ["waiting for confirmation", "quit Safari"]
    activity = c.tool_activity_for_ui()
    assert "waiting for confirmation" in activity and "quit Safari" in activity
    assert len(history_of(*[(e["name"], e["arguments"], e["output"]) for e in turn.tool_events])) == 1
