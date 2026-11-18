"""The confirmation state: what counts as a spoken yes or no, and the one request that waits."""
import threading

import pytest

from vaice.confirm import CONFIRM_SECONDS, NO, YES, Confirmations, Pending, spoken_answer
from vaice.conversation import Turn


@pytest.mark.parametrize("text", [*sorted(YES), "Yes.", "  YES  ", "Yes, please!", "yes   do   it", "Go ahead.",
                                  "Do it!", "Confirmed", "YEP!!!"])
def test_whole_utterances_that_say_yes(text):
    assert spoken_answer(text) == "yes"


@pytest.mark.parametrize("text", [*sorted(NO), "No.", "NO!", "Never mind.", "Don't!", "don’t",  # noqa: RUF001
                                  "Do not.", "No, thanks.", "Cancel that."])
def test_whole_utterances_that_say_no(text):
    assert spoken_answer(text) == "no"


@pytest.mark.parametrize("text", [
    "yes, and also open Safari", "I said yes to the other one", "no way, not now", "tell me yes or no",
    "yes yes", "go", "ahead", "sure", "okay", "", "   ", "...", "please", "don't do it", "stop it",
    "yesterday", "no one knows", "cancel my meeting"])
def test_a_phrase_inside_a_longer_sentence_or_a_near_miss_is_not_an_answer(text):
    assert spoken_answer(text) is None


def test_no_phrase_is_both_yes_and_no():
    assert not {spoken_answer(phrase) for phrase in YES} - {"yes"}
    assert not {spoken_answer(phrase) for phrase in NO} - {"no"}


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


def pending(name="quit_app", turn=None):
    return Pending(name, '{"name": "Safari"}', "Shall I quit Safari?", "call_0", [{"role": "user", "content": "q"}],
                   turn or Turn())


def test_a_pending_has_a_unique_id_and_is_not_armed():
    first, second = pending(), pending()

    assert first.id != second.id and len(first.id) == 32
    assert first.armed_at is None


def test_nothing_waits_at_first():
    confirmations = Confirmations()
    assert confirmations.current() is None
    assert confirmations.take() is None
    confirmations.arm("anything")  # nothing to arm: no error
    confirmations.clear()


def test_a_request_is_shown_at_once_and_armed_later():
    clock = Clock()
    confirmations = Confirmations(clock)
    request = pending()

    confirmations.ask(request)
    assert confirmations.current() is request and request.armed_at is None

    confirmations.arm(request.id)
    assert request.armed_at == 1000.0


def test_arming_a_request_that_was_replaced_does_nothing():
    confirmations = Confirmations(Clock())
    old, new = pending(), pending()
    confirmations.ask(old)
    confirmations.ask(new)  # replaces

    confirmations.arm(old.id)

    assert confirmations.current() is new
    assert old.armed_at is None and new.armed_at is None


def test_a_request_expires_after_the_time_counted_from_when_it_was_armed():
    clock = Clock()
    confirmations = Confirmations(clock)
    request = pending()
    confirmations.ask(request)
    clock.now += 3600  # an unarmed request does not expire: its question may still be being spoken
    assert confirmations.current() is request

    confirmations.arm(request.id)
    clock.now += CONFIRM_SECONDS - 0.1
    assert confirmations.current() is request
    clock.now += 0.1
    assert confirmations.current() is None
    assert confirmations.take(request.id) is None  # gone for good


def test_an_expired_request_cannot_be_taken():
    clock = Clock()
    confirmations = Confirmations(clock)
    request = pending()
    confirmations.ask(request)
    confirmations.arm(request.id)
    clock.now += CONFIRM_SECONDS + 1

    assert confirmations.take() is None
    assert confirmations.take(request.id) is None
    # said once on the asking turn, so the Tool activity panel does not show it waiting for ever
    assert [event["output"] for event in request.turn.tool_events] == ["expired"]


def test_take_by_id_takes_only_that_request_so_an_old_panel_cannot_approve_a_newer_one():
    confirmations = Confirmations(Clock())
    old, new = pending(), pending()
    confirmations.ask(old)
    confirmations.ask(new)

    assert confirmations.take(old.id) is None
    assert confirmations.current() is new
    assert confirmations.take(new.id) is new
    assert confirmations.current() is None


def test_take_without_an_id_takes_whatever_waits_and_only_once():
    confirmations = Confirmations(Clock())
    request = pending()
    confirmations.ask(request)

    assert confirmations.take() is request
    assert confirmations.take() is None


def test_clear_drops_the_request():
    confirmations = Confirmations(Clock())
    confirmations.ask(pending())
    confirmations.clear()
    assert confirmations.current() is None


def race_for(request, confirmations):
    """Two threads take `request` at the same moment (one by its id, one without); what each got."""
    start = threading.Barrier(2)
    won = []

    def take(by_id):
        start.wait()
        won.append(confirmations.take(request.id if by_id else None))

    threads = [threading.Thread(target=take, args=(by_id,)) for by_id in (True, False)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(5)
    return won


def test_of_two_takes_at_once_exactly_one_wins():
    for _ in range(50):  # a race: repeat so a lost lock would show
        confirmations = Confirmations()
        request = pending()
        confirmations.ask(request)
        confirmations.arm(request.id)

        won = race_for(request, confirmations)

        assert sorted(item is not None for item in won) == [False, True]


# --- an auto request: the time left


def auto_request(seconds=5.0):
    return Pending("nav", "{}", "Using nav in 5 seconds.", "call_0", [], Turn(), auto_seconds=seconds)


def test_a_request_without_a_timer_never_has_time_left():
    clock = Clock()
    confirmations = Confirmations(clock)
    request = pending()
    confirmations.ask(request)
    confirmations.arm(request.id)

    assert request.auto_seconds == 0
    assert request.remaining(clock()) is None and confirmations.remaining(request) is None


def test_the_time_left_counts_from_arming_and_is_none_before_it():
    clock = Clock()
    confirmations = Confirmations(clock)
    request = auto_request()
    confirmations.ask(request)
    assert confirmations.remaining(request) is None  # asked, not armed: the question may still be playing

    confirmations.arm(request.id)
    assert confirmations.remaining(request) == 5.0
    clock.now += 3.5
    assert confirmations.remaining(request) == 1.5
    clock.now += 10
    assert confirmations.remaining(request) == 0.0  # never negative


def test_there_is_no_time_left_for_a_request_that_is_gone_replaced_or_expired():
    clock = Clock()
    confirmations = Confirmations(clock)
    old, new = auto_request(), auto_request()
    confirmations.ask(old)
    confirmations.arm(old.id)
    confirmations.ask(new)  # replaces
    confirmations.arm(new.id)
    assert confirmations.remaining(old) is None and confirmations.remaining(new) == 5.0

    confirmations.take(new.id)
    assert confirmations.remaining(new) is None

    long = auto_request(CONFIRM_SECONDS * 2)  # the answer window still ends at CONFIRM_SECONDS
    confirmations.ask(long)
    confirmations.arm(long.id)
    clock.now += CONFIRM_SECONDS
    assert confirmations.remaining(long) is None


def test_the_time_left_includes_the_audio_still_to_play_when_the_request_was_armed():
    clock = Clock()
    confirmations = Confirmations(clock)
    request = auto_request()
    confirmations.ask(request)

    confirmations.arm(request.id, playing=3.0)

    assert request.auto_delay == 3.0 and request.armed_at == 1000.0  # answerability and expiry start now
    assert confirmations.remaining(request) == 8.0
    clock.now += 7.5
    assert confirmations.remaining(request) == 0.5
    clock.now += 1
    assert confirmations.remaining(request) == 0.0


def test_audio_still_playing_is_never_negative_and_defaults_to_none():
    confirmations = Confirmations(Clock())
    negative, plain, ordinary = auto_request(), auto_request(), pending()
    for request in (negative, plain, ordinary):
        confirmations.ask(request)
        confirmations.arm(request.id) if request is not negative else confirmations.arm(request.id, playing=-2.0)

    assert negative.auto_delay == 0.0 and plain.auto_delay == 0.0
    assert ordinary.armed_at == 1000.0 and ordinary.auto_seconds == 0  # a request without a timer: as before
    assert ordinary.remaining(1000.0) is None
