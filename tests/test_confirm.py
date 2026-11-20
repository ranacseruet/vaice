"""The confirmation state: the one request that waits for a click."""
import threading

import pytest

from vaice.confirm import CONFIRM_SECONDS, Confirmations, Pending, Waiter
from vaice.conversation import Turn


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


def pending(name="quit_app", turn=None):
    return Pending(name, '{"name": "Safari"}', "Shall I quit Safari?", turn or Turn())


def test_a_pending_has_a_unique_id_and_has_not_been_asked():
    first, second = pending(), pending()

    assert first.id != second.id and len(first.id) == 32
    assert first.asked_at is None


def test_nothing_waits_at_first():
    confirmations = Confirmations()
    assert confirmations.current() is None
    assert confirmations.take() is None
    confirmations.clear()


def test_a_request_is_answerable_at_once():
    clock = Clock()
    confirmations = Confirmations(clock)
    request = pending()

    confirmations.ask(request)

    assert confirmations.current() is request and request.asked_at == 1000.0
    assert confirmations.take(request.id) is request


def test_a_request_expires_after_the_time_counted_from_when_it_was_asked():
    clock = Clock()
    confirmations = Confirmations(clock)
    request = pending()
    confirmations.ask(request)

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

        won = race_for(request, confirmations)

        assert sorted(item is not None for item in won) == [False, True]


# --- a question of a running task: the answer goes to a waiter, nothing is run


def asking():
    waiter = Waiter()
    return Pending("mcp__chrome__click", '{"uid": "1"}', "Shall I click something?", Turn(), waiter=waiter)


def test_a_waiter_keeps_the_first_answer_and_reports_none_until_then():
    waiter = Waiter()
    assert waiter.wait(0) is None

    waiter.resolve(True)
    waiter.resolve(False)

    assert waiter.wait(0) is True


def test_answering_settles_the_waiter_and_the_request_in_one_step():
    confirmations = Confirmations(Clock())
    request = asking()
    confirmations.ask(request)

    assert confirmations.answer(request.id, True) is request

    assert request.waiter.wait(0) is True and confirmations.current() is None
    assert confirmations.answer(request.id, False) is None  # settled once; a later answer changes nothing
    assert request.waiter.wait(0) is True


def test_answering_a_stale_id_settles_nothing():
    confirmations = Confirmations(Clock())
    request = asking()
    confirmations.ask(request)

    assert confirmations.answer("not-the-id", True) is None

    assert confirmations.current() is request and request.waiter.wait(0) is None


@pytest.mark.parametrize("how", ["take", "clear", "replaced", "expired"])
def test_every_other_way_a_request_leaves_is_a_no_to_its_waiter(how):
    clock = Clock()
    confirmations = Confirmations(clock)
    request = asking()
    confirmations.ask(request)

    if how == "take":
        confirmations.take(request.id)
    elif how == "clear":
        confirmations.clear()
    elif how == "replaced":
        confirmations.ask(pending())
    else:
        clock.now += CONFIRM_SECONDS + 1
        assert confirmations.current() is None

    assert request.waiter.wait(0) is False


def test_a_yes_and_a_click_cancel_race_gives_the_waiter_exactly_the_winners_answer():
    for _ in range(200):
        confirmations = Confirmations()
        request = asking()
        confirmations.ask(request)
        start = threading.Barrier(2)
        won = []

        def yes(start=start, confirmations=confirmations, request=request, won=won):
            start.wait()
            if confirmations.answer(request.id, True) is not None:
                won.append(True)

        def cancel(start=start, confirmations=confirmations, request=request, won=won):
            start.wait()
            if confirmations.take(request.id) is not None:
                won.append(False)

        threads = [threading.Thread(target=yes), threading.Thread(target=cancel)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert len(won) == 1 and request.waiter.wait(0) is won[0]


def test_a_clear_refuses_a_request_from_an_asker_that_began_before_it():
    confirmations = Confirmations()
    began = confirmations.epoch
    waiter = Waiter()
    stale = Pending("browse_step", "{}", "Shall I click?", Turn(), waiter=waiter)

    confirmations.clear()  # "New conversation" while the asker was still preparing its request

    assert confirmations.ask(stale, began) is False
    assert confirmations.current() is None and waiter.wait(0) is False  # a refused step is a no
    fresh = pending()
    assert confirmations.ask(fresh, confirmations.epoch) is True and confirmations.current() is fresh
    assert confirmations.ask(pending()) is True  # without an epoch, as before
