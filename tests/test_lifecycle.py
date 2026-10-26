"""tests for the app's stop signal"""
import signal

import pytest

from vaice import lifecycle


@pytest.fixture(autouse=True)
def clean_state():
    """The stop flag is global: leave it as it was found."""
    was_set = lifecycle.stopping.is_set()
    yield
    lifecycle.stopping.clear()
    if was_set:
        lifecycle.stopping.set()


def test_requesting_a_stop_sets_the_flag():
    assert not lifecycle.stopping.is_set()

    lifecycle.request_stop()

    assert lifecycle.stopping.is_set()


def test_the_sigint_handler_asks_the_app_to_stop_and_keeps_the_default_behaviour():
    previous = signal.getsignal(signal.SIGINT)
    try:
        lifecycle.install_stop_handler()

        with pytest.raises(KeyboardInterrupt):
            signal.raise_signal(signal.SIGINT)

        assert lifecycle.stopping.is_set()
    finally:
        signal.signal(signal.SIGINT, previous)


def test_the_sigint_handler_is_installed_instead_of_the_default_one():
    previous = signal.getsignal(signal.SIGINT)
    try:
        lifecycle.install_stop_handler()

        assert signal.getsignal(signal.SIGINT) is not previous
    finally:
        signal.signal(signal.SIGINT, previous)