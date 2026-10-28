import pytest
from fake_realtime import FakeRealtime

from vaice import lifecycle


@pytest.fixture(autouse=True)
def not_stopping():
    """lifecycle.stopping is process-wide: a test that sets it must not stop the next one."""
    lifecycle.stopping.clear()
    yield
    lifecycle.stopping.clear()



@pytest.fixture
def realtime():
    """Starts fake /v1/realtime servers (FakeRealtime(**kwargs)) and stops them after the test."""
    servers = []

    def start(**kwargs):
        servers.append(FakeRealtime(**kwargs).start())
        return servers[-1]

    yield start
    for server in servers:
        server.stop()
