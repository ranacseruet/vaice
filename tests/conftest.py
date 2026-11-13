import os
import tempfile

import pytest
from fake_realtime import FakeRealtime

from vaice import lifecycle

# config.py loads a .env from the working directory at import, and conftest is imported before
# the test modules (which import vaice.config), so tests run from an empty temp directory: a
# developer's repo .env with local tuning (e.g. VAICE_VAD_THRESHOLD) must not change test
# expectations. Tests that need a specific working directory chdir themselves.
os.chdir(tempfile.mkdtemp(prefix="vaice-tests-"))


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
