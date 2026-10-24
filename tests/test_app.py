import logging

import numpy as np

from vaice.app import make_handler


class FakeResponder:
    name = "fake"

    def respond(self, audio):
        yield (24000, np.zeros(1, dtype=np.float32))


def test_each_reply_logs_which_backend_answered(caplog):
    with caplog.at_level(logging.INFO, logger="vaice.app"):
        chunks = list(make_handler(FakeResponder())((48000, np.zeros((1, 480), dtype=np.int16))))

    assert len(chunks) == 1
    assert "[fake] time to first audio" in caplog.text
    assert "[fake] reply finished" in caplog.text
