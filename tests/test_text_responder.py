import json
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
from fake_realtime import next_on_fresh_thread

from vaice.conversation import Turn
from vaice.responders import text
from vaice.responders.text import TextResponder, complete_sentences


class CountingTurn(Turn):
    def __init__(self):
        super().__init__()
        self.user_texts = []

    def set_user_text(self, value):
        self.user_texts.append(value)
        super().set_user_text(value)


class FakeRecognizer:
    class Stream:
        result = type("Result", (), {"text": "What is two plus two?"})()

        def accept_waveform(self, sample_rate, audio):
            assert sample_rate == 16000

    def create_stream(self):
        return self.Stream()

    def decode_stream(self, stream):
        return None


class FakeTts:
    sample_rate = 10

    def __init__(self):
        self.sentences = []

    def generate(self, sentence, sid, speed):
        self.sentences.append(sentence)
        return type("Generated", (), {"samples": np.ones(40, dtype=np.float32), "sample_rate": self.sample_rate})()


class FakeResponse:
    status_code = 200
    text = ""

    def __init__(self, chunks):
        self.chunks = chunks
        self.closed = threading.Event()

    def raise_for_status(self):
        return None

    def iter_lines(self, chunk_size=512, decode_unicode=True):
        yield from self.chunks

    def close(self):
        self.closed.set()


def sse(*deltas):
    return [f"data: {json.dumps({'choices': [{'delta': {'content': value}}]})}" for value in deltas] + ["data: [DONE]"]


def make_responder(monkeypatch, tts=None, response=None):
    response = response or FakeResponse(sse("Four. ", "The answer is four."))
    monkeypatch.setattr(text.requests, "post", lambda *args, **kwargs: response)
    return TextResponder(
        url="http://text-server:8001/v1",
        prompt="Answer briefly.",
        timeout=5,
        model_id="chat-model",
        recognizer=FakeRecognizer(),
        tts=tts or FakeTts(),
    ), response


def test_sentence_buffer_keeps_partial_text_until_boundary_or_flush():
    sentences, pending = complete_sentences("First. Sec", final=False)
    assert sentences == ["First."]
    assert pending == "Sec"
    assert complete_sentences(pending + "ond?", final=True) == (["Second?"], "")
    assert complete_sentences("The value is 3.", final=False) == ([], "The value is 3.")


def test_importing_text_pipeline_does_not_load_optional_sherpa_models():
    import vaice.responders.text  # noqa: F401

    assert "sherpa_onnx" not in sys.modules


def test_text_pipeline_streams_sentence_audio_and_records_transcript_once(monkeypatch):
    monkeypatch.setattr(text, "FRAME_SAMPLES", 20)
    tts = FakeTts()
    worker, _ = make_responder(monkeypatch, tts=tts)
    turn = CountingTurn()
    generator = worker.respond((16000, np.zeros(1600, dtype=np.int16)), [("assistant", "Earlier answer.")], turn)

    first = next_on_fresh_thread(generator)
    assert isinstance(first, tuple)
    assert first[0] == 10
    assert first[1].dtype == np.float32
    assert turn.transcribed.wait(1)
    assert turn.user_texts == ["What is two plus two?"]
    assert turn.assistant_text == "Four."
    assert tts.sentences == ["Four.", "The answer is four."]

    rest = []
    while True:
        item = next_on_fresh_thread(generator)
        if isinstance(item, StopIteration):
            break
        assert not isinstance(item, BaseException)
        rest.append(item)
    assert len(rest) == 3
    assert turn.assistant_text == "Four. The answer is four."


def test_text_audio_accounting_keeps_completed_sentence_when_later_tts_runs_ahead(monkeypatch):
    monkeypatch.setattr(text, "FRAME_SAMPLES", 20)
    tts = FakeTts()

    def generate(sentence, sid, speed):
        samples = 40 if sentence.startswith("One two three four five") else 80
        return type("Generated", (), {"samples": np.ones(samples, dtype=np.float32), "sample_rate": 10})()

    tts.generate = generate
    worker, _ = make_responder(monkeypatch, tts=tts, response=FakeResponse(sse("One two three four five. Done.")))
    turn = CountingTurn()
    generator = worker.respond((16000, np.zeros(1600, dtype=np.int16)), [], turn)

    next_on_fresh_thread(generator)
    next_on_fresh_thread(generator)
    completed = "One two three four five."
    assert turn.assistant_text == completed
    next_on_fresh_thread(generator)
    assert turn.assistant_text.startswith(completed)


def test_text_chat_request_contains_system_history_and_local_transcript(monkeypatch):
    seen = {}
    response = FakeResponse(sse("Four."))

    def post(url, **kwargs):
        seen["url"] = url
        seen["json"] = kwargs["json"]
        return response

    monkeypatch.setattr(text.requests, "post", post)
    worker = TextResponder(
        url="http://text-server:8001/v1",
        prompt="Answer briefly.",
        timeout=5,
        model_id="chat-model",
        recognizer=FakeRecognizer(),
        tts=FakeTts(),
    )
    turn = CountingTurn()
    generator = worker.respond((16000, np.zeros(1600, dtype=np.int16)), [("user", "Earlier question")], turn)
    while True:
        item = next_on_fresh_thread(generator)
        if isinstance(item, StopIteration):
            break
        assert not isinstance(item, BaseException)

    assert seen["url"] == "http://text-server:8001/v1/chat/completions"
    assert seen["json"]["model"] == "chat-model"
    assert seen["json"]["stream"] is True
    assert seen["json"]["messages"] == [
        {"role": "system", "content": "Answer briefly."},
        {"role": "user", "content": "Earlier question"},
        {"role": "user", "content": "What is two plus two?"},
    ]


def test_text_chat_flushes_a_short_sentence_before_the_http_stream_ends():
    first_event_sent = threading.Event()
    allow_finish = threading.Event()
    done_event_sent = threading.Event()

    class StreamingHandler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_POST(self):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write((sse("Early. ")[0] + "\n\n").encode())
            self.wfile.flush()
            first_event_sent.set()
            allow_finish.wait(3)
            try:
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
            except OSError:
                pass
            done_event_sent.set()

        def log_message(self, format, *args):
            return None

    server = ThreadingHTTPServer(("127.0.0.1", 0), StreamingHandler)
    server.daemon_threads = True
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    worker = TextResponder(
        url=f"http://127.0.0.1:{server.server_port}/v1",
        timeout=5,
        model_id="chat-model",
        recognizer=FakeRecognizer(),
        tts=FakeTts(),
    )
    turn = CountingTurn()
    generator = worker.respond((16000, np.zeros(1600, dtype=np.int16)), [], turn)
    first_chunk = {}
    consumer_finished = threading.Event()

    def consume_first_chunk():
        try:
            first_chunk["value"] = next(generator)
        except BaseException as exc:
            first_chunk["value"] = exc
        consumer_finished.set()

    consumer = threading.Thread(target=consume_first_chunk, daemon=True)
    consumer.start()
    try:
        assert first_event_sent.wait(1)
        audio_arrived_while_open = consumer_finished.wait(0.75)
        done_was_sent_before_release = done_event_sent.is_set()
        allow_finish.set()
        consumer.join(2)
        assert not consumer.is_alive()
        assert audio_arrived_while_open, "the first sentence waited for more SSE bytes or stream closure"
        assert not done_was_sent_before_release
        assert isinstance(first_chunk["value"], tuple)
        assert turn.assistant_text == "Early."
    finally:
        allow_finish.set()
        if consumer.is_alive():
            consumer.join(2)
        if not consumer.is_alive():
            generator.close()
        server.shutdown()
        server.server_close()
        server_thread.join(1)


def test_model_discovery_uses_the_configured_text_api_prefix(monkeypatch):
    seen = []

    class ModelsResponse:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            return None

        def raise_for_status(self):
            pass

        def json(self):
            return {"data": [{"id": "chat-model"}]}

    monkeypatch.setattr(text.requests, "get", lambda url, **kwargs: seen.append(url) or ModelsResponse())
    assert text._model_id("https://chat.example.test/gateway/v1", 10) == "chat-model"
    assert seen == ["https://chat.example.test/gateway/v1/models"]


def test_http_error_still_closes_the_response_and_reports_transcript_once(monkeypatch):
    class BadResponse(FakeResponse):
        def raise_for_status(self):
            raise text.requests.HTTPError("chat endpoint rejected request")

    response = BadResponse([])
    worker, _ = make_responder(monkeypatch, response=response)
    turn = CountingTurn()
    generator = worker.respond((16000, np.zeros(1600, dtype=np.int16)), [], turn)
    error = next_on_fresh_thread(generator)

    assert isinstance(error, text.requests.HTTPError)
    assert "rejected request" in str(error)
    assert response.closed.is_set()
    assert turn.user_texts == ["What is two plus two?"]


def test_interrupt_keeps_only_the_tts_text_covered_by_audio_handed_out(monkeypatch):
    monkeypatch.setattr(text, "FRAME_SAMPLES", 20)
    tts = FakeTts()
    tts.generate = lambda sentence, sid, speed: type(
        "Generated", (), {"samples": np.ones(80, dtype=np.float32), "sample_rate": 10}
    )()
    worker, _ = make_responder(monkeypatch, tts=tts, response=FakeResponse(sse("One two three four.")))
    turn = CountingTurn()
    generator = worker.respond((16000, np.zeros(1600, dtype=np.int16)), [], turn)

    item = next_on_fresh_thread(generator)
    assert isinstance(item, tuple)
    assert turn.transcribed.wait(1)
    turn.audio_handed_out(len(item[1]) / item[0], now=100)
    turn.close(now=100.1)
    assert turn.interrupted is True
    assert turn.assistant_text == "One"
    generator.close()


def test_shutdown_abandons_text_stream_and_closes_the_response(monkeypatch):
    entered = threading.Event()
    response = FakeResponse([])

    def lines(chunk_size=512, decode_unicode=True):
        entered.set()
        while not response.closed.is_set():
            time.sleep(0.01)
            yield None

    response.iter_lines = lines
    worker, _ = make_responder(monkeypatch, response=response)
    turn = CountingTurn()
    generator = worker.respond((16000, np.zeros(1600, dtype=np.int16)), [], turn)
    result = []
    next_worker = threading.Thread(target=lambda: result.append(next_on_fresh_thread(generator)))
    next_worker.start()
    assert entered.wait(1)
    turn.close()
    next_worker.join(1)
    assert not next_worker.is_alive()
    assert isinstance(result[0], StopIteration)
    assert response.closed.wait(1)


def test_shutdown_exits_when_http_response_close_is_blocked():
    script = r"""
import threading
import time
import numpy as np
from vaice import lifecycle
from vaice.conversation import Turn
from vaice.responders import text
from vaice.responders.text import TextResponder

entered = threading.Event()
class Response:
    def raise_for_status(self):
        pass
    def iter_lines(self, chunk_size=512, decode_unicode=True):
        entered.set()
        while True:
            time.sleep(0.01)
            yield None
    def close(self):
        time.sleep(60)

class Recognizer:
    class Stream:
        result = type('Result', (), {'text': 'question'})()
        def accept_waveform(self, rate, audio):
            pass
    def create_stream(self):
        return self.Stream()
    def decode_stream(self, stream):
        pass

class Tts:
    def generate(self, sentence, sid, speed):
        return type('Generated', (), {'samples': np.ones(10, dtype=np.float32), 'sample_rate': 24000})()

text.POLL_SECONDS = 0.01
text.requests.post = lambda *args, **kwargs: Response()
turn = Turn()
responder = TextResponder(recognizer=Recognizer(), tts=Tts(), model_id='chat-model')
generator = responder.respond((16000, np.zeros(1600, dtype=np.int16)), [], turn)
failures = []
def consume():
    try:
        next(generator)
    except StopIteration:
        pass
    except BaseException as exc:
        failures.append(exc)
worker = threading.Thread(target=consume)
worker.start()
if not entered.wait(1):
    raise SystemExit('HTTP stream was not entered')
lifecycle.stopping.set()
worker.join(1)
if worker.is_alive():
    raise SystemExit('fastrtc resume waited for blocked response.close()')
if failures:
    raise SystemExit(repr(failures[0]))
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        timeout=3,
    )
    assert result.returncode == 0, result.stderr or result.stdout
