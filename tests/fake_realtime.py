"""A fake llama-server /v1/realtime endpoint, for the llamacpp backend's tests (no model involved).

Speaks the events the real server does (session.created / session.updated, response.output_*.delta,
response.done, the transcription events, error), records what the client sent, and plays a script
for the reply and for the transcription, so a test can make the server stall, fail or hang up.
"""
import base64
import contextlib
import json
import threading
import time

import numpy as np
from websockets.exceptions import ConnectionClosed
from websockets.sync.server import serve

TRANSCRIPT_DONE = "conversation.item.input_audio_transcription.completed"


def pcm(*values: int) -> np.ndarray:
    return np.array(values, dtype="<i2")


def b64(samples: np.ndarray) -> str:
    return base64.b64encode(samples.astype("<i2").tobytes()).decode()


class Connection:
    """What one client connection did."""

    def __init__(self):
        self.events: list[dict] = []  # every event the client sent, in order
        self.audio = b""  # the PCM16 it appended
        self.session: dict = {}  # the last session.update's session
        self.closed = threading.Event()  # the client went away
        self.finished = threading.Event()  # the script ran to its end

    @property
    def types(self) -> list[str]:
        return [e["type"] for e in self.events]


class FakeRealtime:
    """Scripts are lists of steps, played after response.create (`reply`) or input_audio_buffer.commit
    (`transcription`):

    ("text", str)           response.output_text.delta
    ("audio", int16 array)  response.output_audio.delta
    ("done", status, tokens) response.done (status "completed", tokens 10 unless given)
    ("transcript", str)     the transcription's completed event
    ("error", message)      an error event
    ("close",)              hang up
    ("stall",)              say nothing more, until the client goes away
    ("wait", Event)         pause until the event is set (or the client goes away)
    """

    def __init__(self, reply=None, transcription=None, rates=(16000, 24000)):
        self.reply = reply if reply is not None else [("text", "Hi."), ("audio", pcm(1, 2)), ("done",)]
        self.transcription = transcription if transcription is not None else [("transcript", "hello there")]
        self.rates = rates
        self.connections: list[Connection] = []
        self._sockets: list = []
        self._server = None
        self._thread = None

    # -- lifecycle

    def start(self) -> "FakeRealtime":
        self._server = serve(self._handle, "127.0.0.1", 0)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._server.shutdown()
        for ws in self._sockets:  # a handler thread of a connection nobody closed would keep pytest from exiting
            ws.close()
        self._thread.join(5)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self._server.socket.getsockname()[1]}"

    def conversation(self) -> Connection:
        """The one conversation session (waits for it to be configured)."""
        return self._the_one(lambda session: session.get("type") != "transcription")

    def transcription_session(self) -> Connection:
        return self._the_one(lambda session: session.get("type") == "transcription")

    def _the_one(self, wanted) -> Connection:
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            found = [c for c in self.connections if c.session and wanted(c.session)]  # {} = not configured yet
            if found:
                (conn,) = found
                return conn
            time.sleep(0.01)
        raise AssertionError("no such session")

    # -- the server side of one connection

    def _handle(self, ws) -> None:
        conn = Connection()
        self.connections.append(conn)
        self._sockets.append(ws)
        rate_in, rate_out = self.rates
        try:
            ws.send(
                json.dumps(
                    {
                        "type": "session.created",
                        "session": {
                            "audio": {
                                "input": {"format": {"type": "audio/pcm", "rate": rate_in}, "turn_detection": None},
                                "output": {"format": {"type": "audio/pcm", "rate": rate_out}},
                            }
                        },
                    }
                )
            )
            for raw in ws:
                event = json.loads(raw)
                conn.events.append(event)
                kind = event["type"]
                if kind == "session.update":
                    conn.session = event["session"]
                    ws.send(json.dumps({"type": "session.updated", "session": conn.session}))
                elif kind == "input_audio_buffer.append":
                    conn.audio += base64.b64decode(event["audio"])
                elif kind == "response.create":
                    self._play(ws, conn, self.reply)
                elif kind == "input_audio_buffer.commit":
                    self._play(ws, conn, self.transcription)
        except ConnectionClosed:
            pass
        finally:
            conn.closed.set()

    def _play(self, ws, conn: Connection, steps) -> None:
        for step in steps:
            kind = step[0]
            if kind == "text":
                ws.send(json.dumps({"type": "response.output_text.delta", "delta": step[1]}))
            elif kind == "audio":
                ws.send(json.dumps({"type": "response.output_audio.delta", "delta": b64(step[1])}))
            elif kind == "done":
                status = step[1] if len(step) > 1 else "completed"
                tokens = step[2] if len(step) > 2 else 10
                response = {"status": status, "usage": {"output_tokens": tokens}}
                ws.send(json.dumps({"type": "response.done", "response": response}))
            elif kind == "transcript":
                ws.send(json.dumps({"type": TRANSCRIPT_DONE, "transcript": step[1]}))
            elif kind == "error":
                ws.send(json.dumps({"type": "error", "error": {"message": step[1]}}))
            elif kind == "close":
                ws.close()
                return
            elif kind == "stall":
                conn.finished.set()
                self._until_closed(ws, None)
                return
            elif kind == "wait":
                self._until_closed(ws, step[1])
        conn.finished.set()

    @staticmethod
    def _until_closed(ws, release) -> None:
        """Block until the client goes away, or `release` is set; ConnectionClosed ends the connection."""
        while release is None or not release.is_set():
            with contextlib.suppress(TimeoutError):
                ws.recv(timeout=0.02)



def next_on_fresh_thread(gen, name="fastrtc-worker"):
    """next(gen) on a thread of its own, like fastrtc resumes a reply generator for every chunk.

    Returns the value, or the exception it raised (StopIteration at the end)."""
    box = {}

    def run():
        try:
            box["value"] = next(gen)
        except BaseException as exc:
            box["value"] = exc

    thread = threading.Thread(target=run, name=name)
    thread.start()
    thread.join()
    return box["value"]
