"""The app's lifecycle: the signal that the app is stopping."""
import signal
import threading

# Set when the app is asked to stop (Ctrl+C). The native backend uses it to give up on a reply
# instead of waiting for a model call that may never return: the process would otherwise hang
# at exit, because the thread that waits for the model is not a daemon thread.
stopping = threading.Event()


def request_stop() -> None:
    """Tell the app's parts to stop as soon as they can."""
    stopping.set()


def install_stop_handler() -> None:
    """Make Ctrl+C ask the app to stop: gradio closes the server, the responders give up.

    Gradio does not install a SIGINT handler of its own (its uvicorn server runs in a thread
    and `install_signal_handlers` is a no-op), so this stays in charge.
    """

    def on_sigint(signum, frame):
        request_stop()
        raise KeyboardInterrupt  # what Python does by default; gradio closes the server

    signal.signal(signal.SIGINT, on_sigint)