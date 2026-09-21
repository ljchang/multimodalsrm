"""Small synchronous progress events, without arrays or inference settings."""

import time
from contextlib import contextmanager


class PhaseTimings:
    """Emit phase boundaries; callers synchronize device work before exit."""

    def __init__(self, progress=None):
        if progress is not None and not callable(progress):
            raise TypeError("progress must be callable or None")
        self.progress = progress
        self.seconds = {}

    @contextmanager
    def phase(self, name):
        began = time.monotonic()
        event = dict(phase=name, started_monotonic=began)
        if self.progress is not None:
            self.progress(dict(event, status="started"))
        status = "failed"
        try:
            yield
            status = "completed"
        finally:
            elapsed = time.monotonic() - began
            self.seconds[name] = self.seconds.get(name, 0.0) + elapsed
            if self.progress is not None:
                self.progress(dict(event, status=status, elapsed_seconds=elapsed))
