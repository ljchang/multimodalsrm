"""Independent MAP threads with caller-thread progress and cooperative cleanup."""

from concurrent.futures import CancelledError, ThreadPoolExecutor
from queue import Queue
from threading import Event


def ordered_restarts(run_start, points, workers, progress):
    """Keep scheduling outside JAX tracing and callbacks outside worker threads.

    Worker failures and callback failures propagate. Cancelled searches wait for
    in-flight numerical evaluations, then stop at the next evaluation boundary.
    No worker can continue fitting after this function returns or raises.
    """
    stop = Event()

    def check_cancelled():
        if stop.is_set():
            raise CancelledError("MAP search cancelled")

    if workers == 1:
        return [run_start(i, point, progress, check_cancelled) for i, point in enumerate(points)]

    messages = Queue()

    def emit(event):
        check_cancelled()
        messages.put(("progress", event))

    pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="gp-map")
    futures = []
    records = [None] * len(points)
    try:
        for index, point in enumerate(points):
            future = pool.submit(
                run_start,
                index,
                point,
                emit if progress is not None else None,
                check_cancelled,
            )
            futures.append(future)
            future.add_done_callback(lambda done, i=index: messages.put(("finished", (i, done))))
        remaining = len(futures)
        while remaining:
            kind, payload = messages.get()
            if kind == "progress":
                progress(payload)
            else:
                index, future = payload
                records[index] = future.result()
                remaining -= 1
    finally:
        stop.set()
        pool.shutdown(wait=True, cancel_futures=True)
    return records
