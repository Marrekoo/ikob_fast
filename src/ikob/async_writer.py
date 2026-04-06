"""Background CSV writer – keeps I/O off the computation critical path."""

import logging
import queue
import threading

import numpy as np

import ikob.utils as utils

logger = logging.getLogger(__name__)


class AsyncCsvWriter:
    """Queue-based background CSV writer.

    Submissions are serialised onto *num_workers* daemon threads so the
    caller can continue with the next computation immediately.  Each
    submitted array is **copied** so the caller may safely reuse or
    overwrite its buffer.
    """

    def __init__(self, num_workers: int = 2):
        self._q: queue.Queue = queue.Queue()
        self._workers: list[threading.Thread] = []
        for _ in range(num_workers):
            t = threading.Thread(target=self._drain, daemon=True)
            t.start()
            self._workers.append(t)

    # ── internal ─────────────────────────────────────────────────────

    def _drain(self):
        while True:
            item = self._q.get()
            if item is None:
                break
            data, path, hdr, idx = item
            try:
                utils.write_csv(data, path, header=hdr, index=idx)
            except Exception:
                logger.exception("Async write failed for %s", path)
            finally:
                self._q.task_done()

    # ── public API ───────────────────────────────────────────────────

    def submit(self, data, path, header=None, index=None):
        """Enqueue a CSV write.  *data* is copied immediately."""
        self._q.put((
            np.array(data),
            path,
            list(header) if header else [],
            index if index is not None else utils.CsvIndex(),
        ))

    def join(self):
        """Block until every queued write has completed."""
        self._q.join()

    def shutdown(self):
        """Flush the queue, then stop all worker threads."""
        self.join()
        for _ in self._workers:
            self._q.put(None)
        for w in self._workers:
            w.join()
        self._workers.clear()