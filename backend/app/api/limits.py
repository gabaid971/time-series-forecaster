"""
Abuse protection of the public API: request size, rate limiting per client IP,
and a cap on concurrent trainings (small instance: CPU and memory).
"""

import threading
import time
from collections import defaultdict, deque
from contextlib import contextmanager
from typing import Deque, Dict

from fastapi import HTTPException, Request


def client_ip(request: Request) -> str:
    # Behind Render's proxy, uvicorn --proxy-headers puts the real client IP here
    return request.client.host if request.client else "unknown"


class RateLimiter:
    """Sliding window: at most `max_requests` per client per `window_s` seconds."""

    def __init__(self, max_requests: int, window_s: float = 60.0):
        self.max_requests = max_requests
        self.window_s = window_s
        self._hits: Dict[str, Deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def check(self, key: str) -> None:
        now = time.monotonic()
        with self._lock:
            hits = self._hits[key]
            while hits and hits[0] <= now - self.window_s:
                hits.popleft()
            if len(hits) >= self.max_requests:
                retry_after = int(hits[0] + self.window_s - now) + 1
                raise HTTPException(
                    status_code=429,
                    detail=f"Too many requests: max {self.max_requests} per minute. Retry in {retry_after}s.",
                    headers={"Retry-After": str(retry_after)},
                )
            hits.append(now)

    def reset(self) -> None:
        with self._lock:
            self._hits.clear()


class TrainingSlots:
    """Limit concurrent trainings; waiting requests give up after `timeout_s` (503)."""

    def __init__(self, max_concurrent: int, timeout_s: float):
        self._semaphore = threading.BoundedSemaphore(max_concurrent)
        self.timeout_s = timeout_s

    @contextmanager
    def acquire(self):
        if not self._semaphore.acquire(timeout=self.timeout_s):
            raise HTTPException(status_code=503, detail="Server busy with other trainings, please retry shortly.")
        try:
            yield
        finally:
            self._semaphore.release()
