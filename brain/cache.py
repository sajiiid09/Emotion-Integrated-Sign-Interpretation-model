"""Simple TTL cache for Brain responses."""

from __future__ import annotations

import time
from collections import OrderedDict
from typing import Optional


class ResponseCache:
    def __init__(self, max_items: int = 128, ttl_s: float = 30.0) -> None:
        self.max_items = max_items
        self.ttl_s = ttl_s
        self._store: OrderedDict[str, tuple[float, str]] = OrderedDict()

    def _evict_expired(self) -> None:
        now = time.time()
        keys_to_delete = [key for key, (ts, _) in self._store.items() if now - ts > self.ttl_s]
        for key in keys_to_delete:
            self._store.pop(key, None)
        while len(self._store) > self.max_items:
            self._store.popitem(last=False)

    def get(self, key: str) -> Optional[str]:
        self._evict_expired()
        if key in self._store:
            ts, value = self._store.pop(key)
            self._store[key] = (ts, value)
            return value
        return None

    def set(self, key: str, value: str) -> None:
        self._store[key] = (time.time(), value)
        self._evict_expired()


__all__ = ["ResponseCache"]
