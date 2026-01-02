"""Simple TTL cache for Brain responses."""

from __future__ import annotations

import hashlib
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class CachedItem:
    """Cached response with metadata."""
    text: str
    next_topics: list[str]
    timestamp: float


def make_cache_key(proto_bn: str, mode: str, resolved_emotion: str) -> str:
    """Create a stable cache key from proto sentence, mode, and emotion.

    Uses SHA1 hash for deterministic, collision-resistant keys.
    """
    combined = f"{proto_bn}|{mode}|{resolved_emotion}"
    return hashlib.sha1(combined.encode("utf-8")).hexdigest()


class ResponseCache:
    def __init__(self, max_items: int = 128, ttl_s: float = 30.0) -> None:
        self.max_items = max_items
        self.ttl_s = ttl_s
        self._store: OrderedDict[str, CachedItem] = OrderedDict()

    def _evict_expired(self) -> None:
        now = time.time()
        keys_to_delete = [key for key, item in self._store.items() if now - item.timestamp > self.ttl_s]
        for key in keys_to_delete:
            self._store.pop(key, None)
        while len(self._store) > self.max_items:
            self._store.popitem(last=False)

    def get(self, key: str) -> Optional[tuple[str, list[str]]]:
        """Get cached response and topics, returns (text, topics) or None."""
        self._evict_expired()
        if key in self._store:
            item = self._store.pop(key)
            self._store[key] = item
            return item.text, item.next_topics
        return None

    def set(self, key: str, value: str, next_topics: list[str] | None = None) -> None:
        """Cache a response with optional next topics."""
        topics = next_topics or []
        item = CachedItem(text=value, next_topics=topics, timestamp=time.time())
        self._store[key] = item
        self._evict_expired()


__all__ = ["ResponseCache", "CachedItem", "make_cache_key"]
