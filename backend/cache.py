"""
Simple TTL-based cache for indicator results.
Uses Streamlit session_state as backend (zero dependencies).
Falls back to in-memory dict outside Streamlit.
"""

import time
import hashlib
import json
import logging
from typing import Any, Optional, Callable

logger = logging.getLogger(__name__)

DEFAULT_TTL = 300


class IndicatorCache:
    """
    TTL cache that stores computed indicator results.
    Works both inside Streamlit (uses st.session_state) and standalone.
    """

    def __init__(self, ttl_seconds: int = DEFAULT_TTL):
        self.ttl = ttl_seconds
        self._memory: dict = {}  # fallback

    def _store(self) -> dict:
        """Get the underlying store (session_state or memory dict)."""
        try:
            import streamlit as st
            if "_xapi_cache" not in st.session_state:
                st.session_state["_xapi_cache"] = {}
            return st.session_state["_xapi_cache"]
        except Exception:
            return self._memory

    def _make_key(self, prefix: str, **kwargs) -> str:
        payload = json.dumps(kwargs, sort_keys=True, default=str)
        h = hashlib.md5(payload.encode()).hexdigest()[:8]
        return f"{prefix}:{h}"

    def get(self, key: str) -> Optional[Any]:
        store = self._store()
        entry = store.get(key)
        if entry is None:
            return None
        if time.time() - entry["ts"] > self.ttl:
            del store[key]
            logger.debug(f"Cache expired: {key}")
            return None
        logger.debug(f"Cache hit: {key}")
        return entry["value"]

    def set(self, key: str, value: Any) -> None:
        self._store()[key] = {"value": value, "ts": time.time()}
        logger.debug(f"Cache set: {key}")

    def invalidate(self, key: str) -> None:
        store = self._store()
        if key in store:
            del store[key]

    def clear_all(self) -> None:
        store = self._store()
        store.clear()
        logger.info("Cache cleared")

    def cached(self, key: str, fn: Callable, *args, **kwargs) -> Any:
        """Get from cache or compute and store."""
        result = self.get(key)
        if result is not None:
            return result
        result = fn(*args, **kwargs)
        self.set(key, result)
        return result

    def stats(self) -> dict:
        store = self._store()
        now = time.time()
        alive = sum(1 for v in store.values() if now - v["ts"] <= self.ttl)
        return {"total_keys": len(store), "alive_keys": alive, "ttl_seconds": self.ttl}


# Singleton instance
cache = IndicatorCache(ttl_seconds=DEFAULT_TTL)
