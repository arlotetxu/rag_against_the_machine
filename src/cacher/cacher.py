import json
from pathlib import Path
from typing import Any, Optional


class CacheHandler():
    def __init__(self, cache_path: Path) -> None:
        self.cache_path = cache_path

    def make_key(self, query: str, k: Optional[int]) -> str:
        """Build the cache key. Includes k so k=5 and k=10 don't mix."""
        if k:
            return f"{k}::{query}"
        return query

    def load_cache(self) -> dict[str, list[dict[str, Any]]]:
        """Read the cache from disk.
        Returns an empty dict if missing or broken."""
        if not self.cache_path.exists():
            return {}
        try:
            with open(self.cache_path, "r", encoding="utf-8") as fd:
                data: dict[str, list[dict[str, Any]]] = json.load(fd)
                return data
        except (OSError, json.JSONDecodeError):
            return {}

    def save_cache(self, cache: dict[str, list[dict[str, Any]]]) -> None:
        """Write the whole cache to disk."""
        try:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.cache_path, "w", encoding="utf-8") as fd:
                json.dump(cache, fd)
        except OSError:
            print("[WARNING] Could not save cache")
