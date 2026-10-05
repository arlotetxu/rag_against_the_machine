import json
import hashlib
from pathlib import Path
from src.aux.constants import SYSTEM_PROMPT, MAX_OUT_TOKENS, MODEL_NAME
from typing import Any, Optional


class CacheHandler():
    def __init__(self, cache_path: Path) -> None:
        self.cache_path = cache_path

    def make_key(self, query: str, k: Optional[int]) -> str:
        """Build the cache key. Includes k so k=5 and k=10 don't mix."""
        if k:
            return f"{k}::{query}"
        return query

    def make_answer_key(self, query: str, k: int, prompt_context: str) -> str:
        prompt_hash = hashlib.sha256(
            (SYSTEM_PROMPT + prompt_context).encode("utf-8")).hexdigest()[:12]
        return json.dumps([query, k, MODEL_NAME, MAX_OUT_TOKENS, prompt_hash])

    def load_cache(self) -> dict[str, Any]:
        """Read the cache from disk.
        Returns an empty dict if missing or broken."""
        if not self.cache_path.exists():
            return {}
        try:
            with open(self.cache_path, "r", encoding="utf-8") as fd:
                data: dict[str, Any] = json.load(fd)
                return data
        except (OSError, json.JSONDecodeError):
            return {}

    def save_cache(self, cache: dict[str, Any]) -> None:
        """Write the whole cache to disk."""
        try:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.cache_path, "w", encoding="utf-8") as fd:
                json.dump(cache, fd)
        except OSError:
            print("[WARNING] Could not save cache")
