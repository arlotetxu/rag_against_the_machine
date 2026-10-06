"""JSON file caches for search results and generated answers."""
import json
import hashlib
from pathlib import Path
from src.aux.constants import SYSTEM_PROMPT, MAX_OUT_TOKENS, MODEL_NAME
from typing import Any, Optional


class CacheHandler():
    """Build keys for one JSON cache file, and load and save that file.

    The handler does not keep the cache in memory: callers get the dict
    from ``load_cache``, look keys up and add entries themselves, and pass
    the dict back to ``save_cache``. The same class backs both the search
    cache and the answer cache, each with its own file.
    """

    def __init__(self, cache_path: Path) -> None:
        """Set the cache file to work with.

        Args:
            cache_path (Path): JSON file holding the cache. It does not need
                to exist yet.
        """
        self.cache_path = cache_path

    def make_key(self, query: str, k: Optional[int]) -> str:
        """Build the search cache key for a query.

        ``k`` is part of the key so that results for different ``k`` values
        do not mix.

        Args:
            query (str): Question to search for.
            k (Optional[int]): Number of sources retrieved. If it is ``None``
                or ``0``, the key is the bare query.

        Returns:
            str: ``"<k>::<query>"``, or ``query`` when ``k`` is falsy.
        """
        if k:
            return f"{k}::{query}"
        return query

    def make_answer_key(self, query: str, k: int, prompt_context: str) -> str:
        """Build the answer cache key for a query.

        Besides the query and ``k``, the key holds everything that changes
        the generated answer: ``MODEL_NAME``, ``MAX_OUT_TOKENS`` and a short
        SHA-256 hash of ``SYSTEM_PROMPT`` plus the retrieved context. A
        change to any of them makes the lookup miss instead of returning a
        stale answer.

        Args:
            query (str): Question being answered.
            k (int): Number of sources retrieved for the context.
            prompt_context (str): Retrieved chunks, formatted as they go into
                the prompt.

        Returns:
            str: JSON array of those values, used as the dict key.
        """
        prompt_hash = hashlib.sha256(
            (SYSTEM_PROMPT + prompt_context).encode("utf-8")).hexdigest()[:12]
        return json.dumps([query, k, MODEL_NAME, MAX_OUT_TOKENS, prompt_hash])

    def load_cache(self) -> dict[str, Any]:
        """Read the cache from disk.

        Returns:
            dict[str, Any]: Cached entries by key. Empty if the file does not
            exist, cannot be read or is not valid JSON.
        """
        if not self.cache_path.exists():
            return {}
        try:
            with open(self.cache_path, "r", encoding="utf-8") as fd:
                data: dict[str, Any] = json.load(fd)
                return data
        except (OSError, json.JSONDecodeError):
            return {}

    def save_cache(self, cache: dict[str, Any]) -> None:
        """Write the whole cache to disk, replacing the file.

        Creates the parent folder if needed. A write error does not raise:
        it prints a warning and the cache is simply not saved.

        Args:
            cache (dict[str, Any]): Every entry to store, not only new ones.
        """
        try:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.cache_path, "w", encoding="utf-8") as fd:
                json.dump(cache, fd)
        except OSError:
            print("[WARNING] Could not save cache")
