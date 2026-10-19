"""Pluggable embedding providers for semantic recall.

Retrieval works without any of this: when no provider is configured the engine
is purely lexical. A provider adds a second candidate list (nearest vectors)
that fusion merges with the lexical one. Provider failures are contained — a
timeout or a bad response degrades recall to lexical-only instead of raising.

Vectors are cached by content hash, so unchanged memories are never re-embedded
within a process. Persistence of that cache is a later slice.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import sqlite3
import struct
import urllib.error
import urllib.request
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

from fibmind.ranking import tokenize

ENV_URL = "FIBMIND_EMBEDDING_URL"
ENV_MODEL = "FIBMIND_EMBEDDING_MODEL"
ENV_API_KEY = "FIBMIND_EMBEDDING_API_KEY"
ENV_PROVIDER = "FIBMIND_EMBEDDING"  # "off" (default), "hashing", "openai"


class EmbeddingProvider(Protocol):
    """One call, many texts, one vector each. ``name`` keys the cache.

    ``min_similarity`` is the cosine below which this provider's neighbours are
    noise. It is provider-specific: sparse hashed features score a real match
    near 0.15, dense API embeddings score unrelated text near 0.7.
    """

    name: str
    min_similarity: float

    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    # math.sumprod would be faster but needs Python 3.12; we support 3.11.
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return dot / norm if norm else 0.0


@dataclass
class HashingEmbeddingProvider:
    """Deterministic, dependency-free vectors from hashed token features.

    Not semantic: it knows nothing about meaning. It exists so the whole vector
    path — caching, fusion, fallback — is exercised and testable offline, and it
    gives a little fuzziness through character n-grams (``retry`` and
    ``retries`` share most of theirs). Swap in a real provider for meaning.
    """

    dimensions: int = 256
    min_similarity: float = 0.25
    name: str = field(default="hashing-256", init=False)

    def __post_init__(self) -> None:
        self.name = f"hashing-{self.dimensions}"

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._one(text) for text in texts]

    def _one(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        for feature, weight in self._features(text):
            digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
            index = int.from_bytes(digest[:4], "little") % self.dimensions
            sign = 1.0 if digest[4] & 1 else -1.0
            vector[index] += sign * weight
        norm = math.sqrt(sum(value * value for value in vector))
        return [value / norm for value in vector] if norm else vector

    @staticmethod
    def _features(text: str) -> list[tuple[str, float]]:
        features: list[tuple[str, float]] = []
        for token in tokenize(text):
            features.append((f"t:{token}", 1.0))
            if len(token) >= 4 and token.isascii():
                features.append((f"p:{token[:4]}", 0.8))
            if len(token) >= 5 and token.isascii():
                padded = f"#{token}#"
                for index in range(len(padded) - 2):
                    features.append((f"g:{padded[index : index + 3]}", 0.35))
        return features


@dataclass
class OpenAICompatibleEmbeddingProvider:
    """POST ``{url}/embeddings`` in the OpenAI request/response shape.

    Works with OpenAI, DeepSeek-compatible gateways, Ollama's ``/v1``, and most
    local servers. Standard library only.
    """

    url: str
    model: str
    api_key: str | None = None
    timeout: float = 20.0
    min_similarity: float = 0.75
    name: str = field(default="", init=False)

    def __post_init__(self) -> None:
        self.url = self.url.rstrip("/")
        self.name = f"openai:{self.model}"

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        body = json.dumps({"model": self.model, "input": list(texts)}).encode("utf-8")
        request = urllib.request.Request(
            f"{self.url}/embeddings",
            data=body,
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        if self.api_key:
            request.add_header("Authorization", f"Bearer {self.api_key}")
        with urllib.request.urlopen(request, timeout=self.timeout) as response:  # noqa: S310
            payload = json.loads(response.read().decode("utf-8"))
        rows = sorted(payload["data"], key=lambda item: item.get("index", 0))
        vectors = [[float(value) for value in row["embedding"]] for row in rows]
        if len(vectors) != len(texts):
            raise ValueError("embedding response count does not match input count")
        return vectors


class EmbeddingCache:
    """Vectors keyed by (provider, content hash). Failures are recorded, not raised.

    Can be backed by persistent SQLite storage when ``db_path`` is provided.
    """

    def __init__(
        self,
        provider: EmbeddingProvider | None,
        *,
        db_path: str | Path | None = None,
    ) -> None:
        self.provider = provider
        self.db_path = Path(db_path).resolve() if db_path else None
        self._vectors: dict[tuple[str, str], list[float]] = {}
        self.failures = 0
        self.last_error: str | None = None
        if self.db_path:
            self._init_db()

    def _connect(self) -> sqlite3.Connection:
        assert self.db_path is not None
        conn = sqlite3.connect(self.db_path, timeout=5.0)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=5000")
        return conn

    def _init_db(self) -> None:
        if not self.db_path:
            return
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with self._connect() as conn:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS vector_cache (
                        content_hash TEXT NOT NULL,
                        provider TEXT NOT NULL,
                        dimensions INTEGER NOT NULL,
                        vector BLOB NOT NULL,
                        created_at TEXT NOT NULL,
                        PRIMARY KEY(provider, content_hash)
                    );
                    """
                )
                conn.commit()
        except sqlite3.OperationalError:
            pass

    @property
    def enabled(self) -> bool:
        return self.provider is not None

    def vectors_for(self, texts: Sequence[str]) -> list[list[float] | None]:
        """Vectors for ``texts`` in order; ``None`` where embedding failed."""
        if self.provider is None:
            return [None] * len(texts)
        hashes = [content_hash(text) for text in texts]

        # 1. Check in-memory cache first
        missing_indices: list[int] = []
        for index, digest in enumerate(hashes):
            if (self.provider.name, digest) not in self._vectors:
                missing_indices.append(index)

        # 2. Try loading missing vectors from SQLite if persistent db is configured
        if missing_indices and self.db_path:
            try:
                missing_hashes = list({hashes[i] for i in missing_indices})
                placeholders = ",".join("?" for _ in missing_hashes)
                with self._connect() as conn:
                    rows = conn.execute(
                        f"SELECT content_hash, vector FROM vector_cache WHERE provider = ? AND content_hash IN ({placeholders})",
                        [self.provider.name, *missing_hashes],
                    ).fetchall()
                    for chash, blob in rows:
                        dim = len(blob) // 4
                        vec = list(struct.unpack(f"{dim}f", blob))
                        self._vectors[(self.provider.name, chash)] = vec
            except (sqlite3.OperationalError, struct.error):
                pass

            # Recompute still-missing indices
            missing_indices = [i for i in missing_indices if (self.provider.name, hashes[i]) not in self._vectors]

        # 3. Call provider for any remaining missing texts
        if missing_indices:
            missing_texts = [texts[i] for i in missing_indices]
            try:
                fresh = self.provider.embed(missing_texts)
            except (urllib.error.URLError, OSError, ValueError, KeyError, TimeoutError) as exc:
                self.failures += 1
                self.last_error = f"{type(exc).__name__}: {exc}"
                return [self._vectors.get((self.provider.name, digest)) for digest in hashes]

            to_persist: list[tuple[str, str, int, bytes, str]] = []
            now_iso = datetime.now(UTC).isoformat()
            for idx, vector in zip(missing_indices, fresh, strict=False):
                chash = hashes[idx]
                self._vectors[(self.provider.name, chash)] = vector
                if self.db_path:
                    blob = struct.pack(f"{len(vector)}f", *vector)
                    to_persist.append((chash, self.provider.name, len(vector), blob, now_iso))

            if to_persist and self.db_path:
                try:
                    with self._connect() as conn:
                        conn.executemany(
                            """
                            INSERT INTO vector_cache (content_hash, provider, dimensions, vector, created_at)
                            VALUES (?, ?, ?, ?, ?)
                            ON CONFLICT(provider, content_hash) DO NOTHING
                            """,
                            to_persist,
                        )
                        conn.commit()
                except sqlite3.OperationalError:
                    pass

        return [self._vectors.get((self.provider.name, digest)) for digest in hashes]

    def __len__(self) -> int:
        return len(self._vectors)


def provider_from_env(environ: dict[str, str] | None = None) -> EmbeddingProvider | None:
    """Pick a provider from ``FIBMIND_EMBEDDING*``; ``None`` means lexical only."""
    env = os.environ if environ is None else environ
    choice = env.get(ENV_PROVIDER, "off").strip().casefold()
    if choice in {"", "off", "none", "0", "false"}:
        return None
    if choice == "hashing":
        return HashingEmbeddingProvider()
    if choice == "openai":
        url = env.get(ENV_URL, "").strip()
        model = env.get(ENV_MODEL, "").strip()
        if not url or not model:
            raise ValueError(f"{ENV_PROVIDER}=openai needs {ENV_URL} and {ENV_MODEL}")
        return OpenAICompatibleEmbeddingProvider(url=url, model=model, api_key=env.get(ENV_API_KEY) or None)
    raise ValueError(f"unknown {ENV_PROVIDER} value: {choice!r}")
