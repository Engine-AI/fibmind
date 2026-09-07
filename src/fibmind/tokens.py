"""Token counting behind one small interface.

Budgets are enforced in tokens, not characters, because the cost that matters
is what the model bills and what fits in its window. The default counter is a
deterministic, dependency-free estimate (CJK characters and punctuation count
one each, ASCII runs four characters per token). Pass a real tokenizer through
``TokenCounter`` when the target model is known; the interface is one method.
"""

from __future__ import annotations

import math
import unicodedata
from dataclasses import dataclass
from typing import Callable, Protocol


class TokenCounter(Protocol):
    name: str

    def count(self, text: str) -> int: ...


def _is_cjk(character: str) -> bool:
    codepoint = ord(character)
    return (
        0x3400 <= codepoint <= 0x4DBF
        or 0x4E00 <= codepoint <= 0x9FFF
        or 0xF900 <= codepoint <= 0xFAFF
        or 0x3000 <= codepoint <= 0x303F
        or 0xFF00 <= codepoint <= 0xFFEF
    )


def estimate_tokens(text: str) -> int:
    """Deterministic, model-agnostic token estimate. Relative comparisons only."""
    normalized = unicodedata.normalize("NFKC", text)
    count = 0
    word: list[str] = []

    def flush_word() -> None:
        nonlocal count
        if word:
            count += max(1, math.ceil(len("".join(word)) / 4))
            word.clear()

    for character in normalized:
        if character.isspace():
            flush_word()
        elif _is_cjk(character):
            flush_word()
            count += 1
        elif character.isalnum() or character in {"_", "-"}:
            word.append(character)
        else:
            flush_word()
            count += 1
    flush_word()
    return count


@dataclass(frozen=True, slots=True)
class EstimatingCounter:
    name: str = "estimate"

    def count(self, text: str) -> int:
        return estimate_tokens(text)


@dataclass(frozen=True, slots=True)
class CallableCounter:
    """Wrap any ``str -> int`` tokenizer, e.g. ``lambda t: len(enc.encode(t))``."""

    name: str
    fn: Callable[[str], int]

    def count(self, text: str) -> int:
        return int(self.fn(text))


DEFAULT_COUNTER: TokenCounter = EstimatingCounter()
