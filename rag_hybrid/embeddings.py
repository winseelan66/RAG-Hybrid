from __future__ import annotations

import hashlib
import math
import re

from rag_hybrid.config import get_settings


EMBEDDING_DIMENSION = get_settings().extraction.embedding_dimension


TOKEN_PATTERN = re.compile(r"\b[a-zA-Z0-9]+\b")


def _hash_token(token: str) -> tuple[int, float]:
    digest = hashlib.md5(token.encode("utf-8")).digest()
    index = int.from_bytes(digest[:4], byteorder="big") % EMBEDDING_DIMENSION
    sign = 1.0 if digest[4] % 2 == 0 else -1.0
    return index, sign


def embed_text(text: str) -> list[float]:
    vector = [0.0] * EMBEDDING_DIMENSION

    for token in TOKEN_PATTERN.findall(text.lower()):
        index, sign = _hash_token(token)
        vector[index] += sign

    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return vector

    return [value / norm for value in vector]
