"""Masking rules shared by Spark and Drain3, so both see identical text.

Order matters: an IP is masked before the number rule can split it up, and a
path is masked whole before the block rule can match a block ID inside it.
Every pattern is valid in both Java (Spark) and Python (Drain3) regex syntax.
"""

from __future__ import annotations

import re

from pyspark.sql import Column, functions as F

MASKS: list[tuple[str, str]] = [
    ("IP", r"/?\d{1,3}(?:\.\d{1,3}){3}(?::\d+)?"),
    ("PATH", r"(?<![\w.])/[\w./\-]+"),
    ("BLK", r"blk_-?\d+"),
    ("NUM", r"(?<![\w<])-?\d+(?![\w>])"),
]

BLOCK_RE = r"blk_-?\d+"

_COMPILED = [(f"<{name}>", re.compile(p)) for name, p in MASKS]


def mask_text(text: str) -> str:
    for token, rx in _COMPILED:
        text = rx.sub(token, text)
    return " ".join(text.split())


def mask_col(col: str | Column) -> Column:
    c = F.col(col) if isinstance(col, str) else col
    for name, pattern in MASKS:
        c = F.regexp_replace(c, pattern, f"<{name}>")
    return F.trim(F.regexp_replace(c, r"\s+", " "))


def block_ids(col: str | Column) -> Column:
    """Distinct block IDs named in a message, as an array."""
    c = F.col(col) if isinstance(col, str) else col
    return F.array_distinct(F.regexp_extract_all(c, F.lit(BLOCK_RE), F.lit(0)))
