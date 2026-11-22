"""Masking rules shared by Spark and Drain3, so both see identical text.

Each dataset has its own ordered rules. Order matters: an IP is masked before
the number rule can split it up, and a path is masked whole before the block
rule can match a block ID inside it.
Every pattern is valid in both Java (Spark) and Python (Drain3) regex syntax.
"""

from __future__ import annotations

import re

from pyspark.sql import Column, functions as F

from loglens.config import parent

MASKS_BY_DATASET: dict[str, list[tuple[str, str]]] = {
    "hdfs": [
        ("IP", r"/?\d{1,3}(?:\.\d{1,3}){3}(?::\d+)?"),
        ("PATH", r"(?<![\w.])/[\w./\-]+"),
        ("BLK", r"blk_-?\d+"),
        ("NUM", r"(?<![\w<])-?\d+(?![\w>])"),
    ],
    "bgl": [
        # Blue Gene/L locations such as R02-M1-N0-C:J12-U11, masked before numbers split them.
        ("NODE", r"R\d{2}-M\d[\w:\-]*"),
        ("IP", r"/?\d{1,3}(?:\.\d{1,3}){3}(?::\d+)?"),
        ("PATH", r"(?<![\w.])/[\w./\-]+"),
        ("HEX", r"0x[0-9a-fA-F]+"),
        # Bare 8-digit addresses such as "iar 00106210 dear 0244c20c".
        ("HEX", r"(?<![\w<])[0-9a-fA-F]{8}(?![\w>])"),
        ("NUM", r"(?<![\w<])-?\d+(?![\w>])"),
    ],
    "thunderbird": [
        ("IP", r"/?\d{1,3}(?:\.\d{1,3}){3}(?::\d+)?"),
        ("PATH", r"(?<![\w.])/[\w./\-]+"),
        ("HEX", r"0x[0-9a-fA-F]+"),
        # Thunderbird hosts: an214, cn949, aadmin1, dadmin3, tbird-admin1, tbird-sm1.
        ("NODE", r"(?<![\w-])(?:tbird-[a-z]+|[a-d]admin|[a-e]n)\d+(?![\w])"),
        ("NUM", r"(?<![\w<])-?\d+(?![\w>])"),
    ],
}
# The HDFS rules, kept under the old name for callers that predate other datasets.
MASKS = MASKS_BY_DATASET["hdfs"]

BLOCK_RE = r"blk_-?\d+"

_COMPILED = {
    ds: [(f"<{name}>", re.compile(p)) for name, p in masks] for ds, masks in MASKS_BY_DATASET.items()
}


def masks_for(dataset: str) -> list[tuple[str, str]]:
    return MASKS_BY_DATASET[parent(dataset)]


def mask_text(text: str, dataset: str = "hdfs") -> str:
    for token, rx in _COMPILED[parent(dataset)]:
        text = rx.sub(token, text)
    return " ".join(text.split())


def mask_col(col: str | Column, dataset: str = "hdfs") -> Column:
    c = F.col(col) if isinstance(col, str) else col
    for name, pattern in masks_for(dataset):
        c = F.regexp_replace(c, pattern, f"<{name}>")
    return F.trim(F.regexp_replace(c, r"\s+", " "))


def block_ids(col: str | Column) -> Column:
    """Distinct block IDs named in a message, as an array."""
    c = F.col(col) if isinstance(col, str) else col
    return F.array_distinct(F.regexp_extract_all(c, F.lit(BLOCK_RE), F.lit(0)))
