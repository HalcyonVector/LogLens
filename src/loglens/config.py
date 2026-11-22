"""Paths and dataset-level settings shared across pipeline stages."""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = Path(os.environ.get("LOGLENS_DATA", ROOT / "data"))

RAW = DATA / "raw"
BRONZE = DATA / "bronze"
SILVER = DATA / "silver"
GOLD = DATA / "gold"
RUNLOG = DATA / "runlog"
DASHBOARD = DATA / "dashboard"

# One entry per raw log. Lines that do not match `regex` go to a quarantine
# table instead of being dropped. `ts` names the field(s) that hold the
# timestamp and their format; `keep` lists extra fields carried into bronze.
DATASETS = {
    "hdfs": {
        "raw": RAW / "HDFS_v1" / "HDFS.log",
        "labels": RAW / "HDFS_v1" / "preprocessed" / "anomaly_label.csv",
        # 081109 203615 148 INFO dfs.DataNode$PacketResponder: PacketResponder 1 for block blk_38865049064139660 terminating
        "regex": r"^(\d{6}) (\d{6}) (\d+) ([A-Z]+) ([^:\s]+): (.*)$",
        "fields": ["date", "time", "pid", "level", "component", "message"],
        "ts": {"cols": ["date", "time"], "format": "yyMMddHHmmss"},
        "keep": ["pid"],
        "partition": "hour",
        "expected_lines": 11_175_629,
        "masks": "hdfs",
        # Sessions are HDFS blocks, named in the message; labels come from a file.
        "session": {"kind": "block"},
        "session_col": "block_id",
        "alarm_levels": ["WARN"],
        # Drain3 similarity threshold, chosen by `parse_eval --sweep` on
        # validation-period grouping accuracy (never on test).
        "drain_sim_th": 0.9,
    },
    "bgl": {
        "raw": RAW / "BGL" / "BGL.log",
        # - 1117838570 2005.06.03 R02-M1-N0-C:J12-U11 2005-06-03-15.42.50.675872 R02-M1-N0-C:J12-U11 RAS KERNEL INFO instruction cache parity error corrected
        # The first field is the alert tag ("-" for a normal line); the message can be empty.
        "regex": r"^(\S+) (\d+) (\S+) (\S+) (\S+) (\S+) (\S+) (\S+) (\S+)(?: (.*))?$",
        "fields": ["alert", "epoch", "date", "node", "time", "node_repeat", "type", "component", "level", "message"],
        "ts": {"cols": ["time"], "format": "yyyy-MM-dd-HH.mm.ss.SSSSSS"},
        "keep": ["alert", "node"],
        "partition": "day",
        "expected_lines": 4_747_963,
        "expected_alerts": 348_460,
        "masks": "bgl",
        "ground_truth": RAW / "BGL_loghub2",
        "alarm_levels": ["FATAL", "FAILURE"],
        # BGL has no natural sessions: each windowing scheme below is its own
        # gold table and model run, sharing this dataset's bronze and silver.
        "schemes": ["bgl_1h", "bgl_20ev", "bgl_20ev_allfit"],
    },
    "thunderbird": {
        # Used only by the scale benchmark (python -m loglens.scale thunderbird), with
        # LOGLENS_DATA=/scale/data so the 32 GB log is read from the Docker volume.
        "raw": RAW / "Thunderbird" / "Thunderbird.log",
        # - 1131523501 2005.11.09 aadmin1 Nov 10 00:05:01 src@aadmin1 in.tftpd[14620]: tftp: client does not accept options
        # The date field can disagree with the month/day that follow, so time comes from the epoch field.
        "regex": r"^(\S+) (\d+) (\S+) (\S+) (\w{3})\s+(\d+) (\S+) (\S+)(?: (.*))?$",
        "fields": ["alert", "epoch", "date", "node", "month", "day", "time", "location", "message"],
        "ts": {"cols": ["epoch"], "format": "unix"},
        "expected_lines": 211_212_192,
        "masks": "thunderbird",
        "session": {"kind": "time", "seconds": 3600},
        "session_col": "session_id",
        # Templates are learned inside each scale run, on the first 60% of that run's input.
        "scale_fit": True,
        "drain_sim_th": 0.7,
    },
}

# Windowing schemes for logs without natural sessions. Each behaves like a
# dataset from the split stage on; `parent` names the log it windows.
SCHEMES = {
    # drain_sim_th: chosen per scheme by `parse_eval <scheme> --sweep` on validation grouping accuracy.
    "bgl_1h": {"parent": "bgl", "session": {"kind": "time", "seconds": 3600}, "session_col": "session_id", "drain_sim_th": 0.8},
    "bgl_20ev": {"parent": "bgl", "session": {"kind": "count", "size": 20}, "session_col": "session_id", "drain_sim_th": 0.7},
    # Same windows and split as bgl_20ev, but the parser is fitted on the whole log,
    # as log benchmarks usually do: measures how much that flatters the results.
    "bgl_20ev_allfit": {
        "parent": "bgl", "session": {"kind": "count", "size": 20}, "session_col": "session_id",
        "parser_fit": "all", "drain_sim_th": 0.7,
    },
}


def settings(name: str) -> dict:
    """Settings for a dataset or a windowing scheme (merged with its parent's)."""
    if name in DATASETS:
        return DATASETS[name]
    scheme = SCHEMES[name]
    return {**DATASETS[scheme["parent"]], **scheme, "name": name}


def parent(name: str) -> str:
    """The raw log a dataset or scheme reads from (its own name for a dataset)."""
    return SCHEMES[name]["parent"] if name in SCHEMES else name


def session_col(name: str) -> str:
    return settings(name)["session_col"]
