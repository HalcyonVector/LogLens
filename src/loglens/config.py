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

# Raw file per dataset, and the regex that splits one line into fields.
# Lines that do not match go to a quarantine table instead of being dropped.
DATASETS = {
    "hdfs": {
        "raw": RAW / "HDFS_v1" / "HDFS.log",
        "labels": RAW / "HDFS_v1" / "preprocessed" / "anomaly_label.csv",
        # 081109 203615 148 INFO dfs.DataNode$PacketResponder: PacketResponder 1 for block blk_38865049064139660 terminating
        "regex": r"^(\d{6}) (\d{6}) (\d+) ([A-Z]+) ([^:\s]+): (.*)$",
        "fields": ["date", "time", "pid", "level", "component", "message"],
        "ts_format": "yyMMddHHmmss",
        "partition": "hour",
        "expected_lines": 11_175_629,
    },
}
