"""Small JSON run log per stage. The dashboard's pipeline-health panel reads these."""

from __future__ import annotations

import json
import time
from contextlib import contextmanager
from datetime import datetime, timezone

from loglens.config import RUNLOG


def write(dataset: str, stage: str, record: dict) -> None:
    RUNLOG.mkdir(parents=True, exist_ok=True)
    record = {
        "dataset": dataset,
        "stage": stage,
        "finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        **record,
    }
    path = RUNLOG / f"{dataset}_{stage}.json"
    path.write_text(json.dumps(record, indent=2))
    print(json.dumps(record, indent=2))


@contextmanager
def timer(timings: dict, key: str):
    start = time.perf_counter()
    yield
    timings[key] = round(time.perf_counter() - start, 2)
