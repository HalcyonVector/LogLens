"""Spark session with settings tuned for a single laptop."""

from __future__ import annotations

import os

from pyspark.sql import SparkSession


def get_spark(app: str = "loglens") -> SparkSession:
    cores = os.cpu_count() or 4
    return (
        SparkSession.builder.appName(app)
        .master(os.environ.get("LOGLENS_MASTER", "local[*]"))
        .config("spark.driver.memory", os.environ.get("LOGLENS_DRIVER_MEMORY", "4g"))
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.sql.shuffle.partitions", str(cores * 2))
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.files.maxPartitionBytes", str(64 * 1024 * 1024))
        .config("spark.ui.showConsoleProgress", "false")
        .getOrCreate()
    )
