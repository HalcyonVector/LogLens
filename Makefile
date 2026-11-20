# Run inside the container: docker compose run --rm loglens make <target>
PY := python

.PHONY: download bronze eda test all

download:
	$(PY) scripts/download_data.py HDFS_v1

bronze:
	$(PY) -m loglens.ingest hdfs

eda:
	$(PY) -m loglens.eda hdfs

test:
	$(PY) -m pytest -q tests

all: download bronze eda
