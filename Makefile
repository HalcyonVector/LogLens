# Run inside the container: docker compose run --rm loglens make <target>
PY := python

.PHONY: download bronze eda split parse-sweep parse parse-eval test all

download:
	$(PY) scripts/download_data.py HDFS_v1

bronze:
	$(PY) -m loglens.ingest hdfs

eda:
	$(PY) -m loglens.eda hdfs

test:
	$(PY) -m pytest -q tests

all: download bronze eda split parse-sweep parse parse-eval

split:
	$(PY) -m loglens.split hdfs

parse:
	$(PY) -m loglens.parse hdfs --benchmark

parse-eval:
	$(PY) -m loglens.parse_eval hdfs

parse-sweep:
	$(PY) -m loglens.parse_eval hdfs --sweep
