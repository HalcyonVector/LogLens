# Run inside the container: docker compose run --rm loglens make <target>
PY := python

.PHONY: download bronze eda split parse-sweep parse parse-eval gold baselines lstm-tune lstm evaluate ablation mlflow-ui test all

download:
	$(PY) scripts/download_data.py HDFS_v1

bronze:
	$(PY) -m loglens.ingest hdfs

eda:
	$(PY) -m loglens.eda hdfs

test:
	$(PY) -m pytest -q tests

all: download bronze eda split parse-sweep parse parse-eval gold baselines lstm-tune lstm evaluate ablation

split:
	$(PY) -m loglens.split hdfs

parse:
	$(PY) -m loglens.parse hdfs --benchmark

parse-eval:
	$(PY) -m loglens.parse_eval hdfs

parse-sweep:
	$(PY) -m loglens.parse_eval hdfs --sweep

gold:
	$(PY) -m loglens.gold hdfs

baselines:
	$(PY) -m loglens.baselines hdfs

lstm-tune:
	$(PY) -m loglens.lstm hdfs --tune

lstm:
	$(PY) -m loglens.lstm hdfs

evaluate:
	$(PY) -m loglens.evaluate hdfs

ablation:
	$(PY) -m loglens.ablation hdfs

# docker compose run --rm --service-ports loglens make mlflow-ui  ->  http://localhost:5000
mlflow-ui:
	mlflow ui --backend-store-uri ./mlruns --host 0.0.0.0 --port 5000
