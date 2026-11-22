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

# ---- BGL: one bronze log, two windowing schemes (1-hour and 20-event windows), each with its own parser
BGL_SCHEMES := bgl_1h bgl_20ev bgl_20ev_allfit
# The whole-log-parser scheme reuses bgl_20ev's threshold, so it is not swept.
BGL_SWEPT := bgl_1h bgl_20ev
.PHONY: bgl download-bgl bronze-bgl split-bgl parse-sweep-bgl parse-bgl parse-eval-bgl gold-bgl baselines-bgl drift-bgl lstm-bgl

download-bgl:
	$(PY) scripts/download_data.py BGL BGL_loghub2

bronze-bgl:
	$(PY) -m loglens.ingest bgl

split-bgl:
	for s in $(BGL_SCHEMES); do $(PY) -m loglens.split $$s || exit 1; done

parse-sweep-bgl:
	for s in $(BGL_SWEPT); do $(PY) -m loglens.parse_eval $$s --sweep || exit 1; done

parse-bgl:
	for s in $(BGL_SCHEMES); do $(PY) -m loglens.parse $$s || exit 1; done

parse-eval-bgl:
	for s in $(BGL_SCHEMES); do $(PY) -m loglens.parse_eval $$s || exit 1; done

gold-bgl:
	for s in $(BGL_SCHEMES); do $(PY) -m loglens.gold $$s || exit 1; done

baselines-bgl:
	for s in $(BGL_SCHEMES); do $(PY) -m loglens.baselines $$s || exit 1; done

drift-bgl:
	$(PY) -m loglens.drift bgl

lstm-bgl:
	for s in $(BGL_SCHEMES); do $(PY) -m loglens.lstm $$s || exit 1; done

bgl: download-bgl bronze-bgl split-bgl parse-sweep-bgl parse-bgl parse-eval-bgl gold-bgl baselines-bgl drift-bgl lstm-bgl

# ---- Scale benchmark: HDFS Spark vs pandas and volume vs bind mount, then Thunderbird on the volume
.PHONY: scale download-thunderbird
download-thunderbird:
	LOGLENS_DATA=/scale/data $(PY) scripts/download_data.py Thunderbird

scale:
	$(PY) -m loglens.scale hdfs
	for f in 0.01 0.1 1.0; do $(PY) -m loglens.pandas_baseline hdfs $$f || exit 1; done
	$(PY) -m loglens.scale hdfs --fractions=1.0 --bind
	LOGLENS_DATA=/scale/data LOGLENS_DRIVER_MEMORY=4g $(PY) -m loglens.scale thunderbird
