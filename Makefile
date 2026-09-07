PYTHON ?= python
CONFIG ?= configs/baseline.yaml

install:
	$(PYTHON) -m pip install -e .

inspect:
	$(PYTHON) scripts/inspect_schema.py --config $(CONFIG)

features:
	$(PYTHON) scripts/build_features.py --config $(CONFIG)

train:
	$(PYTHON) scripts/train.py --config $(CONFIG)

predict:
	$(PYTHON) scripts/predict.py --config $(CONFIG) --output submission.csv

run:
	$(PYTHON) scripts/run_all.py --config $(CONFIG) --output submission.csv

test:
	pytest -q
