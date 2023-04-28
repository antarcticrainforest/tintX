# makefile used for testing
#
#
export DATA_FILES := $(PWD)/docs/source/_static/data
all: install test_notebooks test_coverage

.PHONY: docs
install:
	python3 -m pip install -U -e .[tests,docs]
	python3 -m bash_kernel.install

test:
	python3 -m pytest -vv $(PWD)/src/tintx/test
	rm -rf '='

test_notebooks:
	python3 -m pytest -vv --nbval-lax $(PWD)/docs/source
test_coverage:
	python3 -m pytest -vv \
		--cov=$(PWD)/src/tintx --cov-report html:coverage_report \
		--cov-report xml --junitxml report.xml
	rm -rf '='
	python3 -m coverage report

docs:
	make -C docs clean
	make -C docs html


lint:
	flake8 src/tintx --count --select=E9,F63,F7,F82 --show-source --statistics
	flake8 src/tintx --ignore E203 --count --exit-zero --max-complexity=15 --max-line-length=127 --statistics
	black --check -t py311 -l 82 src/tintx
	isort --check --profile black -t py311 -l 82 src/tintx
	mypy
