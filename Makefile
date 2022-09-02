# makefile used for testing
#
#
export DATA_FILES := $(PWD)/docs/source/_static/data
all: install test

.PHONY: docs
install:
	python3 -m pip install -e .[tests,docs]
	python -m bash_kernel.install

test:
	python3 -m pytest -vv $(PWD)/src/tintx/test
	rm -rf '='

test_coverage:
	python3 -m pytest -vv \
	    --cov=$(PWD)/src/tintx --cov-report html:coverage_report \
		--nbval-lax --current-env --cov-report=xml --junitxml report.xml
	rm -rf '='
	python3 -m coverage report

docs:
	make -C docs clean
	make -C docs html


lint:
	mypy
	black --check -t py310 -l 82 src
	flake8 src/tintx --ignore E203 --count --exit-zero --max-complexity=15 --max-line-length=127 --statistics
