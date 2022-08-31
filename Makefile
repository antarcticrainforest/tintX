# makefile used for testing
#
#
all: install test

.PHONY: docs
install:
	python3 -m pip install .[tests]

test:
	python3 -m pytest -vv $(PWD)/src/tintx/test

test_coverage:
	python3 -m pytest -vv \
	    --cov=$(PWD)/src --cov-report=html:coverage_report \
	    --junitxml=report.xml --cov-report xml:coverage_report.xml \
		$(PWD)/src/tintx/test
	python3 -m coverage report

docs:
	make -C docs clean
	make -C docs html

prepdocs:
	python3 -m pip install -e .[docs]


lint:
	mypy
	black --check -t py310 src
	flake8 src/tintx --ignore E203 --count --exit-zero --max-complexity=15 --max-line-length=127 --statistics
