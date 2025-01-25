# Makefile for testing, building, and maintaining Python and Rust components

# Variables
PACKAGE_NAME = tintx
PYTHON_MODULE = src/tintx
RUST_MODULE = tintx_core
CARGO = cargo
MATURIN = maturin

# Paths
export DATA_FILES := $(PWD)/docs/source/_static/data

# Default target: Build and test everything
.PHONY: all
all: test_notebooks test_coverage test-rust

# Python Installation
.PHONY: install
install:
	python3 -m pip install -U -e .[tests,docs]
	python3 -m bash_kernel.install

# Python Tests
.PHONY: test
test:
	python3 -m pytest -vv $(PWD)/tests
	rm -rf '='

# Notebook Tests
.PHONY: test_notebooks
test_notebooks:
	python3 -m pytest -vv --nbval-lax $(PWD)/docs/source

# Python Test Coverage
.PHONY: test_coverage
test_coverage:
	python3 -m pytest -vv \
		--cov=$(PYTHON_MODULE) --cov-report html:coverage_report \
		--cov-report xml --junitxml report.xml
	rm -rf '='
	python3 -m coverage report

# Build Rust Module and Install for Development
.PHONY: develop
develop:
	$(MATURIN) develop --manifest-path $(RUST_MODULE)/Cargo.toml

# Build Python Package with Rust Integration
.PHONY: build
build:
	$(MATURIN) build --manifest-path $(RUST_MODULE)/Cargo.toml --release

# Rust Tests
.PHONY: test-rust
test-rust:
	$(CARGO) test --manifest-path $(RUST_MODULE)/Cargo.toml

# Combined Python and Rust Tests
.PHONY: test-all
test-all: test test-rust

# Python Documentation
.PHONY: docs
docs:
	make -C docs clean
	make -C docs html

# Python and Rust Linting
.PHONY: lint
lint: lint-python lint-rust

.PHONY: lint-python
lint-python:
	flake8 $(PYTHON_MODULE) --count --select=E9,F63,F7,F82 --show-source --statistics
	flake8 $(PYTHON_MODULE) --ignore E203 --count --exit-zero --max-complexity=15 --max-line-length=127 --statistics
	black --check -t py313 -l 82 $(PYTHON_MODULE)
	isort --check --profile black -t py313 -l 82 $(PYTHON_MODULE)
	mypy $(PYTHON_MODULE)

.PHONY: lint-rust
lint-rust:
	$(CARGO) clippy --manifest-path $(RUST_MODULE)/Cargo.toml --all-targets --all-features -- -D warnings

# Formatting for Python and Rust
.PHONY: fmt
fmt: fmt-python fmt-rust

.PHONY: fmt-python
fmt-python:
	black -t py313 -l 82 $(PYTHON_MODULE)
	isort --profile black -t py313 -l 82 $(PYTHON_MODULE)

.PHONY: fmt-rust
fmt-rust:
	$(CARGO) fmt --manifest-path $(RUST_MODULE)/Cargo.toml --all

# Python Test Coverage with Rust
.PHONY: coverage-python
coverage-python:
	python3 -m pytest --cov=$(PYTHON_MODULE) --cov-report=html:coverage_report

# Rust Test Coverage (requires tarpaulin)
.PHONY: coverage-rust
coverage-rust:
	cargo tarpaulin --manifest-path $(RUST_MODULE)/Cargo.toml --out Html

# Debugging Rust with GDB
.PHONY: debug-rust
debug-rust:
	RUSTFLAGS="-C debuginfo=2" $(CARGO) build --manifest-path $(RUST_MODULE)/Cargo.toml
	gdb target/debug/$(PACKAGE_NAME)

# Clean Python and Rust Artifacts
.PHONY: clean
clean:
	$(CARGO) clean --manifest-path $(RUST_MODULE)/Cargo.toml
	rm -rf build dist *.egg-info $(PYTHON_MODULE)/**/*.pyc $(PYTHON_MODULE)/**/__pycache__
