export PYTHONPATH := $(shell pwd)/src
export PYTHONDONTWRITEBYTECODE := 1

.PHONY: run-ipython
run-ipython:
	@uvx --with . \
		--env-file .env \
		ipython

.PHONY: run-pytest
run-pytest:
	@uvx --with . \
		pytest \
		-v

.PHONY: run-pytest-ci
run-pytest-ci:
	@uvx --with . \
		pytest -q --maxfail=1 --disable-warnings

.PHONY: run-ruff --with .
run-ruff:
	@uvx --with . \
		ruff check \
		src tests
