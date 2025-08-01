export PYTHONPATH := $(shell pwd)/src
export PYTHONDONTWRITEBYTECODE := 1

.PHONY: run-ipython
run-ipython:
	@uvx --with . \
		--env-file .env \
		ipython

.PHONY: run-pytest
run-pytest:
	@uvx --with .[pandas] \
		pytest \
		-v

.PHONY: run-pytest-ci
run-pytest-ci:
	@uvx --with .[pandas] \
		pytest -q --maxfail=1 --disable-warnings

.PHONY: run-ruff
run-ruff:
	@uvx --with . \
		ruff check \
		src tests

.PHONY: run-ruff-fix
run-ruff-fix:
	@uvx --with . \
		--optional pandas \
		ruff check \
		src tests \
		--fix