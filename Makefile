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
		ruff check \
		src tests \
		--fix

.PHONY: run-mypy
run-mypy:
	@uv run \
		--dev \
		--all-extras \
		mypy \
		src

.PHONY: run-jupyter
run-jupyter:
	@uvx --with .[examples] \
		--env-file .env \
		jupyter lab \
		--notebook-dir=examples \
		--ip=127.0.0.1 \
		--port=8888 \
		--no-browser

.PHONY: run-pre-commit-install
run-pre-commit-install:
	@uvx pre-commit install
