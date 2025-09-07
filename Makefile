.PHONY: ipython
ipython:
	@PYTHONPATH=$(PWD)/src \
	PYTHONDONTWRITEBYTECODE=1 \
	uvx --with . \
		--env-file .env \
		ipython

.PHONY: pytest
pytest:
	@PYTHONPATH=$(PWD)/src \
	PYTHONDONTWRITEBYTECODE=1 \
	uv run \
		--dev \
		--all-extras \
		pytest \
		-v

.PHONY: pytest-ci
pytest-ci:
	@PYTHONPATH=$(PWD)/src \
	PYTHONDONTWRITEBYTECODE=1 \
	uv run \
		--dev \
		--all-extras \
		pytest \
		-q \
		--maxfail=1 \
		--disable-warnings

.PHONY: mypy
mypy:
	@uv run \
		--dev \
		--all-extras \
		mypy \
		src

.PHONY: ruff
ruff:
	@uvx --with . \
		ruff check \
		src tests

.PHONY: ruff-fix
ruff-fix:
	@uvx --with . \
		ruff check \
		src tests \
		--fix


.PHONY: jupyter
jupyter:
	@PYTHONPATH=$(PWD)/src \
	PYTHONDONTWRITEBYTECODE=1 \
	uvx --with .[examples] \
		--env-file .env \
		jupyter lab \
		--notebook-dir=examples \
		--ip=127.0.0.1 \
		--port=8888 \
		--no-browser

.PHONY: re-commit-install
pre-commit-install:
	@uvx pre-commit install
