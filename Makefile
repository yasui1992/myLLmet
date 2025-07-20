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


.PHONY: run-ruff --with .
run-ruff:
	@uvx --with . \
		ruff check \
		src tests
