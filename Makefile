export PYTHONPATH := $(shell pwd)/src
export PYTHONDONTWRITEBYTECODE := 1

.PHONY: run-ipython
run-ipython:
	@uvx ipython
