PYTHON ?= python3

.PHONY: install-dev lint format typecheck test run

install-dev:
	$(PYTHON) -m pip install -e '.[dev]'

lint:
	ruff check .

format:
	ruff format .

typecheck:
	mypy app tests

test:
	pytest

run:
	uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
