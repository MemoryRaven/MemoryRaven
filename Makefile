.PHONY: lint format test typecheck ci

format:
	python -m ruff format .

lint:
	python -m ruff check .

typecheck:
	python -m mypy src

test:
	python -m pytest --cov=memory_empire --cov-report=term-missing

ci: lint typecheck test
