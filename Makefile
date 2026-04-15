.PHONY: install dev test lint format seed run-pipeline docker-up docker-down docker-logs

install:
	pip install -e ".[dev]"

dev:
	uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

test:
	pytest -v --cov=. --cov-report=term-missing

test-unit:
	pytest agents/ -v

test-integration:
	pytest tests/integration/ -v

lint:
	ruff check .
	mypy .

format:
	black .
	ruff check --fix .

seed:
	python scripts/seed_db.py

run-pipeline:
	python scripts/run_pipeline.py

run-backtest:
	python scripts/run_backtest.py

docker-up:
	docker-compose up -d

docker-down:
	docker-compose down

docker-logs:
	docker-compose logs -f

docker-build:
	docker-compose build --no-cache

docker-restart:
	docker-compose down && docker-compose up -d
