setup:
    uv sync --group dev --group test
test:
    uv run pytest tests/ -v
lint:
    uv run ruff check .
    uv run ruff format --check .
    uv run pyrefly check --python-interpreter-path .venv/bin/python
generate:
    pixi run -e data generate
train:
    pixi run -e schooner train
eval:
    pixi run -e schooner eval
deploy:
    echo "TODO: deploy via CI (Step 6); do not deploy directly"
