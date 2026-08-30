# Thin wrappers; gates run the underlying commands directly.
setup:
	uv sync

build:
	uv run python prepare.py --build

verify:
	uv run python prepare.py --verify

test:
	uv run pytest -q

bench:
	uv run python -m recsys.zoo bench --budget 300

iterate:
	uv run python -m harness iterate

report:
	uv run python -m harness report
