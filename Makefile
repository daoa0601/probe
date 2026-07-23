METAL_TOOL_SOURCES := \
	tools/metal-counter-info.mm \
	tools/metal-moe-reduce-bench.mm \
	tools/metal-pipeline-archive.mm \
	tools/metal-q4-bench.mm \
	tools/metal-q4-moe-bench.mm
METAL_TOOL_BINS := $(patsubst tools/%.mm,build/tools/%,$(METAL_TOOL_SOURCES))

.PHONY: sync check test format doctor plan metal-tools

sync:
	uv sync --all-extras

check:
	uv run ruff check .
	uv run ruff format --check .
	uv run basedpyright
	uv run pytest --cov

test:
	uv run pytest

format:
	uv run ruff check --fix .
	uv run ruff format .

doctor:
	uv run probe doctor

plan:
	uv run probe sweep --dry-run configs/gemma4-26b-a4b-baseline.toml

metal-tools: $(METAL_TOOL_BINS)

build/tools/%: tools/%.mm
	mkdir -p build/tools
	xcrun clang++ -std=c++17 -fobjc-arc -framework Foundation -framework Metal $< -o $@
