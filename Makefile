.PHONY: test

# Default suite: everything that needs neither a GPU nor the released substrate.
test:
	uv run pytest -q -m "not tier2 and not gpu"
