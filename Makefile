.PHONY: test lint release

# Run unit tests with pytest
test:
	pytest

# Lint the codebase using flake8 and pydocstyle
lint:
	flake8 .
	pydocstyle faster_llm

# Publish a new release to PyPI using poetry
release:
	poetry publish --build
