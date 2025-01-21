# Makefile

# Default target
.DEFAULT_GOAL := all

# Phony targets
.PHONY: all help build install install-dev clean clean-build clean-pyc clean-test \
        lint lint-check check-types unittest coverage test run build-whl update-deps

# Colors for help messages
CYAN  := \033[36m
COFF  := \033[0m

# Help Target
help: ## Display this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "$(CYAN)%-30s$(COFF) %s\n", $$1, $$2}'

# Default workflow: clean, build, install dev dependencies, and run tests
all: clean build install-dev test ## Run the full development workflow

# Build and installation
build: ## Build the package
	poetry build

install: ## Install the package and extras
	poetry install --all-extras
	poetry self add poetry-dotenv-plugin

install-dev: ## Install development dependencies
	@echo "$(CYAN)Installing development dependencies...$(COFF)"
	poetry install --only dev
	poetry self add poetry-dotenv-plugin

build-whl: ## Build the wheel package
	@echo "$(CYAN)Building wheel package...$(COFF)"
	poetry build

# Clean operations
clean: clean-build clean-pyc clean-test ## Clean all build, Python, and test artifacts

clean-build: ## Remove build artifacts
	@echo "Cleaning build artifacts..."
	rm -rf build/ dist/ .eggs/
	find . -name '*.egg-info' -exec rm -rf {} +
	find . -name '*.egg' -exec rm -f {} +

clean-pyc: ## Remove Python file artifacts
	@echo "Cleaning Python artifacts..."
	find . -name '*.pyc' -delete
	find . -name '*.pyo' -delete
	find . -name '*~' -delete
	find . -name '__pycache__' -exec rm -fr {} +

clean-test: ## Remove test and coverage artifacts
	@echo "Cleaning test artifacts..."
	rm -rf .tox/ .coverage htmlcov/ .pytest_cache .reports

# Lint and type checking
lint: ## Lint and reformat the code
	poetry run autoflake vn1forecasting tests --remove-all-unused-imports --recursive --remove-unused-variables --in-place --exclude=__init__.py
	poetry run black vn1forecasting tests --line-length 120 -q
	poetry run isort vn1forecasting tests

lint-check: ## Check linting without making changes
	poetry run black vn1forecasting tests --line-length 120 --check --diff
	poetry run isort vn1forecasting tests --check-only --diff
	poetry run autoflake vn1forecasting tests --remove-all-unused-imports --recursive --remove-unused-variables --check-diff --exclude=__init__.py

check-types: ## Run type checking with mypy
	poetry run mypy vn1forecasting tests --ignore-missing-imports --pretty

# Testing
unittest: ## Run unit tests
	poetry run pytest -s -v tests

coverage: ## Run tests with coverage
	@mkdir -p .reports
	poetry run pytest --cov vn1forecasting --cov-report=html:.reports/htmlcov --junitxml=.reports/coverage.xml
	poetry run coverage report -m

test: unittest coverage ## Run all tests

# Additional commands
run: ## Run the application
	poetry run python -m vn1forecasting

update-deps: ## Update all dependencies to their latest compatible versions
	poetry update