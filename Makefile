# Makefile

# Default target
.DEFAULT_GOAL := all

# Phony targets
.PHONY: all help build install install-dev clean clean-build clean-pyc clean-test \
        lint lint-check check-types unittest coverage test run build-whl update-deps download_data

# Colors for help messages
CYAN  := \033[36m
COFF  := \033[0m

# Help Target
help: ## Display this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "$(CYAN)%-30s$(COFF) %s\n", $$1, $$2}'

# Default workflow: clean, build, install dev dependencies, and run tests
all: clean build install-dev test ## Run the full development workflow

# Download competition datasets
download_data: ## Download all competition datasets
	@echo "$(CYAN)Downloading competition datasets...$(COFF)"
	mkdir -p data
	curl https://www.datasource.ai/attachments/eyJpZCI6Ijk4NDYxNjE2NmZmZjM0MGRmNmE4MTczOGMyMzI2ZWI2LmNzdiIsInN0b3JhZ2UiOiJzdG9yZSIsIm1ldGFkYXRhIjp7ImZpbGVuYW1lIjoiUGhhc2UgMCAtIFNhbGVzLmNzdiIsInNpemUiOjEwODA0NjU0LCJtaW1lX3R5cGUiOiJ0ZXh0L2NzdiJ9fQ -o data/phase_0_sales.csv
	curl https://www.datasource.ai/attachments/eyJpZCI6ImM2OGQxNGNmNTJkZDQ1MTUyZTg0M2FkMDAyMjVlN2NlLmNzdiIsInN0b3JhZ2UiOiJzdG9yZSIsIm1ldGFkYXRhIjp7ImZpbGVuYW1lIjoiUGhhc2UgMSAtIFNhbGVzLmNzdiIsInNpemUiOjEwMTgzOTYsIm1pbWVfdHlwZSI6InRleHQvY3N2In19 -o data/phase_1_sales.csv 
	curl https://www.datasource.ai/attachments/eyJpZCI6IjhlNmJmNmU3ZTlhNWQ4NTcyNGVhNTI4YjAwNTk3OWE1LmNzdiIsInN0b3JhZ2UiOiJzdG9yZSIsIm1ldGFkYXRhIjp7ImZpbGVuYW1lIjoiUGhhc2UgMiAtIFNhbGVzLmNzdiIsInNpemUiOjEwMTI0MzcsIm1pbWVfdHlwZSI6InRleHQvY3N2In19 -o data/phase_2_sales.csv 
	curl https://www.datasource.ai/attachments/eyJpZCI6IjI1NDQxYmMyMTQ3MTA0MjJhMDcyYjllODcwZjEyNmY4LmNzdiIsInN0b3JhZ2UiOiJzdG9yZSIsIm1ldGFkYXRhIjp7ImZpbGVuYW1lIjoicGhhc2UgMiBzdWJtaXNzaW9uIGV4YW1pbmUgc21vb3RoZWQgMjAyNDEwMTcgRklOQUwuY3N2Iiwic2l6ZSI6MTk5MzAzNCwibWltZV90eXBlIjoidGV4dC9jc3YifX0 -o data/solution_1st_place.csv
	curl https://www.datasource.ai/attachments/eyJpZCI6IjU3ODhjZTUwYTU3MTg3NjFlYzMzOWU0ZTg3MWUzNjQxLmNzdiIsInN0b3JhZ2UiOiJzdG9yZSIsIm1ldGFkYXRhIjp7ImZpbGVuYW1lIjoidm4xX3N1Ym1pc3Npb25fanVzdGluX2Z1cmxvdHRlLmNzdiIsInNpemUiOjM5MDkzNzksIm1pbWVfdHlwZSI6InRleHQvY3N2In19 -o data/solution_2nd_place.csv
	curl https://www.datasource.ai/attachments/eyJpZCI6ImE5NzcwNTZhMzhhMTc2ZWJjODFkMDMwMTM2Y2U2MTdlLmNzdiIsInN0b3JhZ2UiOiJzdG9yZSIsIm1ldGFkYXRhIjp7ImZpbGVuYW1lIjoiYXJzYW5pa3phZF9zdWIuY3N2Iiwic2l6ZSI6Mzg4OTcyNCwibWltZV90eXBlIjoidGV4dC9jc3YifX0 -o data/solution_3rd_place.csv
	curl https://www.datasource.ai/attachments/eyJpZCI6ImVlZmUxYWY2NDFjOWMwM2IxMzRhZTc2MzI1Nzg3NzIxLmNzdiIsInN0b3JhZ2UiOiJzdG9yZSIsIm1ldGFkYXRhIjp7ImZpbGVuYW1lIjoiVEZUX3R1bmVkX1YyX3NlZWRfNDIuY3N2Iiwic2l6ZSI6NjA3NDgzLCJtaW1lX3R5cGUiOiJ0ZXh0L2NzdiJ9fQ -o data/solution_4th_place.csv
	curl https://www.datasource.ai/attachments/eyJpZCI6IjMwMDEwMmY3NTNhMzlhN2YxNTk3ODYxZTI1N2Q2NzRmLmNzdiIsInN0b3JhZ2UiOiJzdG9yZSIsIm1ldGFkYXRhIjp7ImZpbGVuYW1lIjoiZGl2aW5lb3B0aW1pemVkd2VpZ2h0c2Vuc2VtYmxlLmNzdiIsInNpemUiOjE3OTU0NzgsIm1pbWVfdHlwZSI6InRleHQvY3N2In19 -o data/solution_5th_place.csv
	curl https://www.datasource.ai/attachments/eyJpZCI6IjgyMTNhNzcyNTY0NWUyNTljNzViYWFiZDA0ZmJmNDI2LmNzdiIsInN0b3JhZ2UiOiJzdG9yZSIsIm1ldGFkYXRhIjp7ImZpbGVuYW1lIjoiUGhhc2UgMCAtIFByaWNlLmNzdiIsInNpemUiOjc1ODgyOTgsIm1pbWVfdHlwZSI6InRleHQvY3N2In19 -o data/phase_0_price.csv
	curl https://www.datasource.ai/attachments/eyJpZCI6Ijk4MTNhYmY3ZGZmZmY3NDJhOTBkNWJhMjQwMGQ5ZDkwLmNzdiIsInN0b3JhZ2UiOiJzdG9yZSIsIm1ldGFkYXRhIjp7ImZpbGVuYW1lIjoiUGhhc2UgMSAtIFByaWNlLmNzdiIsInNpemUiOjk2MjI4MywibWltZV90eXBlIjoidGV4dC9jc3YifX0 -o data/phase_1_price.csv
	@echo "$(CYAN)Download complete!$(COFF)"

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
	poetry run python -m ipykernel install --user --name=vn1forecasting

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