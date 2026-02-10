.PHONY: install lint format check test clean setup-hooks

# Install dependencies
install:
	conda env update -f environment.yml --prune

# Setup pre-commit hooks
setup-hooks:
	pre-commit install
	@echo "✓ Pre-commit hooks installed"

# Run linter (check only)
lint:
	ruff check .

# Fix auto-fixable lint issues
lint-fix:
	ruff check --fix .

# Format code
format:
	ruff format .

# Run all checks (lint + format check)
check:
	ruff check .
	ruff format --check .

# Run pre-commit on all files
pre-commit-all:
	pre-commit run --all-files

# Update pre-commit hooks
update-hooks:
	pre-commit autoupdate

# Clean generated files
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true

# Full setup for new developers
setup: install setup-hooks
	@echo "✓ Development environment ready!"
	@echo "Run 'make check' to verify your setup"

# Security scan for secrets
scan-secrets:
	detect-secrets scan --baseline .secrets.baseline

# Help
help:
	@echo "Available commands:"
	@echo "  make install        - Install/update conda environment"
	@echo "  make setup          - Full setup (install + hooks)"
	@echo "  make setup-hooks    - Install pre-commit hooks"
	@echo "  make lint           - Check code with ruff"
	@echo "  make lint-fix       - Fix auto-fixable issues"
	@echo "  make format         - Format code with ruff"
	@echo "  make check          - Run all checks"
	@echo "  make pre-commit-all - Run pre-commit on all files"
	@echo "  make scan-secrets   - Scan for hardcoded secrets"
	@echo "  make clean          - Remove generated files"
