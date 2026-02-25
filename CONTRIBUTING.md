# Contributing to epyc-orchestrator

Thank you for your interest in contributing!

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/YOUR_USERNAME/epyc-orchestrator.git`
3. Create a branch: `git checkout -b my-feature`
4. Install dev dependencies: `pip install -e ".[dev]"`
5. Make your changes
6. Run verification: `make gates`
7. Run tests: `pytest tests/ -n 8`
8. Commit and push
9. Open a pull request

## Development Setup

```bash
cp .env.example .env
# Edit .env to set paths for your system
pip install -e ".[dev,toon]"
```

## Code Style

- Python 3.11+
- Formatting and linting via `ruff` (configured in `pyproject.toml`)
- All filesystem paths should use `get_config()` — never hardcode absolute paths
- Use lazy imports for heavy dependencies
- Run `make gates` before committing (schema validation, shellcheck, format, lint)

## Testing

```bash
pytest tests/ -n 8          # Parallel (safe default)
pytest tests/ -v             # Verbose, single-threaded
pytest tests/ -k "test_name" # Run specific test
```

Tests run in mock mode by default — no model servers required.

## Pull Request Guidelines

- Keep PRs focused on a single change
- Include tests for new functionality
- Ensure `make gates` passes
- Describe what changed and why in the PR description

## Reporting Issues

- Use GitHub Issues
- Include: Python version, OS, steps to reproduce, expected vs actual behavior
- For model routing issues, include your `model_registry.yaml` configuration

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
