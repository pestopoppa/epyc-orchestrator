# Contributing to epyc-orchestrator

Thank you for your interest in contributing!

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/YOUR_USERNAME/epyc-orchestrator.git`
3. Create a branch: `git checkout -b my-feature`
4. Install core and development dependencies: `uv sync`
5. Make your changes
6. Run verification: `make gates`
7. Run the complete test suite: `make test`
8. Commit and push
9. Open a pull request

## Development Setup

```bash
cp .env.example .env
# Edit .env to set paths for your system
uv sync --extra toon
```

## Code Style

- Python 3.11+
- Formatting and linting via `ruff` (configured in `pyproject.toml`)
- All filesystem paths should use `get_config()` — never hardcode absolute paths
- Use lazy imports for heavy dependencies
- Run `make gates` before committing (schema validation, shell and Markdown checks,
  changed-file Python formatting, and Python lint)

## Testing

```bash
make test                                  # Complete suite, including top-level tests/
make test PYTEST_ARGS="-n 1 -k test_name"  # Focused debugging
```

The complete suite uses pytest's `tests/` testpaths. Tests requiring a live server,
live models, or the materialized host trace are opt-in; see [tests/README.md](tests/README.md).
CI runs the same suite in mock mode and reports baseline failures separately from new failures.

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
