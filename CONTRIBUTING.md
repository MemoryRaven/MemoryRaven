# Contributing to Memory Empire

First off, thank you for considering contributing to Memory Empire! It's people like you that make Memory Empire such a great tool for the AI agent community.

## Code of Conduct

This project and everyone participating in it is governed by the [Memory Empire Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code.

## How Can I Contribute?

### 🐛 Reporting Bugs

Before creating bug reports, please check existing issues as you might find out that you don't need to create one. When you are creating a bug report, please include as many details as possible:

- **Use a clear and descriptive title**
- **Describe the exact steps to reproduce the problem**
- **Provide specific examples to demonstrate the steps**
- **Describe the behavior you observed after following the steps**
- **Explain which behavior you expected to see instead and why**
- **Include code samples and error messages**

### 💡 Suggesting Enhancements

Enhancement suggestions are tracked as GitHub issues. When creating an enhancement suggestion, please include:

- **Use a clear and descriptive title**
- **Provide a step-by-step description of the suggested enhancement**
- **Provide specific examples to demonstrate the steps**
- **Describe the current behavior and explain which behavior you expected to see instead**
- **Explain why this enhancement would be useful**

### 🔧 Pull Requests

1. Fork the repo and create your branch from `main`
2. If you've added code that should be tested, add tests
3. If you've changed APIs, update the documentation
4. Ensure the test suite passes
5. Make sure your code lints
6. Issue that pull request!

## Development Process

### 1. Setup Development Environment

```bash
# Clone your fork
git clone https://github.com/YOUR-USERNAME/MemoryRaven.git
cd MemoryRaven

# Add upstream remote
git remote add upstream https://github.com/ravenbadbihh/MemoryRaven.git

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install in development mode
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

### 2. Development Workflow

```bash
# Create a feature branch
git checkout -b feature/your-feature-name

# Make your changes
# ... edit files ...

# Run tests
pytest

# Run linting
make lint

# Commit your changes (will run pre-commit hooks)
git commit -m "feat: add new memory consolidation strategy"

# Push to your fork
git push origin feature/your-feature-name

# Open a pull request
```

### 3. Testing

We use pytest for testing. Please write tests for any new functionality:

```python
# tests/test_your_feature.py
import pytest
from memory_empire import MemoryEmpire

@pytest.mark.asyncio
async def test_your_feature():
    empire = MemoryEmpire("test-agent")
    # Test your feature
    result = await empire.your_new_method()
    assert result == expected_value
```

### 4. Documentation

- Update docstrings for any new functions/classes
- Update README.md if adding new features
- Add examples to the `examples/` directory
- Update API documentation if changing public APIs

## Style Guidelines

### Python Style Guide

We use [Black](https://black.readthedocs.io/) for code formatting and [Ruff](https://github.com/astral-sh/ruff) for linting.

```bash
# Format code
black .

# Run linter
ruff check .

# Type checking
mypy memory_empire
```

### Code Style

- Use type hints where possible
- Write descriptive docstrings for all public functions
- Keep functions focused and small
- Use meaningful variable names
- Follow PEP 8 conventions

### Commit Messages

We follow [Conventional Commits](https://www.conventionalcommits.org/):

- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation changes
- `style:` Code style changes (formatting, etc)
- `refactor:` Code changes that neither fix bugs nor add features
- `perf:` Performance improvements
- `test:` Adding or updating tests
- `chore:` Maintenance tasks

Examples:
```
feat: add Redis backend support
fix: prevent memory duplication in consolidation
docs: update installation guide for Windows
perf: optimize vector search for large datasets
```

## Project Structure

```
MemoryRaven/
├── src/memory_empire/      # Main package code
│   ├── core/               # Core functionality
│   ├── backends/           # Storage backends
│   ├── memory_types/       # Memory type implementations
│   ├── retrieval/          # Search and retrieval
│   └── utils/              # Utilities
├── tests/                  # Test files
├── examples/               # Example usage
├── docs/                   # Documentation
└── scripts/                # Helper scripts
```

## Community

- Join our [Discord server](https://discord.gg/XXX) for discussions
- Follow [@ravenbadbihh](https://twitter.com/ravenbadbihh) on Twitter for updates
- Check out our [blog](https://blog.memoryempire.ai) for technical deep-dives

## Recognition

Contributors will be recognized in:
- The README.md contributors section
- Release notes
- Our Hall of Fame (for significant contributions)

## Questions?

Feel free to:
- Open an issue for questions
- Ask in Discord
- Email ravenbadbihh@gmail.com

Thank you for making Memory Empire better! 🧠⚡