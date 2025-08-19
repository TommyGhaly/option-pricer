# Contributing to Option Pricer

We welcome contributions to the Option Pricer project! This document provides guidelines for contributing.

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/your-username/option-pricer.git`
3. Create a new branch: `git checkout -b feature/your-feature-name`
4. Set up development environment: `make dev-install`

## Development Workflow

1. Make your changes
2. Add tests for new functionality
3. Run tests: `make test`
4. Format code: `make format`
5. Lint code: `make lint`
6. Commit with descriptive message
7. Push to your fork
8. Create a Pull Request

## Code Style

### Python
- Follow PEP 8
- Use Black for formatting
- Use type hints where appropriate

### C++
- Follow Google C++ Style Guide
- Use clang-format for formatting

## Testing

- Write unit tests for all new functionality
- Ensure all tests pass before submitting PR
- Aim for >90% code coverage

## Pull Request Process

1. Update documentation
2. Update CHANGELOG.md
3. Ensure CI passes
4. Request review from maintainers

## Reporting Issues

- Use issue templates
- Provide minimal reproducible example
- Include system information
