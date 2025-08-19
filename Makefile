# Makefile for Option Pricer Project

.PHONY: all clean build test install dev-install format lint docs

# Build C++ extensions
build:
	mkdir -p build && cd build && cmake .. && make -j$(nproc)

# Install Python package
install:
	pip install -e .

# Development install
dev-install:
	pip install -r requirements-dev.txt
	pre-commit install

# Run tests
test:
	pytest tests/ -v --cov=option_pricer --cov-report=html
	cd build && ctest

# Format code
format:
	black python/
	isort python/
	find cpp/ -name "*.cpp" -o -name "*.h" | xargs clang-format -i

# Lint code
lint:
	flake8 python/
	mypy python/
	cpplint cpp/

# Build documentation
docs:
	cd docs && sphinx-build -b html . _build/html

# Clean build artifacts
clean:
	rm -rf build/ dist/ *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.so" -delete
	find . -type f -name "*.o" -delete

# Run benchmarks
benchmark:
	python benchmarks/benchmark_results.py

# Profile code
profile:
	py-spy record -o profile.svg -- python examples/basic_pricing.py
