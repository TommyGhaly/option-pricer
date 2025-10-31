# Installation Guide

Detailed installation instructions for the Option Pricer library.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Quick Installation](#quick-installation)
- [Platform-Specific Instructions](#platform-specific-instructions)
- [Building from Source](#building-from-source)
- [Troubleshooting](#troubleshooting)
- [Verification](#verification)

---

## Prerequisites

### System Requirements

- **Operating System**: Linux, macOS, or Windows
- **Python**: 3.9 or higher
- **C++ Compiler**: GCC 7+, Clang 5+, or MSVC 2019+
- **CMake**: 3.12+ (optional, for manual C++ builds)
- **Memory**: 4GB RAM minimum, 8GB+ recommended
- **Disk Space**: ~500MB for installation

### Python Dependencies

The library requires the following Python packages (automatically installed):

```txt
# Core scientific computing
numpy>=1.24.0
scipy>=1.10.0
pandas>=2.0.0

# Python bindings
pybind11>=2.11.0

# Visualization
plotly>=5.14.0
dash>=2.10.0

# Testing (optional)
pytest>=7.3.0
pytest-cov>=4.0.0

# Market data
yfinance>=0.2.18
pandas-datareader>=0.10.0

# Additional utilities
pytz
pandas-market-calendars
```

---

## Quick Installation

### Step 1: Clone Repository

```bash
git clone https://github.com/tommyghaly/option-pricer.git
cd option-pricer
```

### Step 2: Create Virtual Environment

```bash
# Create venv
python -m venv venv

# Activate (Linux/macOS)
source venv/bin/activate

# Activate (Windows)
venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4: Build C++ Extensions

```bash
python setup.py build_ext --inplace
```

The C++ extension module will be built and placed in `python/option_pricer/`.

### Step 5: Verify Installation

```bash
python -c "import option_pricer as op; print(op.__version__)"
```

You should see the version number printed (e.g., `1.0.0`).

---

## Platform-Specific Instructions

### Linux (Ubuntu/Debian)

**Install C++ Build Tools:**

```bash
sudo apt-get update
sudo apt-get install build-essential python3-dev cmake
```

**Build the Extension:**

```bash
# In the project directory with venv activated
python setup.py build_ext --inplace
```

**Common Issues:**
- If you get "Python.h not found", install `python3-dev`
- If pybind11 headers not found, ensure `pybind11` is installed in your venv

---

### macOS

**Install Xcode Command Line Tools:**

```bash
xcode-select --install
```

**Install Homebrew (if not already installed):**

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

**Install CMake (optional):**

```bash
brew install cmake
```

**Build the Extension:**

```bash
python setup.py build_ext --inplace
```

**Common Issues:**
- **Compiler not found**: Install Xcode Command Line Tools
- **Architecture mismatch**: Ensure your Python and compiler target the same architecture (x86_64 or arm64)

For Apple Silicon (M1/M2) Macs, you may need:

```bash
# Install Rosetta if needed
softwareupdate --install-rosetta

# Or use native ARM Python
arch -arm64 python setup.py build_ext --inplace
```

---

### Windows

**Install Visual Studio Build Tools:**

Download and install [Visual Studio 2019 or later](https://visualstudio.microsoft.com/downloads/) with:
- "Desktop development with C++" workload
- Windows 10 SDK

Or install just the build tools:
- [Build Tools for Visual Studio](https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2019)

**Build the Extension:**

```cmd
# Open "Developer Command Prompt for VS"
# Navigate to project directory
python setup.py build_ext --inplace
```

**Common Issues:**
- **MSVC not found**: Ensure Visual Studio is properly installed and you're using Developer Command Prompt
- **Python version mismatch**: Use Python version matching your MSVC (64-bit Python with 64-bit MSVC)

**Alternative: Use MinGW:**

```bash
# Install MinGW-w64
# Download from: https://www.mingw-w64.org/

# Build with MinGW
python setup.py build_ext --inplace --compiler=mingw32
```

---

## Building from Source

### Manual C++ Build with CMake

For development or custom builds:

```bash
# Create build directory
mkdir build
cd build

# Configure with CMake
cmake ..

# Build
cmake --build . --config Release

# The compiled module will be in build/
# Copy it to python/option_pricer/
```

### Development Install

For development with editable install:

```bash
pip install -e .
```

This creates a link to your source code, so changes take effect immediately without reinstalling.

---

## Troubleshooting

### Issue: "ImportError: option_pricer module not found"

**Solution:**
1. Verify the `.so` (Linux/Mac) or `.pyd` (Windows) file exists in `python/option_pricer/`
2. Check the filename matches your Python version:
   ```bash
   ls python/option_pricer/*.so  # or *.pyd on Windows
   ```
3. Rebuild the extension:
   ```bash
   python setup.py build_ext --inplace --force
   ```

---

### Issue: "error: Microsoft Visual C++ 14.0 or greater is required"

**Solution:** Install Visual Studio Build Tools (Windows only)
- Download from [Microsoft](https://visualstudio.microsoft.com/downloads/)
- Install "Desktop development with C++" workload

---

### Issue: "fatal error: pybind11/pybind11.h: No such file or directory"

**Solution:**
```bash
pip install --upgrade pybind11
python setup.py build_ext --inplace
```

---

### Issue: Compilation succeeds but import fails with symbol errors

**Solution:**
1. Ensure Python and compiler use same architecture (32-bit vs 64-bit)
2. Clean and rebuild:
   ```bash
   rm -rf build/
   rm python/option_pricer/*.so  # or *.pyd
   python setup.py build_ext --inplace
   ```

---

### Issue: "fatal error: 'Python.h' file not found"

**Solution:**

**Linux:**
```bash
sudo apt-get install python3-dev
```

**macOS:**
```bash
xcode-select --install
# Or reinstall Python from python.org
```

**Windows:**
- Reinstall Python and ensure "Install development headers" is checked

---

### Issue: Market data service fails to start

**Solution:**
1. Install market data dependencies:
   ```bash
   pip install yfinance pandas-datareader pytz pandas-market-calendars
   ```
2. Check internet connection
3. Verify yfinance is working:
   ```python
   import yfinance as yf
   ticker = yf.Ticker("AAPL")
   print(ticker.info['regularMarketPrice'])
   ```

---

### Issue: Dashboard won't start

**Solution:**
1. Install visualization dependencies:
   ```bash
   pip install dash plotly pandas numpy
   ```
2. Check port 8050 is not in use:
   ```bash
   # Linux/Mac
   lsof -i :8050

   # Windows
   netstat -ano | findstr :8050
   ```
3. Try a different port:
   ```python
   # In visualization.py, change:
   app.run(debug=True, port=8051)
   ```

---

## Verification

### Test C++ Extension

```python
import option_pricer as op

# Test basic pricing
price = op.black_scholes(100.0, 105.0, 0.05, 0.02, 0.2, 0.25, True)
print(f"Call price: ${price:.2f}")

# Test Greeks
delta = op.delta(100.0, 105.0, 0.05, 0.02, 0.2, 0.25, True)
print(f"Delta: {delta:.4f}")

# Test advanced models
heston_price = op.heston_model(100.0, 105.0, 0.05, 0.02, 0.25, 2.0, 0.04, 0.3, -0.7, 0.04, True)
print(f"Heston price: ${heston_price:.2f}")

print("\nAll tests passed!")
```

### Run Test Suite

```bash
# Run all tests
pytest python/tests/ -v

# Run with coverage
pytest python/tests/ --cov=option_pricer --cov-report=html

# Run specific test
pytest python/tests/test_black_scholes.py -v
```

### Test Market Data Service

```python
from python.option_pricer.market_data import MarketDataService

# Quick test (requires internet)
service = MarketDataService(['SPY'])
success = service.start()

if success:
    import time
    time.sleep(5)  # Wait for initial data
    spy_data = service.get_spot_price('SPY')
    print(f"SPY Price: ${spy_data.get('price', 'N/A')}")
    service.stop()
    print("Market data service working!")
else:
    print("Market data service failed to start")
```

### Test Visualization Dashboard

```bash
# Start the dashboard
python python/option_pricer/visualization.py

# Open browser to http://localhost:8050
# You should see the interactive dashboard
# Press Ctrl+C to stop
```

---

## Next Steps

After successful installation:

1. **Read the Usage Guide**: [USAGE_GUIDE.md](USAGE_GUIDE.md)
2. **Explore API Reference**: [API_REFERENCE.md](API_REFERENCE.md)
3. **Try Examples**: Check the `examples/` directory (if available)
4. **Run Benchmarks**: See `benchmarks/` for performance tests

---

## Getting Help

If you encounter issues not covered here:

1. Check the [GitHub Issues](https://github.com/tommyghaly/option-pricer/issues)
2. Review the [API Reference](API_REFERENCE.md)
3. Open a new issue with:
   - Your OS and Python version
   - Complete error message
   - Steps to reproduce

---

## Uninstallation

To remove the library:

```bash
# Deactivate virtual environment
deactivate

# Remove the directory
cd ..
rm -rf option-pricer/

# Or if installed with pip -e
pip uninstall option-pricer
```

---

## Development Setup

For contributing to the library:

```bash
# Clone with development branch
git clone -b develop https://github.com/tommyghaly/option-pricer.git
cd option-pricer

# Install development dependencies
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install

# Build in development mode
python setup.py develop

# Run tests
pytest python/tests/ -v

# Check code style
black python/
flake8 python/
```

See [CONTRIBUTING.md](../CONTRIBUTING.md) for more details.
