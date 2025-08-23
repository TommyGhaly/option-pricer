# setup.py
from setuptools import setup, Extension

core_module = Extension(
    "core",
    sources=[
        "cpp/core/black_scholes.cpp",
        "cpp/core/greeks.cpp",
        "cpp/core/monte_carlo.cpp",
        "cpp/core/binary_tree.cpp",
    ],
    include_dirs=["cpp/core"],
    language="c++"
)

setup(
    name="option_pricer",
    version="1.0",
    ext_modules=[core_module],
)
