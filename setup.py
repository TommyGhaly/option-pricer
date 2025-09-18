from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

ext_modules = [
    Pybind11Extension(
        "option_pricer",
        [
            "cpp/bindings/python_bindings.cpp",  # Your binding file
            "cpp/core/greeks.cpp",               # Add all implementation files
            "cpp/core/black_scholes.cpp",
            "cpp/core/binomial_tree.cpp",
            "cpp/core/monte_carlo.cpp",
            "cpp/models/heston.cpp",
            "cpp/models/sabr.cpp",
            "cpp/models/local_volatility.cpp",
            "cpp/models/jump_diffusion.cpp",
        ],  # Adjust path to your binding file
        include_dirs=["./cpp/include"],  # Path to your header files
        cxx_std=11,
    ),
]

setup(
    name="option_pricer",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    packages=[],  # Explicitly set empty to avoid auto-discovery
    zip_safe=False,
)
