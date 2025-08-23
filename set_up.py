from setuptools import setup, Extension

setup(
    name="option_pricer",
    version="0.1",
    ext_modules=[
        Extension(
            "option_pricer",
            sources=["python/option_pricer.cpp"],  # adjust paths
            include_dirs=["include"],
            language="c++",
        ),
    ],
)
