# cpp/core/CMakeLists.txt

# Create a library from the core option pricing sources
add_library(core
    black_scholes.cpp
    greeks.cpp
    monte_carlo.cpp
    binomial_tree.cpp
)

# Expose headers to anyone linking against this library
target_include_directories(core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

cpp-build:
runs-on: ubuntu-latest
steps:
    - uses: actions/checkout@v3

    - name: Install dependencies
    run: |
        sudo apt-get update
        sudo apt-get install -y cmake g++ python3-dev

    - name: Configure CMake
    run: cmake -B build -DCMAKE_BUILD_TYPE=Release

    - name: Build
    run: cmake --build build

    - name: Move built library
    run: |
        mkdir -p cpp/core
        cp build/cpp/core/liboptions.so cpp/core/
