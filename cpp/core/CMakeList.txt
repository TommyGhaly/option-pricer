# cpp/core/CMakeLists.txt

# Create a library from the core option pricing sources
add_library(core
    black_scholes.cpp
    greeks.cpp
    monte_carlo.cpp
    binary_tree.cpp
)

# Expose headers to anyone linking against this library
target_include_directories(core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
