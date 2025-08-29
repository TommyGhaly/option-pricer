# cpp/core/CMakeLists.txt

# Create a shared library from the core option pricing sources
add_library(options SHARED
    black_scholes.cpp
    greeks.cpp
    monte_carlo.cpp
    binomial_tree.cpp
)

# Include the header directory (../include from current directory)
target_include_directories(options PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include)

# Set library properties for better compatibility
set_target_properties(options PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED ON
)
