cmake_minimum_required(VERSION 3.12)

# Find Python3 first
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

# Add pybind11
add_subdirectory(pybind11)

# Create the module - note the module name should match PYBIND11_MODULE name
pybind11_add_module(option_pricer python_bindings.cpp)

# Set properties for better compatibility
set_target_properties(option_pricer PROPERTIES
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED ON
)

# Link against your core library
target_link_libraries(option_pricer PRIVATE core)

# Include directories (pybind11_add_module handles pybind11 includes automatically)
target_include_directories(option_pricer PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../core
)

# Optional: Set RPATH for finding the core library at runtime
if(APPLE)
    set_target_properties(option_pricer PROPERTIES
        INSTALL_RPATH "@loader_path"
        BUILD_WITH_INSTALL_RPATH TRUE
    )
elseif(UNIX)
    set_target_properties(option_pricer PROPERTIES
        INSTALL_RPATH "$ORIGIN"
        BUILD_WITH_INSTALL_RPATH TRUE
    )
endif()
