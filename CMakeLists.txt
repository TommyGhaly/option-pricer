cmake_minimum_required(VERSION 3.12)
project(option_pricer VERSION 1.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Enable testing
enable_testing()

# Add subdirectories
add_subdirectory(cpp/bindings)

# Compiler warnings
if(MSVC)
    add_compile_options(/W4 /WX)
else()
    add_compile_options(-Wall -Wextra -Wpedantic -Werror)
endif()

# Find required packages
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
