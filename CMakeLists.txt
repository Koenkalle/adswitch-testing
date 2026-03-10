cmake_minimum_required(VERSION 3.14)
project(adswitch CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Compile with warnings and optimisations.
if(MSVC)
    add_compile_options(/W4 /O2)
else()
    add_compile_options(-Wall -Wextra -Wpedantic -O2)
endif()

# ---------------------------------------------------------------------------
# Include directory shared by all targets
# ---------------------------------------------------------------------------
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# ---------------------------------------------------------------------------
# Executable: experiment runner
# ---------------------------------------------------------------------------
add_executable(adswitch_runner src/main.cpp)

# ---------------------------------------------------------------------------
# Executable: tests
# ---------------------------------------------------------------------------
add_executable(adswitch_tests tests/test_adswitch.cpp)

# ---------------------------------------------------------------------------
# CTest integration
# ---------------------------------------------------------------------------
enable_testing()
add_test(NAME adswitch_tests COMMAND adswitch_tests)
