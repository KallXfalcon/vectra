cmake_minimum_required(VERSION 3.28)

project(tensor_ops_python_binding LANGUAGES CXX)

# ------------------------------------------------------------------
# C++ standard
# ------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ------------------------------------------------------------------
# Windows Python root (REAL Windows Python, NOT Linux)
# ------------------------------------------------------------------
set(WIN_PYTHON_ROOT "$ENV{HOME}/tools")

# Sanity checks
if(NOT EXISTS ${WIN_PYTHON_ROOT}/include/Python.h)
    message(FATAL_ERROR "Python.h not found in ${WIN_PYTHON_ROOT}/include")
endif()

# ------------------------------------------------------------------
# pybind11 (source-based, cross-compile safe)
# ------------------------------------------------------------------
add_subdirectory(/home/falcontreatred/pybind11 pybind11-build)

# ------------------------------------------------------------------
# Interface libraries
# ------------------------------------------------------------------
add_library(KerLow INTERFACE)
target_include_directories(KerLow INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../KerLow
    ${CMAKE_CURRENT_SOURCE_DIR}/../../KerLow/ScalarType
)

add_library(utility INTERFACE)
target_include_directories(utility INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../utils
    ${CMAKE_CURRENT_SOURCE_DIR}/../../utils/VectorUtility
)

add_library(tenlib INTERFACE)
target_include_directories(tenlib INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../TenLib
    ${CMAKE_CURRENT_SOURCE_DIR}/../../TenLib/cpu
)
target_link_libraries(tenlib INTERFACE KerLow utility)

# ------------------------------------------------------------------
# Python module
# ------------------------------------------------------------------
add_library(tensor_ops MODULE binding.cc)

set_target_properties(tensor_ops PROPERTIES
    PREFIX ""
    OUTPUT_NAME "tensor_ops"
    SUFFIX ".pyd"
)

target_include_directories(tensor_ops PRIVATE
    ${WIN_PYTHON_ROOT}/include
)

target_link_directories(tensor_ops PRIVATE
    ${WIN_PYTHON_ROOT}/libs
)

target_link_libraries(tensor_ops PRIVATE
    pybind11::module
    tenlib
)

target_compile_definitions(tensor_ops PRIVATE
    PYBIND11_NO_ASSERT_GIL_HELD_INCREF_DECREF
    PYBIND11_DETAILED_ERROR_MESSAGES
)

# ------------------------------------------------------------------
# Prevent host contamination
# ------------------------------------------------------------------
target_compile_options(tensor_ops PRIVATE
    -Wno-attributes
)

