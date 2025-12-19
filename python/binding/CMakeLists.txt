cmake_minimum_required(VERSION 3.28)
project(tensor_ops_python_binding LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Python COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

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

add_library(tensor_ops MODULE binding.cc)

target_compile_options(tensor_ops PRIVATE -msse4.1 -mssse3 -mfpmath=sse)
target_compile_definitions(tensor_ops PRIVATE USE_VECTRA_SSE)

target_link_libraries(tensor_ops PRIVATE 
    tenlib 
    utility 
    KerLow 
    pybind11::module
)

if(WIN32)
    set(TENSOR_OPS_SUFFIX ".pyd")
else()
    set(TENSOR_OPS_SUFFIX ".so")
endif()

if(NOT DEFINED PYTHON_SITE_PACKAGES)
    execute_process(
        COMMAND ${PYTHON_EXECUTABLE} -c "import site; print(site.getsitepackages()[0])"
        OUTPUT_VARIABLE PYTHON_SITE_PACKAGES
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
endif()

install(TARGETS tensor_ops
    LIBRARY DESTINATION ${PYTHON_SITE_PACKAGES}
    RUNTIME DESTINATION ${PYTHON_SITE_PACKAGES}
)

message(STATUS "Python site-packages: ${PYTHON_SITE_PACKAGES}")