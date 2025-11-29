add_library(TenLib INTERFACE)

# Make cpu/TensorOps.h available
target_include_directories(TenLib INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(TenLib INTERFACE utility KerLow)

target_compile_options(TenLib INTERFACE -mavx2)