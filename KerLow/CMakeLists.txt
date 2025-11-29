add_library(KerLow INTERFACE)

target_include_directories(KerLow INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/ScalarType
)