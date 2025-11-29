add_library(utility INTERFACE)

# MAKE VectorUtility/vector.h DIRECTORY IS AVAILABLE
target_include_directories(utility INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}
)