cmake_minimum_required(VERSION 3.28)
project(vectra LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_subdirectory(test)      # for testing tensor operation
add_subdirectory(TenLib)    # TenLib directory
add_subdirectory(utils)     # utils directory
add_subdirectory(KerLow)    # Scalar type included