# Vectra

**A Tiny Tensor Framework**

## Installation

Clone the repository:

```bash
git clone https://github.com/KallXfalcon/vectra.git

```

Make sure it's cloned into the desired location:

* **Linux:** `/home/YourUsername/vectra`
* **Windows:** `C:\Users\YourUsername\vectra`

**Installation & setup instruction for Linux**
* First step open terminal as default user (NOT SUPER USER) and it will directly inside of home/YourUsername
* Second step run this after do first step
```bash
git clone https://github.com/KallXfalcon/vectra.git

```
* Third step create new PROJECT dir inside of home/YourUsername
* Fourth step create a CMakeLists.txt inside of PROJECT that you want implement vectra and write this inside CMakeLists.txt
```bash
cmake_minimum_required(VERSION 3.28) # Make sure the version is correct (3.28) use latest

set(CMAKE_CXX_STANDARD 17)

# Subdirectory for implement Tensor
add_subdirectory(${CMAKE_SOURCE_DIR}/../vectra/TenLib ${CMAKE_BINARY_DIR}/TenLib_build)  # Main Tensor header are here
add_subdirectory(${CMAKE_SOURCE_DIR}/../vectra/KerLow ${CMAKE_BINARY_DIR}/KerLow_build)  # For data type (e.g int8, int16, int32, float32, float64)
add_subdirectory(${CMAKE_SOURCE_DIR}/../vectra/utils ${CMAKE_BINARY_DIR}/utils_build)    # utility that used inside of Tensor header

# executable your file that you wanted to execute
add_executable(example main.cpp) # or if you use another name -> file-name.cpp

# Link into the framework
target_link libraries(example PRIVATE TenLib utility KerLow)
```
# More Info
This framework still under development

**SUPPORT**
* CPU - based on SIMD (SSE, SSE2, AVX, AVX2) "AVX512" coming soon

**VERSION ; 1.0**

# Release

**This framework might released in 25 December 2025 for Python (Christmas)**
