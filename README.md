# Vectra

**A Tiny Tensor Framework**

---

## Installation

### Clone the repository

```bash
git clone https://github.com/KallXfalcon/vectra.git
```

Make sure it is cloned into the desired location:

* **Linux:** `/home/YourUsername/vectra`
* **Windows:** `C:\Users\YourUsername\vectra`

---

## Installation & Setup Instructions for Linux

1. Open the terminal as your default user (not as superuser). You should start inside `/home/YourUsername`.
2. Clone the repository (if not already done):

```bash
git clone https://github.com/KallXfalcon/vectra.git
```

3. Create a new project directory inside `/home/YourUsername`.

4. Inside your project directory, create a `CMakeLists.txt` file with the following content:

```cmake
cmake_minimum_required(VERSION 3.28) # Make sure the version is correct (3.28) or use the latest

set(CMAKE_CXX_STANDARD 17)

# Subdirectory for Tensor implementation
add_subdirectory(${CMAKE_SOURCE_DIR}/../vectra/TenLib ${CMAKE_BINARY_DIR}/TenLib_build)  # Main Tensor headers
add_subdirectory(${CMAKE_SOURCE_DIR}/../vectra/KerLow ${CMAKE_BINARY_DIR}/KerLow_build)  # Data types (int8, int16, int32, float32, float64)
add_subdirectory(${CMAKE_SOURCE_DIR}/../vectra/utils ${CMAKE_BINARY_DIR}/utils_build)    # Utilities used inside Tensor headers

# Your executable file
add_executable(example main.cpp) # Replace 'main.cpp' with your file name

# Link with the Vectra framework
target_link_libraries(example PRIVATE TenLib utility KerLow)
```

---

# Installation & setup Instructions for Windows

1. Make sure your git bash is installed in Windows and then open the git bash you should start inside `C:/Users/YourUsername/`.
2. Clone the repository
```bash
git clone https://github.com/KallXfalcon/vectra.git
```
3. Go to the vectra project folder and open test (vectra/test)
4. Inside test there are CMakeLists.txt you can edit the executable

## More Information

This framework is still under development.

**Supported CPU Features:**

* SIMD-based operations: SSE, SSE2, AVX, AVX2
* AVX512 support is coming soon

## Update info
**Version:** 1.1

* TensorOps.h updated into 1.1 version : rand, randn, add, sub, mul, div fixed. Released new operation flatten, sum and max
* ScalarType.h updated into 1.1 version : added operator(+, -, *, /) in the class
---

## Release Schedule

* Python bindings (if applicable) might be released on **December 25, 2025 (Christmas)**
* Full framework completion expected by **end of 2025**

---

**Note:** This framework is intended for developers who want a lightweight, modular tensor library for C++ with CPU optimization support.

# Usage

* How to import
```c++
#include <cpu/TensorOps.h>
```

## Tensor basic operation

* Tensor variable
```c++
Tensor<vectra::int8> T;
Tensor<vectra::int16> T;
Tensor<vectra::int32> T;
Tensor<vectra::float32> T;
Tensor<vectra::float64> T;
```

* Tensor ops
```c++
full({3, 3}, 1.0f); // Maximum dimension is 32
zeros({3, 3});
ones({3, 3});
twos({3 ,3});
rand<vectra::type>({3, 3});
randn<vectra::type>({3, 3});

dot(Tensor A, Tensor B);
```

* Tensor arithmetic
``` c++
// Parameter
add(Tensor A, Tensor B);
sub(Tensor A, Tensor B);
mul(Tensor A, Tensor B);
div(Tensor A, Tensor B);
```
