
/*
 *
 * vector.h --- v1.0
 * 
 * Copyright(C) 2025 KallXfalcon
 * GitHub : https://github.com/KallXfalcon
 * 
 * This program is licensed under the GNU General Public License v3.0 (GPL v3.0)
 * See <https://www.gnu.org/licenses> for details.
 * 
 * 
 * Main of vector utility (HEADER) replace std::vector make costumize vector that more flexible to control.
 * This file is a part of VECTRA framework
 * 
*/

// This file is based on POSIX especially LINUX because this framework is created in Linux.
// Based on POSIX especially LINUX not mean it's not support Microsoft Windows or other NON-POSIX Operating system.
// It's also support NON-POSIX operating system such like Microsoft Windows.
// And make sure you have supported hardware (X64 modern CPU) include (SSE, SSE2, AVX, AVX2).

#pragma once
#ifndef VECTOR_H
#define VECTOR_H

#ifdef __cplusplus

#include <immintrin.h>
#include <type_traits>
#include <cstdlib>
#include <cstdio>
#include <new>

namespace utils {

template<typename T>
class vector {
private:
    size_t size_ = 0;

public:
    T* data = nullptr;

    vector() noexcept : size_(0), data(nullptr) {}

    explicit vector(const size_t size_init)
        : size_(0), data(nullptr)
    {
        if (size_init == 0) {
            return;
        }

        size_t alignment = 1;

        if constexpr (std::is_same_v<T, char> || std::is_same_v<T, int8_t>) alignment = 1;
        else if constexpr (std::is_same_v<T, short> || std::is_same_v<T, int16_t>) alignment = 2;
        else if constexpr (std::is_same_v<T, int> || std::is_same_v<T, float> || std::is_same_v<T, int32_t>) alignment = 4;
        else if constexpr (std::is_same_v<T, double>) alignment = 8;
        else if constexpr (std::is_same_v<T, __m128i> || std::is_same_v<T, __m128h> || std::is_same_v<T, __m128>  || std::is_same_v<T, __m128d>) alignment = 16;
        else if constexpr (std::is_same_v<T, __m256i> || std::is_same_v<T, __m256h> || std::is_same_v<T, __m256>  || std::is_same_v<T, __m256d>) alignment = 32;
        else if constexpr (std::is_same_v<T, __m512i> || std::is_same_v<T, __m512h> || std::is_same_v<T, __m512>  || std::is_same_v<T, __m512d>) alignment = 64;
        else if constexpr (std::is_same_v<T, size_t>) alignment = sizeof(size_t);
        else {
            fprintf(stderr, "Error: Unknown vector type.\n");
            exit(EXIT_FAILURE);
        }

        const size_t bytes = size_init * sizeof(T);
        const size_t aligned_bytes = ((bytes + alignment - 1) / alignment) * alignment;

    #ifdef _POSIX_VERSION
        if (posix_memalign(reinterpret_cast<void**>(&data), alignment, aligned_bytes) != 0) {
            fprintf(stderr, "Failed to allocate aligned memory!\n");
            exit(EXIT_FAILURE);
        }
    #else
        data = static_cast<T*>(aligned_alloc(alignment, aligned_bytes));
        if (!data) {
            fprintf(stderr, "Failed to allocate aligned memory!\n");
            exit(EXIT_FAILURE);
        }
    #endif

        size_ = size_init;
    }

    void resize(const size_t new_size)
    {
        if (new_size == size_) return;

        if (data) {
            free(data);
            data = nullptr;
        }

        if (new_size == 0) {
            size_ = 0;
            return;
        }

        size_t alignment = 1;

        if constexpr (std::is_same_v<T, char> || std::is_same_v<T, int8_t>) alignment = 1;
        else if constexpr (std::is_same_v<T, short> || std::is_same_v<T, int16_t>) alignment = 2;
        else if constexpr (std::is_same_v<T, int> || std::is_same_v<T, float> || std::is_same_v<T, int32_t>) alignment = 4;
        else if constexpr (std::is_same_v<T, double>) alignment = 8;
        else if constexpr (std::is_same_v<T, __m128i> || std::is_same_v<T, __m128h> || std::is_same_v<T, __m128>  || std::is_same_v<T, __m128d>) alignment = 16;
        else if constexpr (std::is_same_v<T, __m256i> || std::is_same_v<T, __m256h> || std::is_same_v<T, __m256>  || std::is_same_v<T, __m256d>) alignment = 32;
        else if constexpr (std::is_same_v<T, __m512i> || std::is_same_v<T, __m512h> || std::is_same_v<T, __m512>  || std::is_same_v<T, __m512d>) alignment = 64;
        else if constexpr (std::is_same_v<T, size_t>) alignment = sizeof(size_t);

        size_t bytes = new_size * sizeof(T);
        size_t aligned_bytes = ((bytes + alignment - 1) / alignment) * alignment;

    #ifdef _POSIX_VERSION
        if (posix_memalign(reinterpret_cast<void**>(&data), alignment, aligned_bytes) != 0) {
            fprintf(stderr, "Failed to allocate aligned memory!\n");
            exit(EXIT_FAILURE);
        }
    #else
        data = static_cast<T*>(aligned_alloc(alignment, aligned_bytes));
        if (!data) {
            fprintf(stderr, "Failed to allocate aligned memory!\n");
            exit(EXIT_FAILURE);
        }
    #endif

        size_ = new_size;
    }

    ~vector() {
        if (data) {
            free(data);
            data = nullptr;
        }
        size_ = 0;
    }

    vector(const vector&) = delete;
    vector& operator=(const vector&) = delete;

    vector(vector&& other) noexcept
        : size_(other.size_), data(other.data)
    {
        other.size_ = 0;
        other.data = nullptr;
    }

    vector& operator=(vector&& other) noexcept
    {
        if (this != &other) {
            if (data) {
                free(data);
            }
            size_ = other.size_;
            data = other.data;
            other.size_ = 0;
            other.data = nullptr;
        }
        return *this;
    }

    size_t size() const { return size_; }
};

} // namespace utils

#endif // __cplusplus

#endif // VECTOR_H