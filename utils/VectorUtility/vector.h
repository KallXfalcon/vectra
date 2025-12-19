
/*
 *
 * vector.h (v1.1)
 * 
 * v1.1 updated : Safe malloc update
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

#include <cstdlib>
#include <cstdio>

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

        const size_t bytes = size_init * sizeof(T);
        data = static_cast<T*>(malloc(bytes));
        if (!data) {
            fprintf(stderr, "Failed to allocate memory!\n");
            exit(EXIT_FAILURE);
        }
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

        const size_t bytes = new_size * sizeof(T);
        data = static_cast<T*>(malloc(bytes));
        if (!data) {
            fprintf(stderr, "Failed to allocate memory!\n");
            exit(EXIT_FAILURE);
        }
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

    void fill(const T& value) {
        for (size_t i = 0; i < size_; ++i) {
            data[i] = value;
        }
    }

    T& operator[](size_t index) {
        return data[index];
    }

    const T& operator[](size_t index) const {
        return data[index];
    }
};

} // namespace utils

#endif // __cplusplus

#endif // VECTOR_H