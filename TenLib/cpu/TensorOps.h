
/*
 *
 * TensorOps.h (v1.1)
 * 
 * v1.1 updated : rand, randn, arithmetic bug fixed!
 *                added new operation(flatten, sum)
 * 
 * Copyright(C) 2025 KallXfalcon
 * GitHub : https://github.com/KallXfalcon
 * 
 * This program is licensed under the GNU General Public License v3.0 (GPL v3.0)
 * See <https://www.gnu.org/licenses> for details.
 * 
 * 
 * Main of SIMD based Tensor Operation
 * This file is a part of VECTRA framework
 * 
*/

#pragma once
#ifndef TENSOROPS_H
#define TENSOROPS_H

#ifdef __cplusplus

#include <ScalarType/ScalarType.h>
#include <VectorUtility/vector.h>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include <cmath>

template<typename T>
class Tensor {
public:
    std::vector<size_t> shape;
    utils::vector<ScalarType<T>> data;

    Tensor() = default;

    Tensor(const std::vector<size_t>& shape_)
        : shape(shape_)
    {
        size_t total = 1;
        for (auto dim : shape_) total *= dim;
        data.resize(total);
    }

    Tensor(const std::vector<size_t>& shape_, const T value)
    : shape(shape_.begin(), shape_.end())
    {
        size_t total = 1;
        for (auto dim : shape) {
            total *= dim;
        }

        size_t count = total;

        data.resize(count);

        for (size_t i = 0; i < count; ++i) {
            data.data[i] = value;
        }
    }
};

static constexpr size_t TENSOR_PRINT_LIMIT = 100;

template<typename T>
void print_tensor_recursive(std::ostream& os, const Tensor<T>& t, size_t dim, size_t& index, size_t indent, size_t& printed)
{
    if (printed >= TENSOR_PRINT_LIMIT) {
        os << "...";
        return;
    }

    size_t ndim = t.shape.size();

    if (dim == ndim - 1) {
        os << "[";

        size_t n = t.shape[dim];
        for (size_t i = 0; i < n; ++i) {

            if (printed >= TENSOR_PRINT_LIMIT) {
                os << ", ...";
                break;
            }

            if (i > 0) os << ", ";

            os << t.data.data[index++].get_scalar();
            printed++;
        }

        os << "]";
        return;
    }

    os << "[\n";

    size_t n = t.shape[dim];
    for (size_t i = 0; i < n; ++i) {
        os << std::string(indent + 2, ' ');

        if (printed >= TENSOR_PRINT_LIMIT) {
            os << "...";
            break;
        }

        print_tensor_recursive(os, t, dim + 1, index, indent + 2, printed);

        if (i + 1 < n)
            os << ",\n";
    }

    os << "\n" << std::string(indent, ' ') << "]";
}

template<typename T>
std::ostream& operator<<(std::ostream& os, const Tensor<T>& t)
{
    os << "tensor([";

    // print shape
    for(size_t i = 0; i < t.shape.size(); ++i){
        if (i > 0) os << ", ";
        os << t.shape[i];
    }
    os << "]\n";

    // scalar case
    if(t.shape.empty()){
        os << t.data.data[0].get_scalar() << ")";
        return os;
    }

    size_t index = 0;
    size_t printed = 0;

    print_tensor_recursive(os, t, 0, index, 0, printed);

    os << "\n)";
    return os;
}

/*
 *
 * TEMPLATE FOR TENSOR OPERATION
 * 
*/

template<typename T>
T maxArrayTemplate(const T* array, int n)
{
    T mx = array[0];
    for(size_t i=1; i<n; i++){
        if(array[i] > mx) mx = array[i];
    }
    return mx;
}

/*==============================|
 *                              |
 *                              |
 *                              |
 * The entire Tensor operation  |
 *                              |
 *                              |
 *==============================|                            
*/

template<typename T>
Tensor<T> full(const std::vector<size_t>& shape, const T value)
{
    return Tensor<T>(shape, value);
}

template<typename T>
Tensor<T> zeros(const std::vector<size_t>& shape)
{
    return Tensor<T>(shape, T{0});
}

template<typename T>
Tensor<T> ones(const std::vector<size_t>& shape)
{
    return Tensor<T>(shape, T{1});
}

template<typename T>
Tensor<T> twos(const std::vector<size_t>& shape)
{
    return Tensor<T>(shape, T{2});
}

template<typename T>
Tensor<T> rand(const std::vector<size_t>& shape)
{
    Tensor<T> out(shape);

    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::uniform_real_distribution<T> dist(0.0, 1.0);

    for (size_t i = 0; i < out.data.size(); i++) out.data.data[i] = dist(gen);

    return out;
}

template<typename T>
Tensor<T> randn(const std::vector<size_t>& shape)
{
    Tensor<T> out(shape);

    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::normal_distribution<T> dist(0, 1);

    for (size_t i = 0; i < out.data.size(); i++) out.data.data[i] = dist(gen);

    return out;
}

template<typename T>
Tensor<T> add(const Tensor<T>& A, const Tensor<T>& B)
{
    if(A.data.size() != B.data.size() || A.shape != B.shape){
        fprintf(stderr, "vectra add() : Shape doesn't match!\n");
    }
    Tensor<T> out(A.shape, T{0});
    for (size_t i = 0; i < A.data.size(); ++i) out.data.data[i] = A.data.data[i] + B.data.data[i];
    return out;
}

template<typename T>
Tensor<T> sub(const Tensor<T>& A, const Tensor<T>& B)
{
    if(A.data.size() != B.data.size() || A.shape != B.shape){
        fprintf(stderr, "vectra sub() : Shape doesn't match!\n");
        exit(EXIT_FAILURE);
    }
    Tensor<T> out(A.shape, T{0});
    for (size_t i = 0; i < A.data.size(); ++i) out.data.data[i] = A.data[i] - B.data.data[i];
    return out;
}

template<typename T>
Tensor<T> mul(const Tensor<T>& A, const Tensor<T>& B)
{
    if(A.data.size() != B.data.size() || A.shape != B.shape){
        fprintf(stderr, "vectra mul() : Shape doesn't match!\n");
        exit(EXIT_FAILURE);
    }
    Tensor<T> out(A.shape, T{0});
    for (size_t i = 0; i < A.data.size(); ++i) out.data[i] = A.data.data[i] * B.data.data[i];
    return out;
}

template<typename T>
Tensor<T> div(const Tensor<T>& A, const Tensor<T>& B)
{
    if(A.data.size() != B.data.size() || A.shape != B.shape){
        fprintf(stderr, "vectra div() : Shape doesn't match!\n");
        exit(EXIT_FAILURE);
    }
    Tensor<T> out(A.shape, T{0});
    for (size_t i = 0; i < A.Data.size(); ++i) {
        if(B.Data[i] == T{0}){
            fprintf(stderr, "vectra div() : cannot divide by zero!\n");
            exit(EXIT_FAILURE);
        }
        out.data.data[i] = A.data.data[i] / B.data.data[i];
    }
    return out;
}

template<typename T>
Tensor<T> dot(const Tensor<T>& A, const Tensor<T>& B)
{   
    // Vector * Matrix
    if(A.shape.size() == 1 && B.shape.size() == 2){
        size_t L = A.shape[0];
        size_t M = B.shape[1];

        Tensor<T> out({M}, T{0});

        for(size_t j = 0; j < M; ++j){
            T sum = T{0};
            for (size_t i = 0; i < L; ++i) sum += A.data.data[i].get_scalar() * B.data.data[i * M + j].get_scalar();

            out.data.data[j] = sum;
        }
        return out;
    }
    // Matrix * vector
    else if(A.shape.size() == 2 && B.shape.size() == 1){
        size_t N = A.shape[0];
        size_t K = A.shape[1];

        Tensor<T> out({N}, T{0});

        for(size_t i = 0; i < N; ++i){
            T sum = T{0};
            for(size_t j = 0; j < K; ++j) sum += A.data.data[i * K + j].get_scalar() * B.data.data[j].get_scalar();

            out.data.data[i] = sum;
        }

        return out;
    }
    // Matrix * Matrix
    if(A.shape.size() == 2 && B.shape.size() == 2){
        size_t N = A.shape[0];
        size_t K = A.shape[1];
        size_t M = B.shape[1];

        Tensor<T> out({N, M}, T{0});

        for(size_t i = 0; i < N; ++i){
            T sum = T{0};
            for(size_t j = 0; j < M; ++j){
                for (size_t k = 0; k < K; ++k) sum += A.data.data[i * K + k].get_scalar() * B.data.data[k * M + j].get_scalar();

                out.data.data[i * M + j] = sum;
            }
        }

        return out;
    }
}

template<typename T>
Tensor<T> max(const Tensor<T>& t1)
{
    Tensor<T> out({1}, 0);

    out.data.data[0] = maxArrayTemplate(t1.data.data, t1.data.size);

    return out;
}

template<typename T>
Tensor<T> flatten(const Tensor<T>& t1)
{
    size_t total = 1;
    for (size_t dim : t1.shape)
        total *= dim;

    Tensor<T> out({ total }, T{0 });

    for (size_t i = 0; i < total; ++i) out.data.data[i] = t1.data.data[i];

    return out;
}

template<typename T>
Tensor<T> sum(const Tensor<T>& t1)
{
    size_t total = 1;
    for(auto dim : t1.shape) total *= dim;

    Tensor<T> out({1}, 0);

    for(size_t i = 0; i < total; i++)
        out.data.data[0] += t1.data.data[i];

    return out;
}

#endif // __cplusplus

#endif // TENSOROPS_H