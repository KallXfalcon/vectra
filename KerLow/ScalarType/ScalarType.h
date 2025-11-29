
/*
 *
 * ScalarType.h (v1.0)
 * 
 * Copyright(C) 2025 KallXfalcon
 * GitHub : https://github.com/KallXfalcon
 * 
 * This program is licensed under the GNU General Public License v3.0 (GPL v3.0)
 * See <https://www.gnu.org/licenses> for details.
 * 
 * Main of Data type Header
 * Include SIMD ScalarType
 * This file is a part of VECTRA framework
 * 
*/

#pragma once
#ifndef SCALARTYPE_H
#define SCALARTYPE_H

#ifdef __cplusplus

#include <immintrin.h>
#include <type_traits>
#include <cstring>
#include <cstdint>

namespace vectra {
    using int8   = char;
    using int16  = short;
    using int32  = int;
    using float32= float;
    using float64= double;
}

#ifndef USE_VECTRA_SSE
#define USE_VECTRA_SSE
#endif

#ifndef USE_VECTRA_AVX
#endif

namespace m128 {
template<typename T> struct Scalar { using type = void; };

template<> struct Scalar<vectra::int8>    { using type = __m128i; };
template<> struct Scalar<vectra::int16>   { using type = __m128i; };
template<> struct Scalar<vectra::int32>   { using type = __m128i; };
template<> struct Scalar<vectra::float32> { using type = __m128;  };
template<> struct Scalar<vectra::float64> { using type = __m128d; };

template<typename T>
auto simd_set(const T value){
    if constexpr(std::is_same_v<T, vectra::int8>)     return _mm_set1_epi8(static_cast<char>(value));
    if constexpr(std::is_same_v<T, vectra::int16>)    return _mm_set1_epi16(static_cast<short>(value));
    if constexpr(std::is_same_v<T, vectra::int32>)    return _mm_set1_epi32(static_cast<int>(value));
    if constexpr(std::is_same_v<T, vectra::float32>)  return _mm_set1_ps(static_cast<float>(value));
    if constexpr(std::is_same_v<T, vectra::float64>)  return _mm_set1_pd(static_cast<double>(value));
}
}

namespace m256 {
template<typename T> struct Scalar { using type = void; };

template<> struct Scalar<vectra::int8>    { using type = __m256i; };
template<> struct Scalar<vectra::int16>   { using type = __m256i; };
template<> struct Scalar<vectra::int32>   { using type = __m256i; };
template<> struct Scalar<vectra::float32> { using type = __m256;  };
template<> struct Scalar<vectra::float64> { using type = __m256d; };

template<typename T>
auto simd_set(const T value){
    if constexpr(std::is_same_v<T, vectra::int8>)     return _mm256_set1_epi8(static_cast<char>(value));
    if constexpr(std::is_same_v<T, vectra::int16>)    return _mm256_set1_epi16(static_cast<short>(value));
    if constexpr(std::is_same_v<T, vectra::int32>)    return _mm256_set1_epi32(static_cast<int>(value));
    if constexpr(std::is_same_v<T, vectra::float32>)  return _mm256_set1_ps(static_cast<float>(value));
    if constexpr(std::is_same_v<T, vectra::float64>)  return _mm256_set1_pd(static_cast<double>(value));
}
}

template<typename T>
class ScalarType {
private:
#if defined(USE_VECTRA_AVX)
    using simd_type = typename m256::Scalar<T>::type;
    static constexpr size_t aligned_bytes = 32;
#elif defined(USE_VECTRA_SSE)
    using simd_type = typename m128::Scalar<T>::type;
    static constexpr size_t aligned_bytes = 16;
#else
    using simd_type = T;
    static constexpr size_t aligned_bytes = alignof(T);
#endif
    alignas(aligned_bytes) union {
        T scalar;
        simd_type simd;
    } u;

    static constexpr bool simd_is_same = std::is_same_v<simd_type, T>;

public:
    ScalarType(const T& val) {
        u.scalar = val;
        if constexpr (!simd_is_same) {
#if defined(USE_VECTRA_AVX)
            u.simd = m256::simd_set(val);
#elif defined(USE_VECTRA_SSE)
            u.simd = m128::simd_set(val);
#endif
        }
    }
    template<typename U = simd_type, typename = std::enable_if_t<!std::is_same_v<U, T>>>
    ScalarType(const simd_type& val) {
        u.simd = val;

        if constexpr (std::is_same_v<T, float>) {
#if defined(USE_VECTRA_AVX)
            alignas(32) float buf[8];
            _mm256_store_ps(buf, val);
            u.scalar = buf[0];
#elif defined(USE_VECTRA_SSE)
            alignas(16) float buf[4];
            _mm_store_ps(buf, val);
            u.scalar = buf[0];
#endif
        } else if constexpr (std::is_same_v<T, double>) {
#if defined(USE_VECTRA_AVX)
            alignas(32) double buf[4];
            _mm256_store_pd(buf, val);
            u.scalar = buf[0];
#elif defined(USE_VECTRA_SSE)
            alignas(16) double buf[2];
            _mm_store_pd(buf, val);
            u.scalar = buf[0];
#endif
        } else {
            alignas(aligned_bytes) char tmp[sizeof(simd_type)];
            std::memcpy(tmp, &val, sizeof(simd_type));
            u.scalar = *reinterpret_cast<const T*>(tmp);
        }
    }

    ScalarType& operator=(const T& val) {
        u.scalar = val;
        if constexpr (!simd_is_same) {
#if defined(USE_VECTRA_AVX)
            u.simd = m256::simd_set(val);
#elif defined(USE_VECTRA_SSE)
            u.simd = m128::simd_set(val);
#endif
        }
        return *this;
    }

    template<typename U = simd_type, typename = std::enable_if_t<!std::is_same_v<U, T>>>
    ScalarType& operator=(const simd_type& val) {
        u.simd = val;
        if constexpr (std::is_same_v<T, float>) {
#if defined(USE_VECTRA_AVX)
            alignas(32) float buf[8];
            _mm256_store_ps(buf, val);
            u.scalar = buf[0];
#elif defined(USE_VECTRA_SSE)
            alignas(16) float buf[4];
            _mm_store_ps(buf, val);
            u.scalar = buf[0];
#endif
        } else if constexpr (std::is_same_v<T, double>) {
#if defined(USE_VECTRA_AVX)
            alignas(32) double buf[4];
            _mm256_store_pd(buf, val);
            u.scalar = buf[0];
#elif defined(USE_VECTRA_SSE)
            alignas(16) double buf[2];
            _mm_store_pd(buf, val);
            u.scalar = buf[0];
#endif
        } else {
            alignas(aligned_bytes) char tmp[sizeof(simd_type)];
            std::memcpy(tmp, &val, sizeof(simd_type));
            u.scalar = *reinterpret_cast<const T*>(tmp);
        }
        return *this;
    }
    T get_scalar() const noexcept { return u.scalar; }

    template<typename U = simd_type, typename = std::enable_if_t<!std::is_same_v<U, T>>>
    simd_type get_simd() const noexcept { return u.simd; }
};

#endif // __cplusplus
#endif // SCALARTYPE_H