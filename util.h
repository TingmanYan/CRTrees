#ifndef __UTILS_H__
#define __UTILS_H__

#include <cassert>
#include <chrono>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <curand_kernel.h>
#include <iomanip>
#include <iostream>
#include <string>
#include <numeric>
#include <thrust/execution_policy.h>
#include <thrust/gather.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include <thrust/tuple.h>

#define checkCudaErrors(val) check((val), #val, __FILE__, __LINE__)

template<typename T>
void
check(T err, const char* const func, const char* const file, const int line)
{
  if (err != cudaSuccess) {
    std::cerr << "CUDA error at: " << file << ":" << line << std::endl;
    std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
    exit(1);
  }
}

inline __host__ __device__ float2
operator+(const float2& a, const float2& b)
{
  return make_float2(a.x + b.x, a.y + b.y);
}
inline __host__ __device__ void
operator+=(float2& a, const float2& b)
{
  a.x += b.x;
  a.y += b.y;
}

inline __host__ __device__ float3
operator+(const float3& a, const float3& b)
{
  return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}
inline __host__ __device__ void
operator+=(float3& a, const float3& b)
{
  a.x += b.x;
  a.y += b.y;
  a.z += b.z;
}

inline __host__ __device__ float4
operator+(const float4& a, const float4& b)
{
  return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}
inline __host__ __device__ void
operator+=(float4& a, const float4& b)
{
  a.x += b.x;
  a.y += b.y;
  a.z += b.z;
  a.w += b.w;
}

inline __host__ __device__ float4 operator*(const float4& a, const float4& b)
{
  return make_float4(a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w);
}
inline __host__ __device__ float4 operator*(const float4& a, const float& b)
{
  return make_float4(a.x * b, a.y * b, a.z * b, a.w * b);
}
inline __host__ __device__ void
operator*=(float4& a, const float4& b)
{
  a.x *= b.x;
  a.y *= b.y;
  a.z *= b.z;
  a.w *= b.w;
}

inline __host__ __device__ bool
operator<(const int2& a, const int2& b)
{
  return a.x == b.x ? a.y < b.y : a.x < b.x;
  // if (a.x != b.x) {
  //   return a.x < b.x;
  // } else {
  //   return a.y < b.y;
  // }
}

inline __host__ __device__ bool
operator==(const int2& a, const int2& b)
{
  return a.x == b.x && a.y == b.y;
}

inline __host__ __device__ bool
operator!=(const int2& a, const int2& b)
{
  return ((a.x != b.x) || (a.y != b.y));
}

__device__ __forceinline__ float
atomicMinFloat(float* addr, float value)
{
  float old;
  old =
    (value >= 0)
      ? __int_as_float(atomicMin((int*)addr, __float_as_int(value)))
      : __uint_as_float(atomicMax((unsigned int*)addr, __float_as_uint(value)));

  return old;
}

// TODO: check if this is correct     √√√ DONE it is correct based on a test
__device__ __forceinline__ float
atomicMaxFloat(float* addr, float value)
{
  float old;
  old =
    (value >= 0)
      ? __int_as_float(atomicMax((int*)addr, __float_as_int(value)))
      : __uint_as_float(atomicMin((unsigned int*)addr, __float_as_uint(value)));

  return old;
}

#endif
