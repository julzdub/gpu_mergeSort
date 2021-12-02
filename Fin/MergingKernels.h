#include <stdlib.h>
#include <stdio.h>

const int SHARED_SIZE_LIMIT = 1024;

template<uint sortDir> __global__ void mergeSortSharedKernel(uint *d_DstKey, uint *d_DstVal, uint *d_SrcKey, uint *d_SrcVal, uint arrayLength);
template<uint sortDir> static inline __device__ uint binarySearchExclusive(uint val, uint *data, uint L, uint stride);
template<uint sortDir> static inline __device__ uint binarySearchInclusive(uint val, uint *data, uint L, uint stride);
