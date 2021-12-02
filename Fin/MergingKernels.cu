#include <stdlib.h>
#include <stdio.h>
#include "MergingKernels.h"
#include "KernelSetup.h"


template<uint sortDir> __global__ void mergeSortSharedKernel(
    uint *d_DstKey,
    uint *d_DstVal,
    uint *d_SrcKey,
    uint *d_SrcVal,
    uint arrayLength   //actual length for this block to process in the input array.
)
{
    __shared__ uint s_key[SHARED_SIZE_LIMIT];  //SHARED_SIZE_LIMIT: number of elements a thread block processes
    __shared__ uint s_val[SHARED_SIZE_LIMIT];   // Actually, the number of threads per block is SHARED_SIZE_LIMIT/2
    
    
    //Sets pointers to thread numberth element of all arrays
    d_SrcKey += blockIdx.x * SHARED_SIZE_LIMIT + threadIdx.x;
    d_SrcVal += blockIdx.x * SHARED_SIZE_LIMIT + threadIdx.x;
    d_DstKey += blockIdx.x * SHARED_SIZE_LIMIT + threadIdx.x;
    d_DstVal += blockIdx.x * SHARED_SIZE_LIMIT + threadIdx.x;
    
    
    //populates array so comparisons can be made
    s_key[threadIdx.x +                       0] = d_SrcKey[                      0];
    s_val[threadIdx.x +                        0] =  d_SrcVal[                      0];
    s_key[threadIdx.x + (SHARED_SIZE_LIMIT / 2)] = d_SrcKey[(SHARED_SIZE_LIMIT / 2)];
    s_val[threadIdx.x + (SHARED_SIZE_LIMIT / 2)]  = d_SrcVal[(SHARED_SIZE_LIMIT / 2)];
    
    for (uint stride = 1; stride < arrayLength; stride <<= 1)//'<<=' means multiplies by two or (shifts 									over two bits)
    {
        uint  lPos = threadIdx.x  &  (stride - 1);// '&' with two operands only returns that # if they 								are both that #
        
        //iterates through shared memory array and points to array (thread # - lpos always 0)
        uint *baseKey = s_key + 2 * (threadIdx.x - lPos);
        uint *baseVal = s_val + 2 *   (threadIdx.x - lPos);
        
        
        __syncthreads();
        uint keyA = baseKey[lPos +      0];
        uint valA = baseVal[lPos +      0];
        uint keyB = baseKey[lPos + stride];
        uint valB = baseVal[lPos + stride];
        uint posA = binarySearchExclusive<sortDir>(keyA, baseKey + stride, stride, stride) + lPos;
        uint posB = binarySearchInclusive<sortDir>(keyB, baseKey +      0,    stride, stride) + lPos;

      
        __syncthreads();
        baseKey[posA] = keyA;
        baseVal[posA] = valA;
        baseKey[posB] = keyB;
        baseVal[posB] = valB;
    }
    
    __syncthreads();
    d_DstKey[                      0] = s_key[threadIdx.x +                       0];
    d_DstVal[                       0]  = s_val[threadIdx.x +                       0];
    d_DstKey[(SHARED_SIZE_LIMIT / 2)] = s_key[threadIdx.x + (SHARED_SIZE_LIMIT / 2)];
    d_DstVal[(SHARED_SIZE_LIMIT / 2)] =   s_val[threadIdx.x + (SHARED_SIZE_LIMIT / 2)];
}

template<uint sortDir> static inline __device__ uint binarySearchInclusive(uint val, uint *data, uint L, uint stride)
{
    if (L == 0)
    {
        return 0;
    }

    uint pos = 0;

    for (; stride > 0; stride >>= 1)
    {
        uint newPos = umin(pos + stride, L);

        if ((sortDir && (data[newPos - 1] <= val)) || (!sortDir && (data[newPos - 1] >= val)))
        {
            pos = newPos;
        }
    }

    return pos;
}

template<uint sortDir> static inline __device__ uint binarySearchExclusive(uint val, uint *data, uint L, uint stride)
{
    if (L == 0)
    {
        return 0;
    }

    uint pos = 0;

    for (; stride > 0; stride >>= 1)
    {
        uint newPos = umin(pos + stride, L);

        if ((sortDir && (data[newPos - 1] < val)) || (!sortDir && (data[newPos - 1] > val)))
        {
            pos = newPos;
        }
    }

    return pos;
}
