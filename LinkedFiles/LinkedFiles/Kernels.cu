#include "Kernels.h"
#include "MergeSortSetup.h"

__global__ void mergeSortSharedKernel(
     int *d_DstKey,
     int *d_DstVal,
     int *d_SrcKey,
     int *d_SrcVal,
     int arrayLength,  int sortDir
)
{
                    __shared__  int s_key[SHARED_SIZE_LIMIT];
                    __shared__  int s_val[SHARED_SIZE_LIMIT];
 
 
                    d_SrcKey += blockIdx.x * SHARED_SIZE_LIMIT + threadIdx.x;
                    d_SrcVal += blockIdx.x * SHARED_SIZE_LIMIT + threadIdx.x;
                    d_DstKey += blockIdx.x * SHARED_SIZE_LIMIT + threadIdx.x;
                    d_DstVal += blockIdx.x * SHARED_SIZE_LIMIT + threadIdx.x;
                    s_key[threadIdx.x +                       0] = d_SrcKey[                      0];
                    s_val[threadIdx.x +                       0] = d_SrcVal[                      0];
                    s_key[threadIdx.x + (SHARED_SIZE_LIMIT / 2)] = d_SrcKey[(SHARED_SIZE_LIMIT / 2)];
                    s_val[threadIdx.x + (SHARED_SIZE_LIMIT / 2)] = d_SrcVal[(SHARED_SIZE_LIMIT / 2)];
                                int stride = 1;
                    for (stride = 1; stride < arrayLength; stride <<= 1) //mult
                    {
                                 int     lPos = threadIdx.x & (stride - 1);
                                 int *baseKey = s_key + 2 * (threadIdx.x - lPos);
                                 int *baseVal = s_val + 2 * (threadIdx.x - lPos);
 
                                __syncthreads();
                                 int keyA = baseKey[lPos +      0];
                                 int valA = baseVal[lPos +      0];
                                 int keyB = baseKey[lPos + stride];
                                 int valB = baseVal[lPos + stride];
                                 int posA = binarySearchExclusive(keyA, baseKey + stride, stride, stride, sortDir) + lPos;
                                 int posB = binarySearchInclusive(keyB, baseKey +      0, stride, stride, sortDir) + lPos;
 
                                __syncthreads();
                                baseKey[posA] = keyA;
                                baseVal[posA] = valA;
                                baseKey[posB] = keyB;
                                baseVal[posB] = valB;
                    }
 
                    __syncthreads();
 
                    d_DstKey[                      0] = s_key[threadIdx.x +                       0];
                    d_DstVal[                      0] = s_val[threadIdx.x +                       0];
                    d_DstKey[(SHARED_SIZE_LIMIT / 2)] = s_key[threadIdx.x + (SHARED_SIZE_LIMIT / 2)];
                    d_DstVal[(SHARED_SIZE_LIMIT / 2)] = s_val[threadIdx.x + (SHARED_SIZE_LIMIT / 2)];
 
 
                //            printf("This is that: %d %d\n", s_key[threadIdx.x], s_val[threadIdx.x] );
}

__device__   int binarySearchInclusive( int val,   int *data, int L, int stride,  int sortDir)
{
    if (L == 0)
    {
        return 0;
    }
 
    int pos = 0;
 
    for (; stride > 0; stride >>= 1)
    {
          int newPos = umin(pos + stride, L);
 
        if ((sortDir && (data[newPos - 1] <= val)) || (!sortDir && (data[newPos - 1] >= val)))
        {
            pos = newPos;
        }
    }
 
    return pos;
}
 
__device__  int binarySearchExclusive( int val,  int *data,   int L,   int stride,  int sortDir)
{
    if (L == 0)
    {
        return 0;
    }
 
     int pos = 0;
 
    for (; stride > 0; stride >>= 1)
    {
          int newPos = umin(pos + stride, L);
 
        if((sortDir && (data[newPos - 1] < val)) || (!sortDir && (data[newPos - 1] > val)))
        {
            pos = newPos;
                                //printf("Pos: %d\n", pos);
        }
    }
 
    return pos;
}

__global__ void mergeKernel(int* in, int* out, int dataSize, int tile_size){

   //unsigned int tid = threadIdx.x;
   unsigned int i = (blockIdx.x*2)*blockDim.x + threadIdx.x; //change based on number of elements imported by thread

   int startIndex = i*tile_size;

   int startA = startIndex;
   int startB = startIndex + tile_size;

   int curB = startB;
   int curA = startA;

   int x;
   for(x = startA; x < startB+tile_size; x++){
      if(curA < startB && curB < startB+tile_size && in[curA] < in[curB] ){
         out[x] = in[curA];
         curA++;
      }else if(curA < startB && curB == startB+tile_size){
         out[x] = in[curA];
         curA++;
      }else if(curA < startB && curB < startB+tile_size && in[curB] <= in[curA] ){
         out[x] = in[curB];
         curB++;
      }else if(curB < startB+tile_size && curA == startB){
         out[x] = in[curB];
         curB++;
      }
   }

   
}
