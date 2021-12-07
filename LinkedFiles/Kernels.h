#include <stdio.h>
#include <stdlib.h>
#include <math.h>

__global__ void mergeSortSharedKernel(int *d_DstKey, int *d_DstVal, int *d_SrcKey,int *d_SrcVal, 
int arrayLength,  int sortDir);

__device__   int binarySearchInclusive( int val,   int *data, int L, int stride,  int sortDir);

__device__  int binarySearchExclusive( int val,  int *data,   int L,   int stride,  int sortDir);

__global__ void mergeKernel(int* in, int* out, int dataSize, int tile_size);