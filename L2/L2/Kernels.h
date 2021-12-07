#ifndef Kernels_H
#define Kernels_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <float.h>
#include "timing.h"

__global__ void mergeSortSharedKernel(int *d_DstKey, int *d_DstVal, int *d_SrcKey,int *d_SrcVal, 
int arrayLength,  int sortDir);

__device__   int binarySearchInclusive( int val,   int *data, int L, int stride,  int sortDir);

__device__  int binarySearchExclusive( int val,  int *data,   int L,   int stride,  int sortDir);

__global__ void mergeKernel(int* in, int* out, int dataSize, int tile_size);

const int SHARED_SIZE_LIMIT = 1024;

int * runCUDA( int *array, int dataSize);
int * runMerge(int * arr, int n, int tile_width, float time);
inline void check_cuda_errors(const char *filename, const int line_number);
int * createKeyArray(int size);
int validateOutput(int size, int tile_width, int *arr);
void printArray(int *arr, int n);
void printArray2(int *arr,int start);

#endif
