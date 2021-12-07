#ifndef MERGESORTSETUP_H
#define MERGESORTSETUP_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <float.h>
#include "timing.h"

const int SHARED_SIZE_LIMIT = 1024;

void runCUDA( int *array, int dataSize);
void runMerge(int * arr, int n, int tile_width, float time);
inline void check_cuda_errors(const char *filename, const int line_number);
int * createKeyArray(int size);
int validateOutput(int size, int tile_width, int *arr);
void printArray(int *arr, int n);
void printArray2(int *arr,int start);

#endif