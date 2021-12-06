#ifndef CPUMERGESORT_H
#define CPUMERGESORT_H

#include <stdlib.h>
#include <stdio.h>

void mergeSort(int * inputArray, int start, int end);
void merge(int * inputArray, int start, int mid, int end);

#endif