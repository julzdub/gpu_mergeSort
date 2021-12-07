#include "cpuMergeSort.h"

void mergeSort(int * inputArray, int start, int end) {
    int mid = (start + end)/2;
    if(start < end) {
        //sort left part
        mergeSort(inputArray, start, mid);
        //sort right part
        mergeSort(inputArray, mid + 1, end);

        //merge together
        merge(inputArray, start, mid, end);
    }
}

void merge(int * inputArray, int start, int mid, int end) {

    int i, j, k;

    //int temp[end-start+1];
    int* temp = (int*)malloc(sizeof(int)*(end-start+1));

    i = start; j = mid+1; k = 0;
    //Compare elements of two sorted arrays and store in sorted array

    while(i <= mid && j <= end) {
        if(inputArray[i] < inputArray[j]) {
            temp[k++] = inputArray[i++];
        }
        else {
            temp[k++] = inputArray[j++];
        }
    }
    
    while( i <= mid) {
        temp[k++] = inputArray[i++];
    }
    while(j <= end) {
        temp[k++] = inputArray[j++];
    }

    //copy temp into original array
    for(i = 0; i < k; i++) {
        inputArray[i + start] = temp[i];
    }
    free(temp);
}