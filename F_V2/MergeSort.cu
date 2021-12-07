#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <float.h>
#include "timing.h"


const int SHARED_SIZE_LIMIT = 1024;

//TEMP HEADER
int * fillArray(int n, int upbound);
__device__   int binarySearchInclusive(  int val,   int *data,   int L,   int stride,  int sortDir);
__device__  int binarySearchExclusive(  int val,  int *data,   int L,   int stride, int sortDir);
__global__ void mergeSortSharedKernel(
     int *d_DstKey,
     int *d_DstVal,
     int *d_SrcKey,
     int *d_SrcVal,
     int arrayLength,  int sortDir
);
int * createKeyArray(int size);
void runCUDA( int *array, int dataSize);
int validateOutput(int size, int tile_width, int *arr);
void printArray(int *arr, int n);
void printArray2(int *arr,int start);
void mergeSortCPU(int* arr, int n);
void mergeCPU(int* arr, int l, int m, int r);
void runCPU(int* inputArray, int size);
void copyArray(int* original, int* copy, int size);
//TEMP HEADER


inline void check_cuda_errors(const char *filename, const int line_number)
{
    cudaThreadSynchronize();
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess)
    {
        printf("CUDA error at %s:%i: %s\n", filename, line_number, cudaGetErrorString(error));
        exit(-1);
    }
}

int * fillArray(int n, int upbound)
{
   int i;
   
   int *ret = (int *)malloc(sizeof(int) * n );

   /* Intializes random number generator */
   //seeds the random number generator used by the function rand.
   srand(time(NULL));

   /* generate n random numbers from 0 to unbound - 1 */
   for( i = 0 ; i < n ; i++ ) {
      int num = (rand() % 100);
      ret[i] = num;
      //printf("%f\n", num);
   }
   return ret;
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
                    for (stride = 1; stride < arrayLength; stride <<= 1)
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

__global__ void merge(int *d_DstKey, int *d_DstVal, int *d_SrcKey, int *d_SrcVal, int sortDir){

}

int * createKeyArray(int size)
{
	int * array = (int*)malloc(sizeof(int) * size);
	for(int i = 0; i < size; i++)
	{
		array[i] = i;	
	}
	return array;
}

void runCUDA( int *array, int dataSize)
{  

   int * keys = createKeyArray(dataSize);
   int num_bytes = dataSize * sizeof(int);

   int *inputKeys; 
	int *outputKeys;
	int *inputValues; 
	int *outputValues;
	int *keyResult;
	int *valResult;

   int tileWidth = 1024;

   //Allocating cuda memory for all arrays
   keyResult = (int*)malloc(num_bytes);
	valResult = (int*)malloc(num_bytes);
	cudaMalloc((void**)&inputKeys, num_bytes);
	cudaMalloc((void**)&outputKeys, num_bytes);
	cudaMalloc((void**)&inputValues, num_bytes);
	cudaMalloc((void**)&outputValues, num_bytes);
	
	if(0 == inputKeys || 0 == outputKeys || 0 == inputValues || 0 == outputValues)
	{
		printf("couldnt allocate memory\n");
		exit(-1);
	}

   	//Setting number of blocks and threads
	//uint numBlocks = batchSize * dataSize / SHARED_SIZE_LIMIT;
   uint numBlocks =  dataSize / SHARED_SIZE_LIMIT;
	//uint numBlocks = ceil(dataSize / (double)tileWidth);
	uint numThreads =  SHARED_SIZE_LIMIT/2;

   //Populating existing arrays
	cudaMemcpy(inputKeys, array, num_bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(inputValues, keys, num_bytes, cudaMemcpyHostToDevice);
	cudaMemset(outputKeys, 0, num_bytes);
	cudaMemset(outputValues, 0, num_bytes);

   cudaEvent_t launch_begin, launch_end;
   cudaEventCreate(&launch_begin);
   cudaEventCreate(&launch_end);

   cudaEventRecord(launch_begin, 0);
   //Calling merging kernel
	mergeSortSharedKernel<<<numBlocks, numThreads>>>(outputKeys, outputValues, inputKeys, inputValues, 1024, 1);
	cudaEventRecord(launch_end, 0);
   cudaEventSynchronize(launch_begin);
   cudaEventSynchronize(launch_end);
   check_cuda_errors(__FILE__, __LINE__);
   cudaDeviceSynchronize();
	//Capturing output from kernel
	cudaMemcpy(keyResult, outputKeys, num_bytes, cudaMemcpyDeviceToHost);
	cudaMemcpy(valResult, outputValues, num_bytes, cudaMemcpyDeviceToHost);

      //To record time at the end
   float time = 0;
   cudaEventElapsedTime(&time, launch_begin, launch_end);
   printf("Done! Time cost is. %f\n", time/1000);

   //Destroy cudaEvents
   cudaEventDestroy(launch_begin);
   cudaEventDestroy(launch_end);

     	printf("SUCCESSFUL\n");
    int pk = 0;
    if(pk ==1){
        for(int i = 0; i < dataSize; i++)
	    {
		    printf("%d, %d\n", keyResult[i], valResult[i]);
	    }
    }

    int isVal = validateOutput(dataSize, 1024, keyResult);

    if(isVal == -1){
       puts("\n\nOutput is valid!");
    }else{
       printf("\n\nOutput is NOT valid! At: %d\n\n", isVal);
       printArray2(keyResult, isVal);
    }

	cudaFree(inputKeys);
	cudaFree(inputValues);
	cudaFree(outputKeys);
	cudaFree(outputValues);
	free(keyResult);
	free(valResult);

}

int validateOutput(int size, int tile_width, int *arr){

   int i;
   for(i = 0; i < size/tile_width; i++){
      int j;
      for(j = 1; j < tile_width; j++){
         if(arr[(i*tile_width)+j] < arr[(i*tile_width)+j-1] && arr[(i*tile_width)+j] != arr[(i*tile_width)+j-1]){
            printf("\n\n%d > %d", arr[(i*tile_width)+j], arr[(i*tile_width)+j-1]);
            return (i*tile_width)+j;
         }
      }
   }
   return -1;

}


void printArray(int *arr, int n){

   int i;

   for(i = 0; i < n; i ++)
      printf("%d ", arr[i]);

   printf("\n");
}

void printArray2(int *arr,int start){

   int i;

   for(i = -25; i < 25; i ++)
      printf("%d ", arr[start+i]);

   printf("\n");
}

//////////////////////////////////////////////////////

//int min(int x, int y) { return (x<y)? x :y; }
 
 
/* Iterative mergesort function to sort arr[0...n-1] */
void mergeSortCPU(int* arr, int n)
{
   int curr_size;  // For current size of subarrays to be merged
                   // curr_size varies from 1 to n/2
   int left_start; // For picking starting index of left subarray
                   // to be merged
 
   // Merge subarrays in bottom up manner.  First merge subarrays of
   // size 1 to create sorted subarrays of size 2, then merge subarrays
   // of size 2 to create sorted subarrays of size 4, and so on.
   for (curr_size=1; curr_size<=n-1; curr_size = 2*curr_size)
   {
       // Pick starting point of different subarrays of current size
       for (left_start=0; left_start<n-1; left_start += 2*curr_size)
       {
           // Find ending point of left subarray. mid+1 is starting
           // point of right
           int mid = min(left_start + curr_size - 1, n-1);
 
           int right_end = min(left_start + 2*curr_size - 1, n-1);
 
           // Merge Subarrays arr[left_start...mid] & arr[mid+1...right_end]
           mergeCPU(arr, left_start, mid, right_end);
       }
   }
}
 
/* Function to merge the two haves arr[l..m] and arr[m+1..r] of array arr[] */
void mergeCPU(int* arr, int l, int m, int r)
{
    int i, j, k;
    int n1 = m - l + 1;
    int n2 =  r - m;
 
    /* create temp arrays */
    //int L[n1], R[n2];
    int* L = (int*)malloc(sizeof(int) * n1);
    int* R = (int*)malloc(sizeof(int) * n2);
 
    /* Copy data to temp arrays L[] and R[] */
    for (i = 0; i < n1; i++)
        L[i] = arr[l + i];
    for (j = 0; j < n2; j++)
        R[j] = arr[m + 1+ j];
 
    /* Merge the temp arrays back into arr[l..r]*/
    i = 0;
    j = 0;
    k = l;
    while (i < n1 && j < n2)
    {
        if (L[i] <= R[j])
        {
            arr[k] = L[i];
            i++;
        }
        else
        {
            arr[k] = R[j];
            j++;
        }
        k++;
    }
 
    /* Copy the remaining elements of L[], if there are any */
    while (i < n1)
    {
        arr[k] = L[i];
        i++;
        k++;
    }
 
    /* Copy the remaining elements of R[], if there are any */
    while (j < n2)
    {
        arr[k] = R[j];
        j++;
        k++;
    }
    free(L);
    free(R);
}

void runCPU(int* inputArray, int size) {
	
    	clock_t now, then;
    	
    	printf("Timing CPU implementation…\n");
    	then = clock();
    	mergeSortCPU(inputArray,size);
    	now =  clock();
    	
    	// measure the time spent on CPU
       float time = 0;
       time = timeCost(then, now);
       
       

       printf(" done. CPU time cost in second: %f\n", time);
}


void copyArray(int* original, int* copy, int size) {
   int i;
   for(i = 0; i < size; i++){

      copy[i] = original[i];
   }
}



int main(int argc, char *argv[]){
   int size =  33554432;
   int tile_width = 1024;
   int* arr = fillArray(size, 10000);

   //int* arrCPU = (int*)malloc(sizeof(int) * size );
   //copyArray(arr, arrCPU, size);
   //runCPU(arrCPU, size);

   //int arr_size = sizeof(arr) / sizeof(arr[0]);

    runCUDA( arr, size); // Array, Elements, Tile size
    //mergeSort(arr, 0, arr_size - 1);
    //printArray(arr, size);
}


