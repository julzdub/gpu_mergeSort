#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <float.h>
#include "timing.h"
#include "MergeSortSetup.h"
#include "Kernels.h"
  /////////////////////////////////////////////////////////////////////////////////////
 ////////////////////////////RUN CUDA SORT TILE FUNCTION//////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////

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
//TODO:
	
	
	cudaFree(inputKeys);
	cudaFree(inputValues);
	cudaFree(outputKeys);
	cudaFree(outputValues);
	free(keyResult);
	free(valResult);

}


  /////////////////////////////////////////////////////////////////////////////////////
 ////////////////////////////RUN CUDA MERGE TILE FUNCTION/////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////


void runMerge(int * arr, int n, int tile_width, float time){
   int *h_in = arr; //Filled array
   int *h_out = (int *)malloc(sizeof(int) * n ); //Allocate output mem
   int *d_in;//Device in pointer
   int *d_out;

   cudaMalloc((void**)&d_in, sizeof(int) *n);//Allocate in and out mem
   cudaMalloc((void**)&d_out, sizeof(int) *n);//Allocate in and out mem

   cudaMemcpy(d_in, h_in, sizeof(int)*n, cudaMemcpyHostToDevice);//Copy in array to device
   
   int t_width = tile_width;
   int num_block;


   int* temp;

   cudaEvent_t launch_begin, launch_end;
   cudaEventCreate(&launch_begin);
   cudaEventCreate(&launch_end);

   cudaEventRecord(launch_begin, 0);
   while(t_width <= n/2){
      puts("\n\n RAN \n\n");
      num_block = n /(t_width * 2);
      mergeKernel<<<num_block,1>>>(d_in, d_out, n, t_width);
      t_width *= 2;
      
      temp = d_in;
      d_in = d_out;
      d_out = temp;
   }
   cudaEventRecord(launch_end, 0);
   cudaEventSynchronize(launch_begin);
   cudaEventSynchronize(launch_end);

   //To record time at the end
   float time2 = 0;
   cudaEventElapsedTime(&time2, launch_begin, launch_end);
   printf("Done! Time cost is. %f\n", time2/1000);

   //Destroy cudaEvents
   cudaEventDestroy(launch_begin);

   cudaMemcpy(h_out, d_in, sizeof(int)*n, cudaMemcpyDeviceToHost);

   int isVal2 = validateOutput(n, n, h_out);

    if(isVal2 == -1){
       puts("\n\nOutput is valid!");
    }else{
       printf("\n\nOutput is NOT valid! At: %d\n\n", isVal2);
       printArray2(h_out, isVal2);
    }
    puts("\n\n\n\n\n");

    //printArray(h_out, n);

   cudaFree(d_in);
   cudaFree(d_out);
}


  /////////////////////////////////////////////////////////////////////////////////////
 /////////////////////////////////HELPER FUNCTIONS////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////


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


int * createKeyArray(int size)
{
	int * array = (int*)malloc(sizeof(int) * size);
	for(int i = 0; i < size; i++)
	{
		array[i] = i;	
	}
	return array;
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






















