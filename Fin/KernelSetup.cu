#include <stdlib.h>
#include <stdio.h>
#include "MergingKernels.h"
#include "KernelSetup.h"

void runCuda(uint *array, uint *keys, uint dataSize, uint num_bytes)
{
	uint *inputKeys; 
	uint *outputKeys;
	uint *inputValues; 
	uint *outputValues;
	uint *keyResult;
	uint *valResult;
	
	//TODO: Make sure tile size is not larger than the array itself
	uint tileWidth = 4;
	
	
	//Allocating cuda memory for all arrays
	keyResult = (uint*)malloc(num_bytes);
	valResult = (uint*)malloc(num_bytes);
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
	uint numBlocks = ceil(dataSize / (uint)tileWidth);
	uint numThreads = SHARED_SIZE_LIMIT/2;
	printf("Number of blocks: %d\n", numBlocks);
	
	//Populating existing arrays
	cudaMemcpy(inputKeys, array, num_bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(inputValues, keys, num_bytes, cudaMemcpyHostToDevice);
	cudaMemset(outputKeys, 0, num_bytes);
	cudaMemset(outputValues, 0, num_bytes);
	
	//Calling merging kernel
	mergeSortSharedKernel<1U><<<numBlocks, numThreads>>>(outputKeys, outputValues, inputKeys, inputValues, dataSize);
	
	//Capturing output from kernel
	cudaMemcpy(keyResult, outputKeys, num_bytes, cudaMemcpyDeviceToHost);
	cudaMemcpy(valResult, outputValues, num_bytes, cudaMemcpyDeviceToHost);
	
	
	printf("SUCCESSFUL\n");
	for(int i = 0; i < dataSize; i++)
	{
		printf("%d, %d\n", keyResult[i], valResult[i]);
	}
	
	cudaFree(inputKeys);
	cudaFree(inputValues);
	cudaFree(outputKeys);
	cudaFree(outputValues);
	free(keyResult);
	free(valResult);
}
