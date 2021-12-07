#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <float.h>
#include "cpuMergeSort.h"
#include "timing.h"
#include "Kernels.h"


#define W (sizeof(uint) * 8)

void copyArray(int* original, int* copy, int size);
void printOutput(FILE * f_out, int * arr, int n);

int willPrint = 0;

void usage()
{
	printf("Usage: <./fileName> <arraySize> <printOptional>");
}

void printOutput(FILE * f_out, int * arr, int n) {
   int i;
   
   for(i = 0; i < n; i ++)
      fprintf(f_out, "%d ", arr[i]);

   
}

int * fillArray(int n, int upbound, int realSize)
{
   int i;
   
   int *ret = (int *)malloc(sizeof(int) * realSize);

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

void runCPU(int * inputArray, int start, int end) {
	
    	clock_t now, then;
    	
    	printf("\nTiming CPU implementation…\n");
    	then = clock();
    	mergeSort(inputArray, start, end);
    	now =  clock();
    	
    	// measure the time spent on CPU
       float time = 0;
       time = timeCost(then, now);
       
       

       printf("\nCPU time cost in second: %f\n", time);
}

void copyArray(int* original, int* copy, int size) {
   int i;
   for(i = 0; i < size; i++){

      copy[i] = original[i];
   }
}

inline int nextPowerOfTwo(int x)
	{
    	return 1U << ( W - __builtin_clz(x - 1) );
	}

	void pad(int* array, int realSize, int wPadding){
		int i;
		for(i = realSize; i < wPadding; i++){
			array[i] = -1;
		}
	}

int main(int argc, char *argv[])
{
        FILE * output_CPU = NULL;
        output_CPU = fopen("output_CPU.txt", "w");
        FILE * output_GPU = NULL;
        output_GPU = fopen("output_GPU.txt", "w");
	
	//get user input (number of elements and printing option p)
	if(argc < 2 || argc > 3)
	{
		usage();
		return 1;
	}
	else if(argc == 2)
	{
		willPrint = 0;
	}
	else if(argv[2][0] == 'p')
	{
		willPrint = 1;
	}
	else
	{
		usage();
		return 1;
	}
	
	int arraySize = atoi(argv[1]);
	
	if(!arraySize || arraySize < 1024)
	{
		printf("Array Size invalid please enter a value greater than 1023\n");
		usage();
		return 1;
	}
	
	//Create and fill input array
	int wPadding = nextPowerOfTwo(arraySize);
	//printf("Pad:%d\n", wPadding);
	int *cpuArray = fillArray(arraySize, 99999, wPadding);
	int *gpuArray = (int*)malloc(sizeof(int) *wPadding);
	pad(cpuArray, arraySize, wPadding);
	copyArray(cpuArray, gpuArray, wPadding);


	//Call cpu setup
	runCPU(cpuArray, 0, wPadding - 1);
	int* ptr = cpuArray+(wPadding -arraySize);

	
	//Call kernel setup
	printf("\nTiming GPU implementation…\n");
	int * result = runCUDA(gpuArray, wPadding);
	int* ptr2 = result+(wPadding-arraySize);
	printf("\nWriting results to files output_CPU.txt and output_GPU.txt\n");
	printOutput(output_CPU, ptr, arraySize);
    printOutput(output_GPU, ptr2, arraySize);
    	
    	if(willPrint) {
		printArray(ptr, arraySize);
		printArray(ptr2, arraySize);
	}
	//Call cpu setup


	free(cpuArray);
	free(result);

	return 0;
}
