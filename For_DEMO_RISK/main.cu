#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <float.h>
#include "cpuMergeSort.h"
#include "timing.h"
#include "Kernels.h"

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

void runCPU(int * inputArray, int start, int end) {
	
    	clock_t now, then;
    	
    	printf("Timing CPU implementation…\n");
    	then = clock();
    	mergeSort(inputArray, start, end);
    	now =  clock();
    	
    	// measure the time spent on CPU
       float time = 0;
       time = timeCost(then, now);
       
       

       printf("CPU time cost in second: %f\n", time);
}

void copyArray(int* original, int* copy, int size) {
   int i;
   for(i = 0; i < size; i++){

      copy[i] = original[i];
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
	
	if(!arraySize)
	{
		printf("Array Size Too Large");
		usage();
		return 1;
	}
	
	//Create and fill input array
	int *cpuArray = fillArray(arraySize, 200);
	int *gpuArray = (int*)malloc(sizeof(int) *arraySize);
	copyArray(cpuArray, gpuArray, arraySize);

	//Call cpu setup
	runCPU(cpuArray, 0, arraySize - 1);
	printOutput(output_CPU, cpuArray, arraySize);

	
	//Call kernel setup
	printf("Timing CPU implementation…\n");
	int * result = runCUDA(gpuArray, arraySize);
    printOutput(output_GPU, result, arraySize);
    	
    	if(willPrint) {
		printArray(cpuArray, arraySize);
		printArray(result, arraySize);
	}
	//Call cpu setup


	free(cpuArray);
	free(result);

	return 0;
}
