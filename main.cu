#include <stdlib.h>
#include <stdio.h>
#include "cpuMergeSort.h"

int willPrint = 0;

void printArray(int * arr, int n) {

	int i;
 
	for(i = 0; i < n; i++) {
	   printf("%d ", arr[i]);
	}

	printf("\n");
}

void usage()
{
	printf("Usage: <./fileName> <arraySize> <printOptional>");
}

int * fillArray(int n)
{
   int i;
   
   int *ret = (int *)malloc(sizeof(int) * n);

   /* Intializes random number generator */
   //seeds the random number generator used by the function rand.
   srand(time(NULL));

   /* generate n random numbers from 0 to unbound - 1 */
   for( i = 0 ; i < n ; i++ ) {
      ret[i] = rand();
   }
   return ret;
}

int main(int argc, char *argv[])
{
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
	
	if(!arraySize || arraySize > 1024)
	{
		printf("Array Size Too Large");
		usage();
		return 1;
	}
	
	//Create and fill input array
	int *inputArray = fillArray(arraySize);
	
	//Call kernel setup
	//runCuda(arraySize, *inputArray);
	
	if(willPrint) {
		printArray(inputArray, arraySize);
	}
	//Call cpu setup
	mergeSort(inputArray, 0, arraySize - 1);
	printArray(inputArray, arraySize);

	free(inputArray);

	return 0;
}
