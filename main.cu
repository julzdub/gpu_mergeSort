#include <stdlib.h>
#include <stdio.h>

#include "cpuMergeSort.h"
//include header files for .cu

int willPrint = 0;

void usage()
{
	printf("Usage: <./fileName> <arraySize> <printOptional>");
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
	int *inputArray = makeArray(arraySize);
	
	//Call kernel setup
	runCuda(arraySize, *inputArray);
	
	//Call cpu setup
	mergeSort(inputArray, 0, arraySize - 1);
}
