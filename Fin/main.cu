#include <stdlib.h>
#include <stdio.h>
#include "MergingKernels.h"
#include "KernelSetup.h"

int main()
{
	uint dimx = 16;
	uint array[16] = {4, 5, 8, 9, 1, 3, 4, 6, 4, 5, 8, 9, 1, 2, 4, 6 };
	uint keys[16] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
	uint num_bytes = dimx * sizeof(uint);
	
	runCuda(array, keys, dimx, num_bytes);
	
	return 1;
}
