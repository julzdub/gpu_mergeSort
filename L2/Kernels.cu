#include "Kernels.h"

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
                    for (stride = 1; stride < arrayLength; stride <<= 1) //mult
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

__global__ void mergeKernel(int* in, int* out, int dataSize, int tile_size){

   //unsigned int tid = threadIdx.x;
   unsigned int i = (blockIdx.x*2)*blockDim.x + threadIdx.x; //change based on number of elements imported by thread

   int startIndex = i*tile_size;

   int startA = startIndex;
   int startB = startIndex + tile_size;

   int curB = startB;
   int curA = startA;

   int x;
   for(x = startA; x < startB+tile_size; x++){
      if(curA < startB && curB < startB+tile_size && in[curA] < in[curB] ){
         out[x] = in[curA];
         curA++;
      }else if(curA < startB && curB == startB+tile_size){
         out[x] = in[curA];
         curA++;
      }else if(curA < startB && curB < startB+tile_size && in[curB] <= in[curA] ){
         out[x] = in[curB];
         curB++;
      }else if(curB < startB+tile_size && curA == startB){
         out[x] = in[curB];
         curB++;
      }
   }

   
}


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

    int isVal = validateOutput(dataSize, 1024, keyResult);

    if(isVal == -1){
       puts("\n\nOutput is valid!");
    }else{
       printf("\n\nOutput is NOT valid! At: %d\n\n", isVal);
       printArray2(keyResult, isVal);
    }

//TODO:
	
    runMerge(keyResult, dataSize, 1024, time);
	
	cudaFree(inputKeys);
	cudaFree(inputValues);
	cudaFree(outputKeys);
	cudaFree(outputValues);
	//free(keyResult);
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
   printf("Final time: %f\n", (time+time2)/1000);

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
