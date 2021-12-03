#include <stdio.h>
#include <stdlib.h>
#include <math.h>

//TEMP HEADER
void printArray(uint *arr, uint n);
void runCUDA( uint *arr, uint n, uint tile_width);
__global__ void mergeSortKernel(uint *in, uint *out, uint n);
uint * fillArray(uint n, uint upbound);
inline void check_cuda_errors(const char *filename, const int line_number);


const int SHARED_SIZE_LIMIT = 1024;

void runCudaMerge(uint *array, uint *keys, uint dimx, uint num_bytes);
template<uint sortDir> __global__ void mergeSortSharedKernel(uint *d_DstKey, uint *d_DstVal, uint *d_SrcKey, uint *d_SrcVal, uint arrayLength);
template<uint sortDir> static inline __device__ uint binarySearchExclusive(uint val, uint *data, uint L, uint stride);
template<uint sortDir> static inline __device__ uint binarySearchInclusive(uint val, uint *data, uint L, uint stride);
uint * createKeyArray(uint size);
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

uint * fillArray(uint n, uint upbound)
{
   uint i;
   
   uint *ret = (uint *)malloc(sizeof(uint) * n );

   /* Intializes random number generator */
   //seeds the random number generator used by the function rand.
   srand(time(NULL));

   /* generate n random numbers from 0 to unbound - 1 */
   for( i = 0 ; i < n ; i++ ) {
      uint num = (uint)(rand() % 100);
      ret[i] = num;
      //printf("%f\n", num);
   }
   return ret;
}

__global__ void mergeSortKernel(uint *in, uint *out, uint n)
{
	extern __shared__ double sdata[];
	
	// load the shared memory
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*(blockDim.x*2) + threadIdx.x;

   if(i < n){
      sdata[tid] = in[i];
   }

   if(i+blockDim.x < n){
      sdata[tid+blockDim.x] = in[i+blockDim.x] ;
   }

   for(unsigned int s = 2; s <= (blockDim.x*2); s*=2){
        if(tid < (blockDim.x*2)/s && s > 2){
            int start1 = tid*s; //First array offset
            int start2 = start1+(s/2); //Second array offset
            int size = s/2; //Size of each array will be the stride cut in half

            for(int x = size-1; x>=0; x--){
               int j;
               int last = sdata[start2-1];
         

               for(j = size-2; j>= 0 && sdata[start1+j] > sdata[start2+x]; j--){
                  sdata[start1+(j+1)] = sdata[start1+(j)];
               }

               if(j != size-2 || last > sdata[start2+x]){
                  sdata[start1+(j+1)] = sdata[start2+x];
                  sdata[start2+x] = last;
               }
            }
        }else if(tid < (blockDim.x*2)/s){ //The initial comparison of just two elements
           int start = tid*s;
           if(sdata[start] > sdata[start+1]){
              int temp = sdata[start];
              sdata[start] = sdata[start+1];
              sdata[start+1] = temp;
           }
        }
        __syncthreads();
    }

   if(i < n){
      out[i] = sdata[tid];
   }

   if(i+blockDim.x < n){
      out[i+blockDim.x] = sdata[tid+blockDim.x] ;
   }


}

template<uint sortDir> __global__ void mergeSortSharedKernel(
    uint *d_DstKey,
    uint *d_DstVal,
    uint *d_SrcKey,
    uint *d_SrcVal,
    uint arrayLength   //actual length for this block to process in the input array.
)
{
    __shared__ uint s_key[SHARED_SIZE_LIMIT];  //SHARED_SIZE_LIMIT: number of elements a thread block processes
    __shared__ uint s_val[SHARED_SIZE_LIMIT];   // Actually, the number of threads per block is SHARED_SIZE_LIMIT/2
    
    
    //Sets pointers to thread numberth element of all arrays
    d_SrcKey += blockIdx.x * SHARED_SIZE_LIMIT + threadIdx.x;
    d_SrcVal += blockIdx.x * SHARED_SIZE_LIMIT + threadIdx.x;
    d_DstKey += blockIdx.x * SHARED_SIZE_LIMIT + threadIdx.x;
    d_DstVal += blockIdx.x * SHARED_SIZE_LIMIT + threadIdx.x;
    
    
    //populates array so comparisons can be made
    s_key[threadIdx.x +                       0] = d_SrcKey[                      0];
    s_val[threadIdx.x +                        0] =  d_SrcVal[                      0];
    s_key[threadIdx.x + (SHARED_SIZE_LIMIT / 2)] = d_SrcKey[(SHARED_SIZE_LIMIT / 2)];
    s_val[threadIdx.x + (SHARED_SIZE_LIMIT / 2)]  = d_SrcVal[(SHARED_SIZE_LIMIT / 2)];
    
    for (uint stride = 1; stride < arrayLength; stride <<= 1)//'<<=' means multiplies by two or (shifts 									over two bits)
    {
        uint  lPos = threadIdx.x  &  (stride - 1);// '&' with two operands only returns that # if they 								are both that #
        
        //iterates through shared memory array and points to array (thread # - lpos always 0)
        uint *baseKey = s_key + 2 * (threadIdx.x - lPos);
        uint *baseVal = s_val + 2 *   (threadIdx.x - lPos);
        
        
        __syncthreads();
        uint keyA = baseKey[lPos +      0];
        uint valA = baseVal[lPos +      0];
        uint keyB = baseKey[lPos + stride];
        uint valB = baseVal[lPos + stride];
        uint posA = binarySearchExclusive<sortDir>(keyA, baseKey + stride, stride, stride) + lPos;
        uint posB = binarySearchInclusive<sortDir>(keyB, baseKey +      0,    stride, stride) + lPos;

      
        __syncthreads();
        baseKey[posA] = keyA;
        baseVal[posA] = valA;
        baseKey[posB] = keyB;
        baseVal[posB] = valB;
    }
    
    __syncthreads();
    d_DstKey[                      0] = s_key[threadIdx.x +                       0];
    d_DstVal[                       0]  = s_val[threadIdx.x +                       0];
    d_DstKey[(SHARED_SIZE_LIMIT / 2)] = s_key[threadIdx.x + (SHARED_SIZE_LIMIT / 2)];
    d_DstVal[(SHARED_SIZE_LIMIT / 2)] =   s_val[threadIdx.x + (SHARED_SIZE_LIMIT / 2)];
}

template<uint sortDir> static inline __device__ uint binarySearchInclusive(uint val, uint *data, uint L, uint stride)
{
    if (L == 0)
    {
        return 0;
    }

    uint pos = 0;

    for (; stride > 0; stride >>= 1)
    {
        uint newPos = umin(pos + stride, L);

        if ((sortDir && (data[newPos - 1] <= val)) || (!sortDir && (data[newPos - 1] >= val)))
        {
            pos = newPos;
        }
    }

    return pos;
}

template<uint sortDir> static inline __device__ uint binarySearchExclusive(uint val, uint *data, uint L, uint stride)
{
    if (L == 0)
    {
        return 0;
    }

    uint pos = 0;

    for (; stride > 0; stride >>= 1)
    {
        uint newPos = umin(pos + stride, L);

        if ((sortDir && (data[newPos - 1] < val)) || (!sortDir && (data[newPos - 1] > val)))
        {
            pos = newPos;
        }
    }

    return pos;
}

void runCUDA( uint *arr, uint n, uint tile_width)
{    
   uint *h_in = arr; //Filled array
   uint *h_out = (uint *)malloc(sizeof(uint) * n ); //Allocate ouput mem
   uint *d_in;//Device in pointer
   uint *d_out;

   cudaMalloc((void**)&d_in, sizeof(uint) *n);//Allocate in and out mem
   cudaMalloc((void**)&d_out, sizeof(uint) *n);//Allocate in and out mem

   cudaMemcpy(d_in, h_in, sizeof(uint)*n, cudaMemcpyHostToDevice);//Copy in array to device

   int num_block = (ceil(n/(double)tile_width));//Calculate grid size
   printf("Num of blocks is %d\n", num_block);
   dim3 block(tile_width/2,1,1);//Only 1/2 threads per block
   dim3 grid(num_block, 1,1);//Define grid

   //launch this shit and hope it works
   mergeSortKernel<<<grid, block, (tile_width)*sizeof(int)>>>(d_in, d_out, n);
   check_cuda_errors(__FILE__, __LINE__);
   cudaDeviceSynchronize();
   
   cudaMemcpy(h_out, d_out, sizeof(uint)*n, cudaMemcpyDeviceToHost);
   
   uint * keys = createKeyArray(n);
   uint num_bytes = n * sizeof(uint);
   runCudaMerge(h_out, keys, n, num_bytes);

   printArray(h_out, n);
}

void runCudaMerge(uint *array, uint *keys, uint dataSize, uint num_bytes)
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


void printArray(uint *arr, uint n){

   int i;

   for(i = 0; i < n; i ++)
      printf("%d ", arr[i]);

   printf("\n");
}

uint * createKeyArray(uint size)
{
	uint * array = (uint*)malloc(sizeof(uint) * size);
	for(int i = 0; i < size; i++)
	{
		array[i] = i;	
	}
	return array;
}

int main(int argc, char *argv[]){
    uint * arr = fillArray(128, 200);
    printArray(arr, 128);
    runCUDA( arr, 128, 32); // Array, Elements, Tile size
    
    return 0;
}


