#include <stdio.h>
#include <stdlib.h>
#include <math.h>

//TEMP HEADER
void printArray(int *arr, int n);
void runCUDA( int *arr, int n, int tile_width);
__global__ void mergeSortKernel(double *in, int n);
int * fillArray(int n, int upbound);
inline void check_cuda_errors(const char *filename, const int line_number);
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

__global__ void mergeSortKernel(int *in, int *out, int n)
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

void runCUDA( int *arr, int n, int tile_width)
{    
   int *h_in = arr; //Filled array
   int *h_out = (int *)malloc(sizeof(int) * n ); //Allocate ouput mem
   int *d_in;//Device in pointer
   int *d_out;

   cudaMalloc((void**)&d_in, sizeof(int) *n);//Allocate in and out mem
   cudaMalloc((void**)&d_out, sizeof(int) *n);//Allocate in and out mem

   cudaMemcpy(d_in, h_in, sizeof(int)*n, cudaMemcpyHostToDevice);//Copy in array to device

   int num_block = (ceil(n/(double)tile_width));//Calculate grid size
   printf("Num of blocks is %d\n", num_block);
   dim3 block(tile_width/2,1,1);//Only 1/2 threads per block
   dim3 grid(num_block, 1,1);//Define grid

   //launch this shit and hope it works
   mergeSortKernel<<<grid, block, (tile_width)*sizeof(int)>>>(d_in, d_out, n);
   check_cuda_errors(__FILE__, __LINE__);

   cudaMemcpy(h_out, d_out, sizeof(int)*n, cudaMemcpyDeviceToHost);

   printArray(h_out, n);
}


void printArray(int *arr, int n){

   int i;

   for(i = 0; i < n; i ++)
      printf("%d ", arr[i]);

   printf("\n");
}

int main(int argc, char *argv[]){
    int* arr = fillArray(128, 200);
    printArray(arr, 128);
    runCUDA( arr, 128, 32); // Array, Elements, Tile size
}