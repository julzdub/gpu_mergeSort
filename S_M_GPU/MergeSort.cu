#include <stdio.h>
#include <stdlib.h>
#include <math.h>

//TEMP HEADER
void printArray(int *arr, int n);
void runCUDA( int *arr, int n, int tile_width);
__global__ void mergeSortKernel(double *in, int n);
int * fillArray(int n, int upbound);
inline void check_cuda_errors(const char *filename, const int line_number);
int validateOutput(int size, int tile_width, int *arr);
void printArray2(int *arr,int start);
__global__  void mergeKernel(int* in, int* out, int dataSize, int tile_size);
void runMerge(int * arr, int n, int tile_width);
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
	extern __shared__ int sdata[];
	
	// load the shared memory
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*(blockDim.x*2) + threadIdx.x;
   

   if(i < n){
      sdata[tid] = in[i];
   }

   if(i+blockDim.x < n){
      sdata[tid+blockDim.x] = in[i+blockDim.x] ;
   }
   __syncthreads();


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

void runCUDA( int *arr, int n, int tile_width)
{    
   int *h_in = arr; //Filled array
   int *h_out = (int *)malloc(sizeof(int) * n ); //Allocate output mem
   int *d_in;//Device in pointer
   int *d_out;

   cudaMalloc((void**)&d_in, sizeof(int) *n);//Allocate in and out mem
   cudaMalloc((void**)&d_out, sizeof(int) *n);//Allocate in and out mem

   cudaMemcpy(d_in, h_in, sizeof(int)*n, cudaMemcpyHostToDevice);//Copy in array to device

   int num_block = (ceil(n/(double)tile_width));//Calculate grid size
   printf("\n\n\nArray size is %d\n", n);
   printf("Num of blocks is %d\n", num_block);
   printf("Tile size is %d\n", tile_width);
   printf("Active threads on run 1 are %f\n\n\n", ((double)tile_width)/2);
   dim3 block(tile_width/2,1,1);//Only 1/2 threads per block
   dim3 grid(num_block, 1,1);//Define grid

   //launch this shit and hope it works
   mergeSortKernel<<<grid, block, (tile_width)*sizeof(int)>>>(d_in, d_out, n);
   cudaDeviceSynchronize();
   check_cuda_errors(__FILE__, __LINE__);

   cudaMemcpy(h_out, d_out, sizeof(int)*n, cudaMemcpyDeviceToHost);

   //printArray(h_out, n);
   //puts("\n\n\n\n\n");
   
   int isVal = validateOutput(n, tile_width, h_out);

    if(isVal == -1){
       puts("\n\nOutput is valid!");
    }else{
       printf("\n\nOutput is NOT valid! At: %d\n\n", isVal);
       printArray2(h_out, isVal);
    }

   runMerge(h_out, n, tile_width);

   free(h_in);
   free(h_out);
   cudaFree(d_in);
   cudaFree(d_out);
}

void runMerge(int * arr, int n, int tile_width){
   int *h_in = arr; //Filled array
   int *h_out = (int *)malloc(sizeof(int) * n ); //Allocate output mem
   int *d_in;//Device in pointer
   int *d_out;

   cudaMalloc((void**)&d_in, sizeof(int) *n);//Allocate in and out mem
   cudaMalloc((void**)&d_out, sizeof(int) *n);//Allocate in and out mem

   cudaMemcpy(d_in, h_in, sizeof(int)*n, cudaMemcpyHostToDevice);//Copy in array to device
   
   int t_width = tile_width;
   int num_block;

  // mergeKernel<<<grid,block>>>(d_in, d_out, n, 1024);
   //t_width *= 2;
   
   //mergeKernel<<<1,block>>>(d_out, d_in, n, 2048);
   int* temp;

   while(t_width <= n/2){
      puts("\n\n RAN \n\n");
      num_block = n /(t_width * 2);
      mergeKernel<<<num_block,1>>>(d_in, d_out, n, t_width);
      t_width *= 2;
      
      temp = d_in;
      d_in = d_out;
      d_out = temp;
   }

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


int main(int argc, char *argv[]){
   int size =  33554432;
   int tile_width = 1024;
   int* arr = fillArray(size, 10000);
   //int arr_size = sizeof(arr) / sizeof(arr[0]);

    runCUDA( arr, size, tile_width); // Array, Elements, Tile size
    //mergeSort(arr, 0, arr_size - 1);
    //printArray(arr, size);
}


