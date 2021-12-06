#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <float.h>
#include "timing.h"




//TEMP HEADER
void printArray(int *arr, int n);
void runCUDA( int *arr, int n, int tile_width);
__global__ void mergeSortKernel(double *in, int n);
int * fillArray(int n, int upbound);
inline void check_cuda_errors(const char *filename, const int line_number);
int validateOutput(int size, int tile_width, int *arr);
void printArray2(int *arr,int start);
__global__  void mergeKernel(int* in, int* out, int dataSize, int tile_size);
void runMerge(int * arr, int n, int tile_width, float time);
void mergeSortCPU(int* arr, int n);
void mergeCPU(int* arr, int l, int m, int r);
void runCPU(int* inputArray, int size);
//int min(int x, int y);
void copyArray(int* original, int* copy, int size);
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
   cudaEvent_t launch_begin, launch_end;
   cudaEventCreate(&launch_begin);
   cudaEventCreate(&launch_end);

   cudaEventRecord(launch_begin, 0);
   mergeSortKernel<<<grid, block, (tile_width)*sizeof(int)>>>(d_in, d_out, n);
   cudaEventRecord(launch_end, 0);
   cudaEventSynchronize(launch_begin);
   cudaEventSynchronize(launch_end);
   check_cuda_errors(__FILE__, __LINE__);

   //To record time at the end
   float time = 0;
   cudaEventElapsedTime(&time, launch_begin, launch_end);
   printf("Done! Time cost is. %f\n", time/1000);

   //Destroy cudaEvents
   cudaEventDestroy(launch_begin);
   cudaEventDestroy(launch_end);


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

   runMerge(h_out, n, tile_width, time);

   free(h_in);
   free(h_out);
   cudaFree(d_in);
   cudaFree(d_out);
}

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

//////////////////////////////////////////////////////

//int min(int x, int y) { return (x<y)? x :y; }
 
 
/* Iterative mergesort function to sort arr[0...n-1] */
void mergeSortCPU(int* arr, int n)
{
   int curr_size;  // For current size of subarrays to be merged
                   // curr_size varies from 1 to n/2
   int left_start; // For picking starting index of left subarray
                   // to be merged
 
   // Merge subarrays in bottom up manner.  First merge subarrays of
   // size 1 to create sorted subarrays of size 2, then merge subarrays
   // of size 2 to create sorted subarrays of size 4, and so on.
   for (curr_size=1; curr_size<=n-1; curr_size = 2*curr_size)
   {
       // Pick starting point of different subarrays of current size
       for (left_start=0; left_start<n-1; left_start += 2*curr_size)
       {
           // Find ending point of left subarray. mid+1 is starting
           // point of right
           int mid = min(left_start + curr_size - 1, n-1);
 
           int right_end = min(left_start + 2*curr_size - 1, n-1);
 
           // Merge Subarrays arr[left_start...mid] & arr[mid+1...right_end]
           mergeCPU(arr, left_start, mid, right_end);
       }
   }
}
 
/* Function to merge the two haves arr[l..m] and arr[m+1..r] of array arr[] */
void mergeCPU(int* arr, int l, int m, int r)
{
    int i, j, k;
    int n1 = m - l + 1;
    int n2 =  r - m;
 
    /* create temp arrays */
    int L[n1], R[n2];
 
    /* Copy data to temp arrays L[] and R[] */
    for (i = 0; i < n1; i++)
        L[i] = arr[l + i];
    for (j = 0; j < n2; j++)
        R[j] = arr[m + 1+ j];
 
    /* Merge the temp arrays back into arr[l..r]*/
    i = 0;
    j = 0;
    k = l;
    while (i < n1 && j < n2)
    {
        if (L[i] <= R[j])
        {
            arr[k] = L[i];
            i++;
        }
        else
        {
            arr[k] = R[j];
            j++;
        }
        k++;
    }
 
    /* Copy the remaining elements of L[], if there are any */
    while (i < n1)
    {
        arr[k] = L[i];
        i++;
        k++;
    }
 
    /* Copy the remaining elements of R[], if there are any */
    while (j < n2)
    {
        arr[k] = R[j];
        j++;
        k++;
    }
}

void runCPU(int* inputArray, int size) {
	
    	clock_t now, then;
    	
    	printf("Timing CPU implementation…\n");
    	then = clock();
    	mergeSortCPU(inputArray,size);
    	now =  clock();
    	
    	// measure the time spent on CPU
       float time = 0;
       time = timeCost(then, now);
       
       

       printf(" done. CPU time cost in second: %f\n", time);
}

void copyArray(int* original, int* copy, int size){
   int i;
   for(i = 0; i < size; i++){

      copy[i] = original[i];
   }
}



int main(int argc, char *argv[]){
   int size =  2097152;
   int tile_width = 1024;
   int* arr = fillArray(size, 10000);

   int* arrCPU = (int*)malloc(sizeof(int) * size );
   copyArray(arr, arrCPU, size);
   runCPU(arrCPU, size);

   //int arr_size = sizeof(arr) / sizeof(arr[0]);

    //runCUDA( arr, size, tile_width); // Array, Elements, Tile size
    //mergeSort(arr, 0, arr_size - 1);
    //printArray(arr, size);


    //printing to output files
    FILE * output_CPU = fopen("output_CPU.txt", w);
    FILE * output_GPU = fopen("output_GPU.txt", w);

    printOutput(output_CPU, arrCPU);
    printOutput(output_GPU, arr);

}

void printOutput(FILE * f_out, int * arr) {
   int i;
   time_t t; 
   time(&t);

   fprintf(f_out, "%s \n", ctime(&t));
   for(i = 0; i < n; i ++)
      fprintf(f_out, "%d ", arr[i]);

   fprintf("\n");
}
