# Build tools
NVCC = /usr/local/cuda/bin/nvcc
CXX = g++

# here are all the objects
GPUOBJS = MergeSort.o 

# make and compile
MergeSort:$(GPUOBJS)
	$(NVCC) -o MergeSort $(GPUOBJS) 

MergeSort.o: MergeSort.cu
	$(NVCC) -arch=sm_52 -c MergeSort.cu 

clean:
	rm -f *.o
	rm -f MergeSort
