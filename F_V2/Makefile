# Build tools
NVCC = /usr/local/cuda/bin/nvcc
CXX = g++

# here are all the objects
GPUOBJS = MergeSort.o timing.o

# make and compile
MergeSort:$(GPUOBJS)
	$(NVCC) -o MergeSort $(GPUOBJS) 

MergeSort.o: MergeSort.cu
	$(NVCC) -arch=sm_52 -lineinfo -c MergeSort.cu 

timing.o: timing.c 
	$(CXX) -c timing.c 

clean:
	rm -f *.o
	rm -f MergeSort
