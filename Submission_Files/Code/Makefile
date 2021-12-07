# Build Tools
NVCC = nvcc
CXX = g++

# Here are all the objects
GPUOBJS = main.o Kernels.o
OBJS = cpuMergeSort.o timing.o

project:$(OBJS) $(GPUOBJS) 
	$(NVCC) -arch=sm_52 -o project $(OBJS) $(GPUOBJS)
	
main.o: main.cu
	$(NVCC) -arch=sm_52 -c main.cu
	
Kernels.o: Kernels.cu
	$(NVCC) -arch=sm_52 -c Kernels.cu
	
cpuMergeSort.o: cpuMergeSort.c
	$(CXX) -c cpuMergeSort.c
	
timing.o: timing.c
	$(CXX) -c timing.c
	

	
clean:
	rm -f *.o
	rm -f project
	rm -f *.txt
