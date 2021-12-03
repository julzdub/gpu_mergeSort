# Build tools
NVCC = /usr/local/cuda/bin/nvcc
CXX = g++

# here are all the objects
GPUOBJS = main.o  
OBJS = cpuMergeSort.o timing.o

# make and compile
main:$(OBJS) $(GPUOBJS)
	$(NVCC) -o main $(OBJS) $(GPUOBJS) 

main.o: main.cu
	$(NVCC) -arch=sm_52 -c main.cu 

cpuMergeSort.o: cpuMergeSort.c
	$(CXX) -c cpuMergeSort.c

cpuMergeSort.o: cpuMergeSort.h

timing.o: timing.c 
	$(CXX) -c timing.c 

clean:
	rm -f *.o
	rm -f main
