# Build tools
NVCC = nvcc
CXX = g++

# here are all the objects
GPUOBJS = KernelSetup.o main.o  


# make and compile
project:$(OBJS) $(GPUOBJS)
	$(NVCC) -o project $(GPUOBJS) 

KernelSetup.o: KernelSetup.cu
	$(NVCC) -arch=sm_30 -c KernelSetup.cu
	
main.o: main.cu
	$(NVCC) -arch=sm_30 -c main.cu
	




clean:
	rm -f *.o
	rm -f project
