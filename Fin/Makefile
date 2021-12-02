# Build tools
NVCC = nvcc
CXX = g++

# here are all the objects
GPUOBJS = main.o MergingKernels.o KernelSetup.o


# make and compile
project:$(OBJS) $(GPUOBJS)
	$(NVCC) -o project $(OBJS) $(GPUOBJS) 

main.o: main.cu
	$(NVCC) -arch=sm_30 -c main.cu 

MergingKernels.o: MergingKernels.cu
	$(NVCC) -arch=sm_30 -c MergingKernels.cu
	
KernelSetup.o: KernelSetup.cu
	$(NVCC) -arch=sm_30 -c KernelSetup.cu
	
main.o: MergingKernels.h KernelSetup.h

MergingKernels.o: MergingKernels.h KernelSetup.h

KernelSetup.o: MergingKernels.h KernelSetup.h

clean:
	rm -f *.o
	rm -f project
