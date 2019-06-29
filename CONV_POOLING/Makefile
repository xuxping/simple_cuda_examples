
NVCC = /usr/local/cuda/bin/nvcc
NVCC_FLAGS = -g -G -Xcompiler -Wall

all: conv.exe pooling.exe

conv.exe: conv.cu
	$(NVCC) $^ -o $@

pooling.exe: pooling.cu
	$(NVCC) $^ -o $@
	
clean:
	rm -f *.o *.exe

