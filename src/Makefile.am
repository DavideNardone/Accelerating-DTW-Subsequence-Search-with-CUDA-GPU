cc = nvcc 
cflags = -arch=sm_30 -Xcompiler "-O3 -Wall"
lflags = -Iinclude/
unitt_flags = -lcheck -lm -lpthread -lrt

all: comp1

comp1:
	$(cc) $(cflags) MD_DTW.cu module.cu -o mdtwObj $(lflags)
