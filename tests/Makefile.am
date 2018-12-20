cc = nvcc 
cflags = -arch=sm_30 -Xcompiler "-O3 -Wall"
lflags = -Iinclude/
unitt_flags = -lcheck -lm -lpthread -lrt

all: comp1 comp2 comp3

comp1: 
	$(cc) $(cflags) -c check_MD_DTW.cu -o check_MD_DTW.o

comp2:
	$(cc) $(cflags) ../src/module.o check_MD_DTW.o $(unitt_flags) -o unit_test

comp3:
	$(cc) $(cflags) ../src/module.cu benchmark.cu $(unitt_flags) -w -o benchmark