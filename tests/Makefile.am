cc = nvcc 
cflags = -arch=sm_30 -Xcompiler "-O3 -Wall"
lflags = -Iinclude/
unitt_flags = -lcheck -lm -lpthread -lrt

all: comp1 comp2 comp3

comp1:
	$(cc) $(cflags) -c src/module.cu -o src/module.o $(lflags)

comp2: 
	$(cc) $(cflags) -c tests/check_MD_DTW.cu -o tests/check_MD_DTW.o

comp3:
	$(cc) $(cflags) src/module.o tests/check_MD_DTW.o $(unitt_flags) -o unit_test

# run:
# 	./unit_test