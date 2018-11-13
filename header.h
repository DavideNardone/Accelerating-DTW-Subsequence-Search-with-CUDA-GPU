#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <ctype.h>
#include <math.h>
#include <float.h>
#include <time.h>
#include <sys/time.h>
#include <cuda.h>
#include <unistd.h>


// const int WS = 1197; //T length=nt aka time_length-1 [i.e. (153-1=152)]

struct data
{
  int tot_size;
  int train_size;
  int test_size;

};

#if defined W_SIZE
static const int WS = W_SIZE;
#endif

// const int D=3; // number of features


#ifndef HEADER_H
#define HEADER_H

#ifdef __cplusplus
extern "C" {
	#endif

	//device
	__global__ void MD_ED_D(float* , float* , int , int , float* , int , int, int );
	__global__ void MD_DTW_D(float* , float* , int , int , int , float* ,int, int, int );
	__global__ void MD_ED_I(float* , float* , int , int , float* , int , int );
	__global__ void MD_DTW_I(float* , float* , int , int , int , float* ,int, int);
	__global__ void rMD_ED_D(float* , float* , int , int , float* , int, int );
	__global__ void rMD_DTW_D(float* , float* , int , int , int , float*, int, int);
	__device__ float stdDev(float *, int ,float *);


	//host
	__host__ void createTrainingTestingSet(float *, int *, int , int , int , float *, int *, int, float *, int *, int, int *, int , int );
	__host__ int checkFlagOpts(char **, int , int , int );
	__host__ void checkCUDAError (const char* ); 
	__host__ void readFile(char** ,int *, int, int , float *, data , int , int *, int , int );
	__host__ void readFileSubSeq(char **, int *, int , float *, int , float *, int , int , int );
	__host__ float standard_deviation(float *, int, float *);
	__host__ void z_normalize2D(float *, int ,int); 
	__host__ float short_dtw_c(float *, float *,int , int );
	__host__ float short_md_dtw_c(float *, float *,int , int ,int, int );
	__host__ float short_ed_c(float *, float *, int );
	__host__ float short_md_ed_c(float *, float *, int , int ,int );
	__host__ void print_help(void);
	__host__ void print_version(void);
	__host__ void infoDev();
	__host__ cudaDeviceProp getDevProp(int );
	__host__ void initializeArray(float *, int, float );
	__host__ void initializeMatrix(float *, int , int );
	__host__ void equalArray(float *, float *, int );
	__host__ void compareArray(float *, float *, int );
	__host__ void printArray(float* , int );
	__host__ void printArrayI(int* , int );
	__host__ void printMatrix(float *, int , int );
	__host__ float min_arr(float *,int ,int *);
	__host__ int cmpfunc (const void * , const void * );
	__host__ void generateArray(int ,int *,int );
	__host__ void findInd(int* , int , int* , int );
	__host__ int unique_val(int* ,int );
	__host__ int* accumarray(int* ,int ,int* );
	__host__ float timedifference_msec(struct timeval , struct timeval );
	__host__ void shuffle(int *, size_t , size_t );
	__host__ void idAssign(int* ,int ,int* ,int ,int* ,int* ,int* );
	__host__ int* crossvalind_Kfold(int* ,int ,int, int );
	__host__ int countVal(int *,int ,int );
	__host__ void fakeK_fold(int *array, int n,int m);

	#ifdef __cplusplus 
}
#endif

#endif