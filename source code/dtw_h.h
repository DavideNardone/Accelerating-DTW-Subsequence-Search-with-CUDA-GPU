#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <time.h>
#include <cuda.h>

const int WS = 250; //query length/window size

#ifndef DTW_H_H
#define DTW_H_H

#ifdef __cplusplus
extern "C" {
	#endif

	//device
	__global__ void DTW_GM (float *, float *, int , float *);
	__global__ void DTW_SM (float *, float *, int , float *);
	__global__ void owpThresholdGPU(float *,int ,float ,float *);
	__global__ void rDTW (float *, float *, float *,int );
	__global__ void rED (float *, float *, float *, int );


	//host
	__host__ void z_normalize(float *,int ,float *);
	__host__ float short_dtw_c(float *, float *,int , int );
	__host__ void infoDev(int );
	__host__ void initializeArray(float *, int );
	__host__ void equalArray(float *, float *, int );
	__host__ void compareArray(float *, float *, int );
	__host__ void printArray(float *, float );
	__host__ void checkCUDAError (const char *);
	__host__ float* owpThreshold(float *,int , float );
	__host__ void printoutput2File(FILE *,float *,float *,int );
	__host__ float min_arr(float *,int ,int *);

	#ifdef __cplusplus 
}
#endif

#endif