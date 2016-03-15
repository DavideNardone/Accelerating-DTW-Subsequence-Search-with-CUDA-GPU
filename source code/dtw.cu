#include "dtw_h.h"

using namespace std;



__global__ void DTW_GM (float* data_in, float* query, int nt, float* data_out) {
	int k,l,g;
	
	long long int i,j;
	int idx = blockIdx.x * blockDim.x + threadIdx.x; //i.e [0...6]*[512]+[0...511]

	float min_nb;
	float array[WS][2];
    float instance[WS];


    float sum = 0, sum_sqr = 0, mean = 0, mean_sqr = 0, variance = 0, std_dev = 0;

    int t = idx;
    if(idx+WS>nt)
    	return;


    for(i=t; i<t+WS; i++)
    {
      sum += data_in[i];
		sum_sqr += data_in[i] * data_in[i]; //pow(data[i],2);
	}


    //measures for normalization step
	mean = sum/WS;
	mean_sqr = mean*mean;

	variance = (sum_sqr/WS) - mean_sqr;
	std_dev = sqrt(variance);


	//normalizing subsequence C[t+1,t+2,...,t+WS]
	for(i=0; i<WS; i++)
		instance[i] = (data_in[t+i]-mean)/std_dev; 

    k = 0;
    l = 1;

    //initialization step
    for(i=0;i<WS;i++)
    {
        if (i==0)
            array[i][k]=pow((instance[0]-query[i]),2); //squared difference (ins[0]-query[0])^2 
        else
            array[i][k]=pow((instance[0]-query[i]),2)+array[i-1][k];
    }

    k = 1;
    l = 0;
    
    //computing DTW
    for(j=1; j<WS; j++)
    {
        i = 0;
        array[i][k]=pow((instance[j] - query[i]),2)+array[i][l];

        for (i=1; i<WS; i++)
        {
            double a = array[i-1][l];
            double b = array[i][l];
            double c = array[i-1][k];
            
            min_nb = fminf(a,b);
            min_nb = fminf(c,min_nb);

            array[i][k]=pow((instance[j] - query[i]),2)+min_nb;
        }
        g = k;
        k = l;
        l = g;
    }
    data_out[idx] = array[WS-1][g];
}



__global__ void DTW_SM (float* data_in, float* query, int nt, float* data_out)
{
    int k,l,g;
    
    long long int i,j;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    float min_nb;
    float instance[WS];

    __shared__ float array[WS][2];
    extern __shared__ float query_sm[];
    
    for( i = 0 ; i< WS ; i++ )
       query_sm[i] = query[i];

   float sum = 0, sum_sqr = 0, mean = 0, mean_sqr = 0, variance = 0, std_dev = 0;

   int t = idx;
   if(idx+WS>nt)
    return;


    //measures for normalization step
    for(i=t; i<t+WS; i++)
    {
        sum += data_in[i];
        sum_sqr += data_in[i] * data_in[i]; //pow(data[i],2);
    }


    mean = sum/WS;
    mean_sqr = mean*mean;

    variance = (sum_sqr/WS) - mean_sqr;
    std_dev = sqrt(variance);

    //normalizing subsequence C[t+1,t+2,...,t+WS]
    for(i=0; i<WS; i++)
        instance[i] = (data_in[t+i]-mean)/std_dev; 

    k = 0;
    l = 1;

    //initialization step
    for(i=0;i<WS;i++)
    {
        if (i==0)
            array[i][k]=pow((instance[0]-query_sm[i]),2); //squared difference (ins[0]-query[0])^2 
        else
            array[i][k]=pow((instance[0]-query_sm[i]),2)+array[i-1][k];
    }

    k = 1;
    l = 0;
    
    //computing DTW
    for(j=1; j<WS; j++)
    {
        i = 0;
        array[i][k]=pow((instance[j] - query_sm[i]),2)+array[i][l];

        for (i=1; i<WS; i++)
        {
            double a = array[i-1][l];
            double b = array[i][l];
            double c = array[i-1][k];
            
            min_nb = fminf(a,b);
            min_nb = fminf(c,min_nb);

            array[i][k]=pow((instance[j] - query_sm[i]),2)+min_nb;
        }
        g = k;
        k = l;
        l = g;;
    }

    float min=array[WS-1][g];
    data_out[idx] = min;    
}



__global__ void owpThresholdGPU(float *owp,int size,float th,float *bin) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x; //i.e [0...6]*[512]+[0...511]

    if (idx>size)
        return;

    if(owp[idx]<th)
        bin[idx]=owp[idx];
    else
        bin[idx]=-1.0;
}



__global__ void rED (float* data_in, float* query, float* data_out, const int window_size)
{
    
    long long int i;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    float instance[1024];

    float sum = 0, sum_sqr = 0, mean = 0, mean_sqr = 0, variance = 0, std_dev = 0;
    
    int s = 2*window_size*(idx/window_size);
    int t = s + idx%window_size;

    for(i=t; i<t+window_size; i++)
    {
        sum += data_in[i];
        sum_sqr += data_in[i] * data_in[i];
    }


    mean = sum / window_size;
    mean_sqr = mean*mean;

    variance = (sum_sqr/window_size) - mean_sqr;
    std_dev = sqrt(variance);


    i = 0;
    for(; i<window_size; i++)
        instance[i] = (data_in[t+i]-mean) / std_dev; 

    
    float sumErr  = 0;
    //euclidean distance
    for(i=0; i < window_size; i++)
        sumErr += (instance[i] - query[i])*(instance[i] - query[i]); 
    
    data_out[idx] = sqrt(sumErr);

}



__global__ void rDTW (float* data_in, float* q, float* data_out,int window_size)
{

    int k, l , g;
    
    long long int i,j;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    float min_nb;
    float array[1024][2];
    float instance[1024];

    __shared__ float query[1024];
    
    for( i = 0 ; i< window_size ; i++ )
        query[i] = q[i];

    float sum = 0, sum_sqr = 0, mean = 0, mean_sqr = 0, variance = 0, std_dev = 0;
    
    //offset training set
    int s = 2*window_size*(idx/window_size);
    int t = s + idx%window_size;

    // int t = idx;
    // if(idx+window_size>N-1)
    //    return;

    //normalizing i-th light curve
    for(i=t; i<t+window_size; i++) {
        sum += data_in[i];
        sum_sqr += data_in[i] * data_in[i];
    }

    mean = sum / window_size;
    mean_sqr = mean*mean;

    variance = (sum_sqr/window_size) - mean_sqr;
    std_dev = sqrt(variance);

    k = 0;
    l = 1;

    //////////////////////////////////////////MYDTW//////////////////////////////////////////
    for(i=0; i<window_size; i++)
        instance[i] = (data_in[t+i]-mean) / std_dev;

    //initialization step
    for(i=0;i<window_size;i++)
    {
        if (i==0)
            array[i][k]=pow((instance[0]-query[i]),2); //squared difference (ins[0]-query[0])^2 
        else
            array[i][k]=pow((instance[0]-query[i]),2)+array[i-1][k];
    }

    k = 1;
    l = 0;
    
    //computing DTW
    for(j=1; j<window_size; j++)
    {
        i = 0;
        array[i][k]=pow((instance[j] - query[i]),2)+array[i][l];

        for (i=1; i<window_size; i++)
        {
            double a = array[i-1][l];
            double b = array[i][l];
            double c = array[i-1][k];

            min_nb = fminf(a,b);
            min_nb = fminf(c,min_nb);

            array[i][k]=pow((instance[j] - query[i]),2)+min_nb;
        }
        g = k;
        k = l;
        l = g;
    }

    data_out[idx] = array[window_size-1][g];
}



__host__ void z_normalize(float *query,int nq,float *query_norm) {


    float sum = 0, sum_sqr = 0, mean = 0, mean_sqr = 0, variance = 0, std_dev = 0;

    for(int i=0; i<nq; i++) {

        sum += query[i];
        sum_sqr += query[i] * query[i]; //pow(data[i],2);
    }


    mean = sum / nq;
    mean_sqr = mean*mean;

    variance = (sum_sqr/nq) - mean_sqr;
    std_dev = sqrt(variance);
    
    for(int i=0; i<nq; i++)
        query_norm[i] = (query[i]-mean)/std_dev; 

}



__host__ float short_dtw_c(float *instance, float *query,int ns, int nt){


    int k, l , g;
    long long int i,j;
    float **array;
    float min_nb;

    // create array
    array=(float **)malloc((nt)*sizeof(float *));
    for(i=0;i<nt;i++)
    {
        array[i]=(float *)malloc((2)*sizeof(float));
    }


    k = 0;
    l = 1;

    //initialization step
    for(i=0;i<nt;i++)
    {
        if (i==0)
            array[i][k]=pow((instance[0]-query[i]),2); //squared difference (ins[0]-query[0])^2 
        else
            array[i][k]=pow((instance[0]-query[i]),2)+array[i-1][k];
    }

    k = 1;
    l = 0;

    //computing DTW    
    for(j=1; j<ns; j++)
    {
        i = 0;
        array[i][k]=pow((instance[j] - query[i]),2)+array[i][l];

        for (i=1; i<nt; i++)
        {
            float a = array[i-1][l];
            float b = array[i][l];
            float c = array[i-1][k];
            
            min_nb = fminf(a,b);
            min_nb = fminf(c,min_nb);

            array[i][k]=pow((instance[j] - query[i]),2)+min_nb;
        }
        g = k;
        k = l;
        l = g;
    }

    float min=array[nt-1][g];

    for(i=0;i<ns;i++)
        free(array[i]);
    free(array);

    return min;
}



__host__ void infoDev(int p) {

    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    int device;
    cudaDeviceProp deviceProp;
    //retrieving all devices
    if(p==1){
        for (device = 0; device < deviceCount; ++device)
        {
            //getting information about i-th device
            cudaGetDeviceProperties(&deviceProp, device);
            //printing information about i-th device
            printf("\n\n>>>>>>>>>>>>>>>>>>\nSelected device:%d\n<<<<<<<<<<<<<<<<<<\n\n", device);
            printf("\ndevice %d : %s\n",device,deviceProp.name);
            printf("major/minor : %d.%d compute capability\n",deviceProp.major,deviceProp.minor);
            printf("Total global mem : %u bytes\n", deviceProp.totalGlobalMem);
            printf("Shared block mem : %d bytes\n",deviceProp.sharedMemPerBlock);
            printf("Max memory pitch : %d bytes\n",deviceProp.memPitch);
            printf("RegsPerBlock : %d \n", deviceProp.regsPerBlock);
            printf("WarpSize : %d \n", deviceProp.warpSize);
            printf("MaxThreadsPerBlock : %d \n", deviceProp.maxThreadsPerBlock);
            printf("TotalConstMem : %d bytes\n", deviceProp.totalConstMem);
            printf("ClockRate : %d (kHz)\n", deviceProp.clockRate);
            printf("deviceOverlap : %d \n", deviceProp.deviceOverlap);
            printf("deviceOverlap : %d \n", deviceProp.deviceOverlap);
            printf("MultiProcessorCount: %d \n", deviceProp.multiProcessorCount);
            printf("\n");
        }
    }
    device=0;
    cudaSetDevice(device);
}



__host__ float* owpThreshold(float *owp,int size,float th) {

    float *bin = (float*) malloc (size*sizeof(float));

    //probing
    for (int i = 0; i < size; i++)
    {
        if (owp[i]<th) {
            bin[i]=owp[i];
        }
        else
            bin[i]=-1.0;;
    }

    return bin;
}



__host__ void printoutput2File(FILE *out,float *x_data,float *y_data,int size){


    for(int i = 0; i < size; i++){
        if(y_data[i]!=-1.0)
            fprintf(out, "%f,%f\n", x_data[i],y_data[i]);
    }
}



__host__ void initializeArray(float *array, int n) {
    int i;
    for (i = 0; i < n; i++)
        array[i] = i+1;
}



__host__ void printArray(float* array, int n) {

    int i;
    for (i = 0; i < n; i++)
        printf("%f ", array[i]);
    printf("\n");
}



__host__ void equalArray(float *a, float *b, int n) {

	int i = 0;

	while (a[i] == b[i])
        i++;
    if (i < n) {
        printf("I risultati dell'host e del device sono diversi\n");
        printf("CPU[%d]: %f, GPU[%d]: %f \n",i,a[i],i,b[i]);
    }
    else
        printf("I risultati dell'host e del device coincidono\n");
}



__host__ void compareArray(float *a, float *b, int n) {

    int i = 0;

    for (i = 0; i < n; ++i) {
        if(a[i]!=b[i])
            printf("CPU[%d]: %f, GPU[%d]: %f \n",i,a[i],i,b[i]);
    }
}



__host__ void checkCUDAError (const char* msg) { 

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess ) {
        fprintf(stderr, "Cuda error: %s %s\n",msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}



float min_arr(float *arr,int n,int *ind) {

    float min=FLT_MAX;
    for (int i = 0; i < n; ++i)
    {
        if (arr[i]<min){
            min=arr[i];
            *ind=i;
        }
    }

    return min;
}


