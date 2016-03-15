#include "dtw_h.h"

using namespace std;

int main( int argc, char** argv)
{
	int i,nss; 
	cudaEvent_t start,stop;
	float timeCPU,timeGM_GPU,timeSM_GPU;

    long int window_size;
    float t_size,q_size,n_attr_t,n_attr_q;

    int blockSize=atoi(argv[1]);

    if(argc<4){
        printf("Too few arguments!!!\n");
        exit(-1);
    }

    FILE *tFile,*qFile;
    tFile=fopen(argv[2],"r");
    if(tFile==NULL) printf("The file %s cannot be opened!\n", argv[2]);
    qFile=fopen(argv[3],"r");
    if(qFile==NULL) printf("The file %s cannot be opened!\n", argv[3]);


	//reading T and Q lengths series, and number of their attributes
    fscanf(tFile,"%f",&t_size);fscanf(tFile,"%f",&n_attr_t);
    fscanf(qFile,"%f",&q_size);fscanf(qFile,"%f",&n_attr_q);

    //nss=number of subsequences
    nss=t_size-q_size+1;

	window_size=q_size; //query size

	//T and Q lengths series bytes
	unsigned long long int t_bytes=t_size*sizeof(float);
	unsigned long long int q_bytes=q_size*sizeof(float);
	
	////////////////////////////////CPU MEMORY ALLOCATION////////////////////////////////
    float* x_t_serie = (float*) malloc (t_bytes);
	float* t_serie = (float*) malloc (t_bytes);
    float* x_q_serie = (float*) malloc (q_bytes);    
	float* q_serie = (float*) malloc (q_bytes);
	float* q_serie_norm = (float*) malloc (q_bytes);
	float* subseq_norm = (float*) malloc (q_bytes);
    float* owp = (float*) malloc (nss*sizeof(float));
    memset(owp,0,nss*sizeof(float));

    //Setting CUDA variables and structure
	float grid_size=ceil((double)nss/blockSize); 
	dim3 grid(grid_size,1); //number of blocks
	dim3 threads(blockSize,1); //number of threads for blocks


    infoDev(1);
    printf("\n****************Subsequence Search Parameters****************\n\n");
	printf("Path Time Series T: %s, length: %f, n_attrs: %f byte_size: %f\n",argv[2],t_size,n_attr_t,sizeof(float)*t_size);
	printf("Path Time Series Q: %s, length: %f, n_attrs: %f, byte_size: %f\n",argv[3],q_size,n_attr_q,sizeof(float)*q_size);
    printf("Number of Subsequences to search: %d\n", nss);
    printf("Windows size: %d\n", window_size);
    printf("Grid_size_x: %d, number_of_threads_x: %d \n", grid.x,threads.x);
    printf("Grid_size_y: %d, number_of_threads_y: %d \n\n", grid.y,threads.y);


    //T series file reading
    printf("Reading Time Series T...\t");
    for(i = 0; i < t_size; i++) {
        fscanf(tFile,"%f",&x_t_serie[i]);
        fscanf(tFile,"%f",&t_serie[i]);
    }
    fclose(tFile);
    printf("done!\n");

    printf("Reading Time Series Q...\t");
    //Q series file reading
    for(i = 0; i < q_size; i++) {
        fscanf(qFile,"%f",&x_q_serie[i]);
        fscanf(qFile,"%f",&q_serie[i]);
    }
    fclose(qFile);
    printf("done!\n");

    printf("Data has been read!\n\n");

    //Query normalization
    z_normalize(q_serie,q_size,q_serie_norm);


	////////////////////////////////DTW CPU ALGORITHM////////////////////////////////
    printf("DTW CPU version processing...\n");
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start,0);

    for (int i = 0; i <nss;i++) {

    	float min=0.0;
    	z_normalize(&t_serie[i],window_size,subseq_norm);
    	min=short_dtw_c(subseq_norm,q_serie_norm,window_size,window_size);
    	owp[i]=min;
    }

    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&timeCPU,start,stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    //computing minimum value
    int* ind_min_val = (int*) malloc (sizeof(int));
    float min_val = min_arr(owp,nss,ind_min_val);
    printf("ind_min_val_CPU_version: %d, min_val_CPU_version: %f\n\n",*ind_min_val,min_val);

    //owp's indices
    float* owp_ind = (float*) malloc (nss*sizeof(float));
    initializeArray(owp_ind,nss);


	////////////////////////////////DTW GPU_GM ALGORITHM////////////////////////////////
    float* d_t_serie = 0,*d_owp=0,* d_query_norm=0;
    cudaMalloc((void**)&d_t_serie, t_bytes);
    cudaMemcpy(d_t_serie, t_serie, t_bytes, cudaMemcpyHostToDevice);
    cudaMalloc((void**)&d_query_norm,q_bytes);
	cudaMemcpy(d_query_norm, q_serie_norm, q_bytes, cudaMemcpyHostToDevice); //already nornalized
	cudaMalloc((void**)&d_owp,nss*sizeof(float));
	cudaMemset(d_owp, 0, nss*sizeof(float));


    printf("DTW GPU_GM version processing...\n");
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start,0);

    DTW_GM <<<grid, threads>>> (d_t_serie,d_query_norm,t_size,d_owp);
    checkCUDAError("Kernel DTW_GM");

    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&timeGM_GPU,start,stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);


    //TESTING
    float *d_owp_copy=(float*) malloc(nss*sizeof(float));
    cudaMemcpy(d_owp_copy, d_owp,nss*sizeof(float),cudaMemcpyDeviceToHost);
    min_val = min_arr(d_owp_copy,nss,ind_min_val);
    printf("ind_min_val_GPU_GM_version: %d, min_val_GPU_GM_version: %f\n\n",*ind_min_val,min_val);



    ////////////////////////DTW GPU_SM ALGORITHM//////////////////////// 
    float query_sm=(window_size)*sizeof(float);

    printf("DTW GPU_SM version processing...\n");
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start,0);

    DTW_SM <<<grid, threads, query_sm>>> (d_t_serie,d_query_norm,t_size,d_owp);
    checkCUDAError("Kernel DTW_SM1");

    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&timeSM_GPU,start,stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    min_val = min_arr(d_owp_copy,nss,ind_min_val);
    printf("ind_min_val_GPU_SM_version: %d, min_val_GPU_SM_version: %f\n\n",*ind_min_val,min_val);
    

    //Execution time
    printf("Execution time for DTW with CPU %f ms\n",timeCPU);
    printf("Execution time for DTW with GPU_GM %f ms\n",timeGM_GPU);
    printf("Execution time for DTW with GPU_SM %f ms\n",timeSM_GPU);


    cudaFree(d_t_serie);
    cudaFree(d_query_norm);
    cudaFree(d_t_serie);
    cudaFree(d_owp);  

    free(t_serie);
    free(x_t_serie);    
    free(q_serie);
    free(x_q_serie);
    free(q_serie_norm);
    free(owp);
    free(owp_ind);
    printf("\nMemory deallocated!\n\n");

    return 0;
}	
