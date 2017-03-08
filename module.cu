#include "header.h"

using namespace std;


__device__ void printArrayDev(float* array,int n){

    int i;
    for (i = 0; i < n; i++)
        printf("arr[%d]: %f \n",i, array[i]);
}



__device__ void printMatrixDev(float *matrix, int M, int N) {

    int i,j;
    for(i=0;i<M;i++) {
    for(j=0;j<N;j++)
        printf("%f ",matrix[i*N+j]);
    printf("\n");
    }
}


//not working (must be changed)
__device__ float stdDev(float *data, int n,float *avg)
{
    printf("N_SAMPLE: %d\n", n);
    printf("DATA_SIZE: %d\n", sizeof(data));
    float mean=0.0, sum_deviation=0.0;
    int i;
    for(i=0; i<n;++i)
    {
        mean+=data[i];
    }
    mean=mean/n;
    *avg=mean;
    for(i=0; i<n;++i)
    sum_deviation+=(data[i]-mean)*(data[i]-mean);

    return sqrt(sum_deviation/(n-1));
}



__global__ void MD_DTW_I(float* S, float* T, int ns, int nt, int dimensions, float* data_out, int trainSize, int task) {

    //int idx = blockIdx.x * blockDim.x + threadIdx.x; //i.e [0...6]*[512]+[0...511]

    int idx, offset_x;

    long long int i,j;  
    long long int k,l,g;  

    float min_nb=0;
    float array[WS][2];

    // float mean=0.0, std_dev=0.0;

    extern __shared__ float sh_mem[];

    float* T2 = (float *)sh_mem;
    float* DTW_single_dim = (float *)&sh_mem[dimensions*nt]; //offset on the shared memory for the segment T2
    // extern __shared__ float T2[];    


    if ( task == 0 ){
        idx = threadIdx.x*dimensions+threadIdx.y;
        offset_x = ((blockDim.x*blockDim.y*ns)*blockIdx.x)+idx*ns;

    if( ( (blockDim.x*blockDim.y*blockIdx.x)+idx) >= trainSize*dimensions) //120=train_size
        return;

    }
    else{ //SUBSEQ_SEARCH

        idx = threadIdx.x*dimensions+threadIdx.y;
        offset_x = (blockDim.x*blockIdx.x) + ((threadIdx.y * trainSize) + threadIdx.x); //use blockIdx and other measure to set well the offset

        if( (idx + WS) > trainSize)
            return;
        // else
        //     printf("thx: %d, thy: %d, idx: %d\n",threadIdx.x,threadIdx.y,idx);
        
    }

    if(idx==0){
        for (i = 0; i < dimensions; i++)
            for (j = 0; j < nt; j++)
                *(T2+(nt*i+j)) = T[nt*i+j];
    }
    __syncthreads(); //it may be avoided with the removing of the if condition
    // it may influences the algorithm performance (try!!!)

    // else
        // printf("block: %d, thread_x: %d, thread_y: %d IDX:%d\n",blockIdx.x, threadIdx.x, threadIdx.y,idx);



    // if(blockIdx.x==0) {
    //     for (i = 0; i < blockDim.y; i++)
    //         for (j = 0; j < nt; j++)
    //             printf("T2: %f\n", *T2+(nt*i+j));

    // }


    //NOT WORKING!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    //INSERT Z-NORMALIZATION STEP
    // std_dev = stdDev(&S[offset_x],ns,&mean);

    // for (i = 0; i < ns; i++)
    //     S[offset_x+i]=(S[offset_x+i]-mean)/std_dev;


   // if(blockIdx.x==0){
        
    

    k = 0;
    l = 1;

    //initialization step
    for(i=0;i<nt;i++)
    {
        if (i==0)
            array[i][k]=pow((S[offset_x]-T[nt*threadIdx.y]),2);
        else
            array[i][k]=pow((S[offset_x]-T[nt*threadIdx.y+i]),2)+array[i-1][k];
    }

    k = 1;
    l = 0;
    
    //computing DTW
    for(j=1; j<ns; j++)
    {
        i = 0;
        // array[i][k]=0.0;
        array[i][k]=pow((S[offset_x+j] - T[nt*threadIdx.y+i]),2)+array[i][l];

        for (i=1; i<nt; i++)
        {
            double a = array[i-1][l];
            double b = array[i][l];
            double c = array[i-1][k];
            
            min_nb = fminf(a,b);
            min_nb = fminf(c,min_nb);

            array[i][k]=pow((S[offset_x+j] - T[nt*threadIdx.y+i]),2)+min_nb;
        }
        g = k;
        k = l;
        l = g;
    }
    // data_out[idx] = array[WS-1][g];
    DTW_single_dim[idx] = array[WS-1][g];

    // printf("block: %d, thread_x: %d, thread_y: %d, IDX: %d, DTW: %f\n",blockIdx.x, threadIdx.x, threadIdx.y,idx,DTW_single_dim[idx]);
    __syncthreads();
    // if (blockIdx.x==1)
    // {
        
    if (idx==0) {
        for (i = 0; i < blockDim.x; i++) {
            data_out[(blockIdx.x*blockDim.x)+i] = 0.0;
            for (j = 0; j < blockDim.y; j++) {
                // printf("partial_DTW: %f\n", DTW_single_dim[i*dimensions+j]);
                // printf("offset: %d\n", (blockIdx.x*blockDim.x)+i);
                data_out[(blockIdx.x*blockDim.x)+i] += DTW_single_dim[i*dimensions+j]; //rivedere formula!
            }
            // printf("block: %d, thread_x: %d, thread_y: %d, data_out[%lld]: %f\n", blockIdx.x, threadIdx.x, threadIdx.y,(blockIdx.x*blockDim.x)+i,data_out[(blockIdx.x*blockDim.x)+i]);
        }
    }
    // printf("block: %d, thread_x: %d, thread_y: %d IDX:%d, OFFSET: %d S_val: %f\n",blockIdx.x, threadIdx.x, threadIdx.y,idx,offset,S[offset]);
    

// }
}



__global__ void rMD_DTW_D(float* S, float* T, int ns, int nt, int dimensions, float* data_out, int trainSize) {

    long long int k,l,g;
    
    long long int i,j,p;
    int idx = blockIdx.x * blockDim.x + threadIdx.x; //i.e [0...6]*[512]+[0...511]    


    float min_nb=0;
    float array[WS][2];
    // float instance[D][WS];


    extern __shared__ float T2[];


    // float sum = 0, sum_sqr = 0, mean = 0, mean_sqr = 0, variance = 0, std_dev = 0;


    //offset training set
    int s =dimensions*2*WS*(idx/WS);
    int t = s + idx%WS;


    if (idx >= (trainSize*ns) ) //
        return;
     

    if(threadIdx.x==0){
        for (i = 0; i < dimensions; i++)
            for (j = 0; j < nt; j++)
                T2[nt*i+j]=T[nt*i+j];
    }
    __syncthreads(); //it may be avoided with the removing of the if condition
    // it may influences the algorithm performance (try!!!)



    //NOT WORKING!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    //Z-NORMALIZATION
    // for (p = 0; p < dimensions; p++)
    // {
    //     std_dev = stdDev(&S[t+(dimensions*WS)],WS,&mean);
    //     // printf("block: %d, thread: %d mean: %f\n",blockIdx.x, threadIdx.x, mean);
    //     // printf("block: %d, thread: %d std_dev: %f\n",blockIdx.x, threadIdx.x, std_dev);

    //     for (i = t; i < dimensions*WS; i++)
    //     {
    //         S[t+i]=(S[t+i]-mean)/std_dev;
            
    //     }
    // }


    // if (blockIdx.x==0) {


    // if (idx==315) {
    //     for (i = 0; i < dimensions; i++) {
    //         for (j = 0; j <ns; j++) {
    //             printf("block_id: %d, thread: %d, S[%lld]: %f\n",blockIdx.x,t,(2*ns*i+t)+j,S[(2*ns*i+t)+j]);
    //         }
    //     }
    // }
    // __syncthreads();

    // }
 
    k = 0;
    l = 1;
    
    //computing first row (instace versus query)
    for(i=0;i<nt;i++)
    {
        array[i][k]=0.0;
        for (p = 0; p < dimensions; p++)
        {
            if (i==0)
                array[i][k]+=pow((S[t+p*2*ns]-T2[p*nt]),2);
            else
                array[i][k]+=pow((S[t+p*2*ns]-T2[p*nt+i]),2);
        }
        // printf("array[%d][%d]: %f\n",i,k,array[i][k]);
        if(i!=0)
            array[i][k]+=array[i-1][k];
        
    }

    k = 1;
    l = 0;
    
    
    for(j=1; j<ns; j++)
    {
        // printf("j: %d\n",j);
        i = 0;
        array[i][k]=0.0;

        for (p = 0; p < dimensions; p++){
            // float val=S[t+p*ns+j];
            array[i][k]+=pow((S[t+p*2*ns+j]-T2[p*nt+i]),2);
         }

        
        array[i][k]+=array[i][l];
        
        for (i=1; i<nt; i++)
        {
            // printf("i: %d\n",i);
            array[i][k]=0.0;
            float a = array[i-1][l];
            float b = array[i][l];
            float c = array[i-1][k];
            
            min_nb = fminf(a,b);
            min_nb = fminf(c,min_nb);

            for (p = 0; p < dimensions; p++)
                array[i][k]+=pow((S[t+p*2*ns+j]-T2[p*nt+i]),2);
            
            array[i][k]+=min_nb;
        }
        g = k;
        k = l;
        l = g;
    }


    data_out[idx]=array[nt-1][g];
    // printf("block_id: %d, idx: %d thread: %d, DTW: %f\n",blockIdx.x,idx,t,array[nt-1][g]);

// }

}



__global__ void MD_DTW_D(float* S, float* T, int ns, int nt, int dimensions, float* data_out, int trainSize, int task) {

    long long int k,l,g;
    
    long long int i,j,p;
    int idx = blockIdx.x * blockDim.x + threadIdx.x; 

    float min_nb=0;
    float array[WS][2];
    // float instance[D][WS];

    //query timeseries
    extern __shared__ float T2[];

    // float mean = 0, std_dev = 0;
    int t, offset;
    if( task == 0 ){

        offset = ns;

        int wind=dimensions*WS;
        t = idx*wind;
        if((idx*wind)+wind > trainSize*wind) //CHANGE FORMULA 120=train_size
            return;


        if(threadIdx.x==0){
            for (i = 0; i < dimensions; i++)
                for (j = 0; j < nt; j++)
                    T2[nt*i+j]=T[nt*i+j];
        }
        __syncthreads(); //it may be avoided with the removing of the if condition
        // it may influences the algorithm performance (try!!!)
    }
    else {

        offset = trainSize;

        t = idx;        
        if( (idx + WS) > trainSize)
            return;

        for (i = 0; i < dimensions; i++)
            for (j = 0; j < nt; j++)
                T2[nt*i+j] = T[nt*i+j];
        // __syncthreads();
        // if(threadIdx.x==0)
        //     printArrayDev(T2,dimensions*nt);

            
    }
    // __syncthreads();

    //NOT WORKING!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    //Z-NORMALIZATION
    // for (p = 0; p < dimensions; p++)
    // {
    //     std_dev = stdDev(&S[t+(dimensions*WS)],dimensions*WS,&mean);
    //     // printf("block: %d, thread: %d mean: %f\n",blockIdx.x, threadIdx.x, mean);
    //     // printf("block: %d, thread: %d std_dev: %f\n",blockIdx.x, threadIdx.x, std_dev);

    //     for (i = t; i < dimensions*WS; i++)
    //     {
    //         S[t+i]=(S[t+i]-mean)/std_dev;
            
    //     }
    // }

    // for(i=t; i<t+(D*WS); i++) {

    //   sum += S[i];
    //   sum_sqr += S[i] * S[i]; //pow(data[i],2);
    // }


    // // //measures for normalization step
    // mean = sum/WS;
    // mean_sqr = mean*mean;

    // variance = (sum_sqr/WS) - mean_sqr;
    // std_dev = sqrt(variance);


    // if(idx==1){
        // printMatrixDev(T,D,WS);
    // //normalizing subsequence C[t+1,t+2,...,t+WS]
        // instance2[0][0]=3.0;
        // printf("instance: %f\n",instance2[0][0] );
    // for(i=0; i<dimensions; i++){
    //     for (j = 0; j < ns; j++){
    //         // printf("val: %f\n",S[t+i*WS+j]);
    //         instance[i][j] = S[t+i*ns+j];
    //         // printf("instance[%d][%d]: %f\n",i,j,instance[i][j]);
    //     }
    // }
    
            
    //     // (data_in[t+i]-mean)/std_dev; 
    // }

 
    k = 0;
    l = 1;
    
    //computing first row (instace versus query)
    for(i=0;i<nt;i++)
    {
        array[i][k]=0.0;
        for (p = 0; p < dimensions; p++)
        {
            if (i==0)
                array[i][k]+=pow((S[t+p*offset]-T2[p*nt]),2);
            else
                array[i][k]+=pow((S[t+p*offset]-T2[p*nt+i]),2);
        }
        // printf("array[%d][%d]: %f\n",i,k,array[i][k]);
        if(i!=0)
            array[i][k]+=array[i-1][k];
        
    }
// }
    k = 1;
    l = 0;
    
    
    for(j=1; j<ns; j++)
    {
        // printf("j: %d\n",j);
        i = 0;
        array[i][k]=0.0;

        for (p = 0; p < dimensions; p++){
            // float val=S[t+p*ns+j];
            array[i][k]+=pow((S[t+p*offset+j]-T2[p*nt+i]),2);
         }

        
        array[i][k]+=array[i][l];
        
        for (i=1; i<nt; i++)
        {
            // printf("i: %d\n",i);
            array[i][k]=0.0;
            float a = array[i-1][l];
            float b = array[i][l];
            float c = array[i-1][k];
            
            min_nb = fminf(a,b);
            min_nb = fminf(c,min_nb);

            for (p = 0; p < dimensions; p++)
                array[i][k]+=pow((S[t+p*offset+j]-T2[p*nt+i]),2);
            
            array[i][k]+=min_nb;
        }
        g = k;
        k = l;
        l = g;
    }
    
    
    data_out[idx] = array[nt-1][g];
    // printf("data_out[%d]: %f\n",idx,data_out[idx]);
    // __syncthreads();
    
// }


    // data_out[idx] = array[WS-1][g];
    
    // return min;

}



__host__ int checkFlagOpts(char **input_args, int num_args, int ind, int num_opts){

    int count=0;
    printf("num_opts: %d\n", num_opts);
    char *pch = NULL;

    if (ind + num_opts < num_args ){ //it means a wrong number of options params and that there's no other flag option

        while (pch == NULL && count <= num_opts)
        {
            pch = strchr(input_args[ind],'-');
            // printf ("found at %d\n",pch);
            ind++;
            count++;
            // printf("count: %d\n",count);
        }

        if( count-1 != num_opts)
            return 0;
        else
            return 1;
    }
    else if(ind + num_opts > num_args )
        return 0;

    else
        return 1;
}


__host__ void readFileSubSeq(char **file_name, int *ind_files, int n_file, float *t_series, int t_size, float *q_series, int window_size, int n_feat, int read_mode){


int i,k;

FILE **inputFile = NULL;

inputFile = (FILE **)malloc(n_file*sizeof(FILE*));

for (i = 0; i < n_file; i++)
{
    char *curr_file = file_name[ind_files[i]];
    inputFile[i] = fopen(curr_file,"r");
    printf("file %s opened\n", curr_file);
    //FIXME: doesnt work whether a path file not exist
    if ((inputFile + i) == NULL ) {
        fprintf(stderr, "Failed to open file \"");
        fprintf(stderr, curr_file);
        fprintf(stderr, "\".\n");
        exit(2);
    }
}

float *tmp;

if(read_mode == 0){ // dimension on x axis (columns) and time on y axis (rows)
    tmp = (float*)malloc(n_feat*sizeof(float));

    //reading t_series file
    for(i = 0; i < t_size; i++){

        for (k = 0; k < n_feat; k++) {
            fscanf(inputFile[0],"%f",&tmp[k]); //fd=0 t_series descriptor
            t_series[(k*t_size)+i] = tmp[k];
        }
    }

    //reading q_series file
    for (i = 0; i < window_size; i++) {
        //reading data

        for (k = 0; k < n_feat; k++) {
            fscanf(inputFile[1],"%f",&tmp[k]); //fd=1 q_series descriptor
            q_series[(k*window_size)+i] = tmp[k];
                // printf("h_train_orig[%d]:\t %f\n",(n_feat*i*window_size)+(k*window_size)+j,data[(n_feat*i*window_size)+(k*window_size)+j]);
        }
       // printf("STEP_J: %d\n",j);
        // exit(-1);
    }
}
else if(read_mode == 1){ // time on x axis (row) and dimensions on y axis (columns)

        tmp = (float*)malloc(t_size*sizeof(float));

        for (k = 0; k < n_feat; k++) {
            for(i = 0; i < t_size; i++){
                fscanf(inputFile[0],"%f",&tmp[i]); //fd=0 t_series descriptor
                t_series[(k*window_size)+i] = tmp[i];
            }
        }
        free(tmp);
        
        tmp = (float*)malloc(window_size*sizeof(float));        

        for (k = 0; k < n_feat; k++) {
            for(i = 0; i < window_size; i++){
                fscanf(inputFile[1],"%f",&tmp[i]); //fd=1 q_series descriptor
                q_series[(k*window_size)+i] = tmp[i];
            }
        }        
    }
}


__host__ void readFile(
    char **file_name,
    int *ind_files,
    int n_file,
    int read_mode,
    float *data,
    struct data data_struct,
    int window_size,
    int *dataLabels,
    int n_feat,
    int class_alg
    ) {


FILE **inputFile = NULL;

inputFile = (FILE **)malloc(n_file*sizeof(FILE*));

for (int i = 0; i < n_file; i++)
{
    char *curr_file = file_name[ind_files[i]];
    inputFile[i] = fopen(curr_file,"r");
    printf("file %s opened\n", curr_file);
    //FIXME: doesnt work whether a path file not exist
    if ((inputFile + i) == NULL ) {
        fprintf(stderr, "Failed to open file \"");
        fprintf(stderr, curr_file);
        fprintf(stderr, "\".\n");
        exit(2);
    }
} 

int i,j,k;
float *tmp;
float label = 0;

//reading data from 1 big file
if (read_mode==0) { //read_mode=0

    tmp = (float*)malloc(n_feat*sizeof(float));

    // checking order input file
    fseek(inputFile[0], 0L, SEEK_END);

    int sz_file0 = ftell(inputFile[0]);
    fseek(inputFile[0], 0L, SEEK_SET);

    fseek(inputFile[1], 0L, SEEK_END);
    int sz_file1 = ftell(inputFile[1]);
    fseek(inputFile[1], 0L, SEEK_SET);

    // obtaining indices on the basis of the files order
    int lab_ind, data_ind; 

    if (sz_file0 > sz_file1) {
        lab_ind = 1;
        data_ind = 0;
    }
    else {
        lab_ind = 0;
        data_ind = 1;
    }

    for(i = 0; i < data_struct.tot_size; i++) {

            //reading labels
            fscanf(inputFile[lab_ind],"%f",&label); //fd=1 label descript
                 
            dataLabels[i] = (int)label;
            // printf("label: %d\n",dataLabels[i]);
            //exit(-1);
            for (j = 0; j < window_size; j++) {
                //reading data
                for (k = 0; k < n_feat; k++)
                    fscanf(inputFile[data_ind],"%f",&tmp[k]); //fd=0 data descript
                
                for (k = 0; k < n_feat; k++) {

                    if (class_alg < 2) { // MDT_D or MDT_I
                        data[(n_feat*i*window_size)+(k*window_size)+j] = tmp[k];
                        // printf("h_train_orig[%d]:\t %f\n",(n_feat*i*window_size)+(k*window_size)+j,data[(n_feat*i*window_size)+(k*window_size)+j]);
                    }
                    else {

                        data[(n_feat*2*i*window_size)+(2*k*window_size)+j] = tmp[k];
                        data[(n_feat*2*i*window_size)+((2*k*window_size)+window_size)+j] = tmp[k];
                        // printf("h_train_orig[%d]:\t %f\n",(n_feat*2*i*window_size)+(2*k*window_size)+j,data[(n_feat*2*i*window_size)+(2*k*window_size)+j]);
                        // printf("h_train_copy[%d]:\t %f\n",(n_feat*2*i*window_size)+((2*k*window_size)+window_size)+j,data[(n_feat*2*i*window_size)+((2*k*window_size)+window_size)+j]);
                    }
                }
               // printf("STEP_J: %d\n",j);
                // exit(-1);
            }
           // printf("STEP_I: %d\n",i);
        }
    }

//reading from k-files
else if(read_mode == 1) {

    tmp = (float*)malloc(n_feat*sizeof(float));
    // n_file = n_feat;

    for(i = 0; i < data_struct.tot_size; i++) {
        //reading labels 
        for (k = 0; k < n_feat; k++)
            fscanf(inputFile[k],"%f", &label);


            dataLabels[i] = (int)label;

            for (j = 0; j < window_size; j++) {
                //reading data
                for (k = 0; k < n_feat; k++)
                    fscanf(inputFile[k],"%f",&tmp[k]);

                for (k=0; k<n_feat;k++) {

                    if (class_alg<2) { // MDT_D or MDT_I
                        data[(n_feat*i*window_size)+(k*window_size)+j] = tmp[k];
                            // printf("h_train_orig[%d]:\t %f\n",(n_feat*i*window_size)+(k*window_size)+j,data[(n_feat*i*window_size)+(k*window_size)+j]);
                    }
                    else {
                        data[(n_feat*2*i*window_size)+(2*k*window_size)+j] = tmp[k];
                        data[(n_feat*2*i*window_size)+((2*k*window_size)+window_size)+j] = tmp[k];
                            // printf("h_train_orig[%d]:\t %f\n",(3*2*i*window_size)+(2*k*window_size)+j,data[(3*2*i*window_size)+(2*k*window_size)+j]);
                            // printf("h_train_copy[%d]:\t %f\n",(3*2*i*window_size)+((2*k*window_size)+window_size)+j,data[(3*2*i*window_size)+((2*k*window_size)+window_size)+j]);
                    }
                }
            // exit(-1);
            }
        }
}
else {

    tmp = (float*)malloc(window_size*sizeof(float));

    printf("mode 2, train_size: %d, test_size: %d!\n",data_struct.train_size,data_struct.test_size);
    // exit(-1);

    int i = 0;

    int size_arr[2] = {data_struct.train_size, data_struct.test_size};

    for (int ll = 0; ll < n_file; ll++) {
        for(int inn=0; inn < size_arr[ll]; inn++) {

            //reading data
            for (k = 0; k < n_feat; k++) {

                //reading labels from either train or test set
                fscanf(inputFile[ll], "%f", &label);
                // printf("%f\n",label);
                     
                dataLabels[i] = (int)label;
            
                for (j = 0; j < window_size; j++) {

                    fscanf(inputFile[ll],"%f",&tmp[j]); //fd=0 data descript

                    if (class_alg < 2) { // MDT_D or MDT_I
                        data[(n_feat*i*window_size) + (k*window_size)+j] = tmp[j];
                        // printf("h_train_orig[%d]:\t %f\n",(n_feat*i*window_size)+(k*window_size)+j,data[(n_feat*i*window_size) + (k*window_size)+j]);
                        // exit(-1);
                    }
                    else {
                        //TODO: CHECK AND FIX FORMULA
                        data[(n_feat*2*i*window_size)+(2*k*window_size)+j] = tmp[j];
                        data[(n_feat*2*i*window_size)+((2*k*window_size)+window_size)+j] = tmp[j];
                        // printf("h_train_orig[%d]:\t %f\n",(n_feat*2*i*window_size)+(2*k*window_size)+j,data[(n_feat*2*i*window_size)+(2*k*window_size)+j]);
                        // printf("h_train_copy[%d]:\t %f\n",(n_feat*2*i*window_size)+((2*k*window_size)+window_size)+j,data[(n_feat*2*i*window_size)+((2*k*window_size)+window_size)+j]);
                    }
                }
           // printf("STEP_J: %d\n",j);
            // exit(-1);
            }
            i++;
        }
       // printf("STEP_I: %d\n",i);
        
    }

} //END ELSE

printf("DONE reading\n");
//Closing and deallocatin all files
for (k=0; k<n_file; k++)
    fclose(inputFile[k]);

free(inputFile);


}

__host__ void createTrainingTestingSet(float *data, int *dataLabels, int dataSize, int window_size, int n_feat, float *h_train, int *trainLabels, int trainSize, float *h_test, int *testLabels, int testSize, int *tInd, int k_th_fold, int class_mode) {

    int i,j,k,i_train=0,i_test=0;

    if(tInd != NULL) {
        /* Creating Training and Testing set */
        for (i=0; i<dataSize; i++) {

            if(tInd[i] != k_th_fold) //training set
            {
                trainLabels[i_train] = dataLabels[i];
                // printf("Label: %d\n",trainLabels[i_train] );
                // printf("i=%d TRAINING_IND: %d, i_train: %d\n",i+1,tInd[i],i_train);

                for (j=0; j<window_size; j++)//j
                {
                    //reading data
                    // for (k=0; k<n_file; k++)
                    // fscanf(&inputFile[k],"%f",&tmp[k]);

                    for (k=0; k<n_feat;k++) 
                    {
                        // h_train[(3*2*i_train*window_size)+(2*k*window_size)+j]=0;
                        if(class_mode<2){
                            h_train[(n_feat*i_train*window_size)+(k*window_size)+j] = data[(n_feat*i*window_size)+(k*window_size)+j];
                            // printf("h_train_orig[%d]:\t %f\n",(n_feat*i*window_size)+(k*window_size)+j,data[(n_feat*i*window_size)+(k*window_size)+j]);
                            // h_train[(n_feat*i_train*window_size)+((k*window_size)+window_size)+j] = data[(n_feat*i*window_size)+((k*window_size)+window_size)+j];
                        }
                        else
                        {
                            h_train[(n_feat*2*i_train*window_size)+(2*k*window_size)+j] = data[(n_feat*2*i*window_size)+(2*k*window_size)+j];
                            h_train[(n_feat*2*i_train*window_size)+((2*k*window_size)+window_size)+j] = data[(n_feat*2*i*window_size)+((2*k*window_size)+window_size)+j];
                           // printf("h_train_orig[%d]:\t %f\n",(3*2*i_train*window_size)+(2*k*window_size)+j,h_train[(3*2*i_train*window_size)+(2*k*window_size)+j]);
                           // printf("h_train_copy[%d]:\t %f\n",(3*2*i_train*window_size)+((2*k*window_size)+window_size)+j,h_train[(3*2*i_train*window_size)+((2*k*window_size)+window_size)+j]);
                        }
                    }
                }
                i_train++;
            }
            else //testing set
            {
                testLabels[i_test] = dataLabels[i];
                // printf("Label: %d\n",testLabels[i_test] );
                // printf("i=%d TESTING_IND: %d, i_test: %d\n",i+1,tInd[i],i_test);
                for (j=0; j<window_size; j++) {

                    for (k=0; k<n_feat;k++) 
                    {
                        if(class_mode < 2)
                        {
                            h_test[(window_size*n_feat*i_test)+window_size*k+j] = data[(n_feat*i*window_size)+(k*window_size)+j];   
                        }
                        else 
                        {
                            h_test[(window_size*n_feat*i_test)+window_size*k+j] = data[(n_feat*2*i*window_size)+(2*k*window_size)+j];
                            // printf("h_test_orig[%d]:\t %f\n",(window_size*n_feat*i_test)+window_size*k+j,h_test[(window_size*n_feat*i_test)+window_size*k+j]);
                        }
                    }
                }
                i_test++;
            }
        }
    }
    else{

        int i = 0;
        // int size_arr[2] = {trainSize, testSize};
        
        for(int i_train = 0; i < trainSize; i++) {

            trainLabels[i_train] = dataLabels[i];

            for (j=0; j<window_size; j++) {

                //reading data
                // for (k=0; k<n_file; k++)
                // fscanf(&inputFile[k],"%f",&tmp[k]);

                for (k=0; k<n_feat;k++)
                {
                    // h_train[(3*2*i_train*window_size)+(2*k*window_size)+j]=0;
                    if(class_mode < 2){
                        h_train[(n_feat*i_train*window_size)+(k*window_size)+j] = data[(n_feat*i*window_size)+(k*window_size)+j];
                        // printf("h_train_orig[%d]:\t %f\n",(n_feat*i*window_size)+(k*window_size)+j,data[(n_feat*i*window_size)+(k*window_size)+j]);
                        // h_train[(n_feat*i_train*window_size)+((k*window_size)+window_size)+j] = data[(n_feat*i*window_size)+((k*window_size)+window_size)+j];
                    }
                    else
                    {
                        h_train[(n_feat*2*i_train*window_size)+(2*k*window_size)+j] = data[(n_feat*2*i*window_size)+(2*k*window_size)+j];
                        h_train[(n_feat*2*i_train*window_size)+((2*k*window_size)+window_size)+j] = data[(n_feat*2*i*window_size)+((2*k*window_size)+window_size)+j];
                       // printf("h_train_orig[%d]:\t %f\n",(3*2*i_train*window_size)+(2*k*window_size)+j,h_train[(3*2*i_train*window_size)+(2*k*window_size)+j]);
                       // printf("h_train_copy[%d]:\t %f\n",(3*2*i_train*window_size)+((2*k*window_size)+window_size)+j,h_train[(3*2*i_train*window_size)+((2*k*window_size)+window_size)+j]);
                    }
                }
            }
            i_train++;
        }

        for(int i_test = 0; i_test < testSize; i++) {

            testLabels[i_test] = dataLabels[i];

            for (j = 0; j < window_size; j++) {
                
                for (k = 0; k < n_feat;k++) 
                {
                    if(class_mode < 2)
                        h_test[(window_size*n_feat*i_test)+window_size*k+j] = data[(n_feat*i*window_size)+(k*window_size)+j];   
                    else 
                        h_test[(window_size*n_feat*i_test)+window_size*k+j] = data[(n_feat*2*i*window_size)+(2*k*window_size)+j];
                        // printf("h_test_orig[%d]:\t %f\n",(window_size*n_feat*i_test)+window_size*k+j,h_test[(window_size*n_feat*i_test)+window_size*k+j]);
                }
            }
            i_test++;
        }
    }

}



__host__ int cmpfunc (const void * a, const void * b) {
    return ( *(int*)a - *(int*)b );
}



__host__ void generateArray(int size,int *arrayG,int offset) {
    
    int i,j=0;
    for (i=offset; size>0; i++){
        arrayG[j++]=i;
        size--;
    }
}



__host__ void findInd(int* array, int size, int* arrayG, int g) {
    
    int i,j=0;
    for (i=0; i<size; i++) {
        if(array[i]==g){
            arrayG[j++]=i;
        }
    }
}



__host__ int unique_val(int* array,int size) {
    
    
    int i;
    
    qsort(array, size, sizeof(int), cmpfunc);
    
    int unique = 1; //incase we have only one element; it is unique!
    
    for(i = 0; i < size -1 /*since we don't want to compare last element with junk*/; i++)
    {
        if(array[i]==array[i+1]){
            continue;
        }
        else{
            unique++;
        }
    }
    return unique;
    
}



__host__ int* accumarray(int* array,int size,int* val) {
    
    
    int i,j=0;
    
    int u_val=unique_val(array,size);
    // printf("u_val: %d\n",u_val);
    
    int* nS=(int*)malloc(u_val*sizeof(int));
    memset(nS, 0, u_val*sizeof(int));
    
    
    for(i=0;i<size;i++)
    {
        if(array[i]==array[i+1]){
            nS[j]++;
            continue;
        }
        else{
            val[j]=array[i];
            nS[j]++;
            j++;
        }
    }
    
    return nS;
    
}


__host__ void shuffle(int *array, size_t array_size, size_t shuff_size) {

    if (array_size > 1)
    {
        size_t i;
        for (i = 0; i < shuff_size - 1; i++)
        {
            size_t j = i + rand() / (RAND_MAX / (array_size - i) + 1);
            int t = array[j];
            array[j] = array[i];
            array[i] = t;
        }
    }
}



__host__ void idAssign(int* perm,int size_perm,int* group,int size_group,int* rand_ind,int* h,int* tInd) {
    
    
    int i;
    int group_perm;
    for (i=0; i<size_group;i++) {
        group_perm=perm[group[i]];
        //        printf("%d ",group_perm);
        //        printf("rand_ind[%d]: %d\n",i,rand_ind[i]);
        //        printf("h[%d]: %d\n",i,h[i]);
        //        printf("h[rand_ind[%d]]: %d\n",i,h[rand_ind[i]]);
        tInd[h[rand_ind[i]]]=group_perm;
        //        printf("%d-%d ",h[rand_ind[i]],tInd[h[rand_ind[i]]]);
        
    }
    //        printf("\n");
    
}

__host__ void checkCUDAError (const char* msg) { 

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess ) {
        fprintf(stderr, "Cuda error: %s %s\n",msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}


__host__ int* crossvalind_Kfold(int* label,int N,int K) {
    
    
    //    int N=sizeof(label)/sizeof(int);
    printf("label size: %d\n",N);
    
    int* label_copy=(int *)malloc(N*sizeof(int));
    memcpy(label_copy, label, N*sizeof(int));
    
    //output
    int* tInd=(int *)malloc(N*sizeof(int));
    memset(tInd, 0, N*sizeof(int));
    
    
    int ul=unique_val(label_copy, N);
    
    int *arr_val=(int*)malloc(ul*sizeof(int));
    
    int* nS=accumarray(label_copy,N,arr_val);
    
    int i,j;
    // for (i=0; i<ul; i++) {
    //     printf("val[%d]: %d\n",arr_val[i],nS[i]);
    // }
    //    exit(-1);
    int* pq =(int*)malloc(K*sizeof(int));
    generateArray(K, pq,0);
    
    for (i=0; i<ul; i++) {
        
        int* randInd=(int*)malloc(nS[i]*sizeof(int));
        generateArray(nS[i], randInd,0);
        
        int* q=(int*)malloc(nS[i]*sizeof(int));
        int* h=(int*)malloc(nS[i]*sizeof(int));
        //        printf("val: %d, nS: %d\n",arr_val[i],nS[i]);
        //        generateArray(nS[i], h, offset);
        findInd(label, N, h, arr_val[i]);
        //        printArray(h, nS[i]);
        //        exit(-1);
        
        for (j=0; j<nS[i]; j++) {
            float val=(float)(K*(j+1))/nS[i]; //j+1 because we need no zero values; MATLAB: q = ceil(K*(1:nS(g))/nS(g));
            q[j]=(int)ceil(val)-1; //C indices start from 0
        }
        shuffle(pq,K,K);
        //        printf("pq: ");
        //        printArray(pq, K);
        //        printf("q: ");
        //        printArray(q, nS[i]);
        //        printf("h: ");
        //        printArray(h, nS[i]);
        shuffle(randInd,nS[i],nS[i]);
        //        printf("randInd: ");
        //        printArray(randInd, nS[i]);
        
        idAssign(pq, K, q, nS[i],randInd,h,tInd);
        //        exit(-1);
        
        free(randInd);
        free(q);
        free(h);
    }
    
    return tInd;
    
}



__host__ int countVal(int *data,int N,int key) {

    int i,cnt=0;
    for (i=0; i<N; i++) {
        if(data[i]==key)
            cnt++;
    }
    return cnt;
}




__host__ float standard_deviation(float *data, int n,float *avg)
{
    float mean=0.0, sum_deviation=0.0;
    int i;
    for(i=0; i<n;++i)
    {
        mean+=data[i];
    }
    mean=mean/n;
    *avg=mean;
    for(i=0; i<n;++i)
    sum_deviation+=(data[i]-mean)*(data[i]-mean);
    return sqrt(sum_deviation/(n-1));
}




__host__ void z_normalize2D(float *M, int nrow,int ncol) {

    int i;
    float std_dev=0;
    float *mean=(float*)malloc(sizeof(float));

    for (i = 0; i < nrow; i++) {
        std_dev=0;
        *mean=0;
        // for (j = 0; j < ncol; j++){
        //     sum+=M[i*nrow+j];
        //     sum_sqr+=M[i*nrow+j]*M[i*nrow+j];
        // }
        // mean=sum/ncol;
        // mean_sqr=mean*mean;
        // variance = (sum_sqr/ncol) - mean_sqr;
        // std_dev = sqrt(variance);
        std_dev=standard_deviation(&M[i*ncol],ncol,mean);
        // printf("mean: %f\n", *mean);
        // printf("std: %f\n",std_dev);
        for (int k = 0; k < ncol; k++){
            M[i*ncol+k]=(M[i*ncol+k]-(*mean))/std_dev;
        }
    }
    free(mean);

}



__host__ float short_dtw_c(float *instance, float *query,int ns, int nt) {


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



__host__ float short_md_dtw_c(float *S, float *T, int ns, int nt, int dim, int offset){
    
    
    int k, l , g;
    long long int i,j;
    float **array;
    float min_nb;
    

    // create array
    array = (float **)malloc((nt)*sizeof(float *));
    for(i = 0;i < nt; i++)
    {
        array[i] = (float *)malloc((2)*sizeof(float));
    }

    
    k = 0;
    l = 1;
    
    //computing first row (instace versus query)
    for(i=0;i<nt;i++)
    {
        array[i][k]=0.0;
        for (int p = 0; p < dim; p++)
        {
            if (i==0)
                array[i][k]+=pow((S[p*offset+i]-T[p*nt+i]),2); //squared difference (ins[0]-query[0])^2
            else
                array[i][k]+=pow((S[p*offset+0]-T[p*nt+i]),2); //initally, array[i-1][k] is always the minimum
        }
        if(i!=0)
            array[i][k]+=array[i-1][k];
    }

    k = 1;
    l = 0;
    
    
    for(j=1; j<ns; j++)
    {
        i = 0;
        array[i][k]=0.0;
        for (int p = 0; p < dim; p++)
            array[i][k]+=pow((S[p*offset+j]-T[p*nt+i]),2);
        
        array[i][k]+=array[i][l];
        
        for (i=1; i<nt; i++)
        {
            array[i][k]=0.0;
            float a = array[i-1][l];
            float b = array[i][l];
            float c = array[i-1][k];
            
            min_nb = fminf(a,b);
            min_nb = fminf(c,min_nb);

            for (int p = 0; p < dim; p++)
                array[i][k]+=pow((S[p*offset+j]-T[p*nt+i]),2);
            
            array[i][k]+=min_nb;
        }
        g = k;
        k = l;
        l = g;
    }
    
    
    float min=array[nt-1][g];
    
    return min;
}



//./mdtwObj -i 3 128 1 ../DATASET/HandwritingGyroscope/X_MAT ../DATASET/HandwritingGyroscope/Y_MAT ../DATASET/HandwritingGyroscope/Z_MAT -o 1000 152 2 0 -d 0
__host__ void print_help(void) {
    
    fprintf(stderr,
            "\nUsage: MD_DTW_Classification [OPTIONS]\n"
            "Multi-Dimensional Time Serie Classification (MD-TSC) using Multi-Dimensional Dynamic Time Warping\n"
            "\n"
            "OPTIONS:\n"
            "-t Task           \t\tParameter\n"
            "String value       \t\tThis parameter represents the kind of task you want to perform (CLASSIFICATION or SUBSEQ_SEARCH)\n\n"
            "-i Input           \t\tParameter\n"
            "Integer values     \t\tThe first argument represents the dimensionality of the Time Series (TS) (e.g., 1,2,3, ect)\n"
            "Integer values     \t\tThe second argument represents the desired number of threads with whom the kernel will be executed (e.g., 64,128,...,1024)\n"
            "Integer values     \t\tThe third argument represents the type of reading mode adopted (0=big file, 1=K-file)\n\n"
            "-f Files           \t\tParameter\n"
            "String values      \t\tFollow two or more text file representing the data (fore more information about the structure of these files see the README file provided with the software)\n\n"
            "-k Cross Validation \t\tParameter\n"
            "Integer values     \t\tThis parameter specify the number of K-fold to use int he K-cross validation step\n\n"
            "-o Option Parameters \t\tParameter.\n"
            "Integer values     \t\tThe first argument represents the size of the dataset (number of sample)\n"
            "Integer values     \t\tThe second argument represents the window size of the TS\n\n"
            "-m Algorithm Mode  \t\tParameter\n"
            "Integer values     \t\tThis parameter represents the type of MD_DTW algorithm to use in the classification (see the README file for more information)\n\n"
            "-d Device Choice   \t\tParameter\n"
            "Integer values     \t\tThis parameter specify the GPU device (on your machine) you want to use to execute the  MD_DTW_Classification algorithm\n\n"
            "--version          \t\tDisplay version information.\n"
            "--help             \t\tDisplay help information.\n"
            "\n"
            "e.g.\n"
            "./mdtwObj -t CLASSIFICATION -i GPU 3 128 1 ../DATASET/X_MAT ../DATASET/Y_MAT ../DATASET/Z_MAT -k 10 -o 1000 152 -m 0 -d 0\n"
            "./mdtwObj -t SUBSEQ_SEARCH -i CPU 1 0 -f ../DATASET/SUB_SEQ_SEARCH/ECGseries ../DATASET/SUB_SEQ_SEARCH/ECGquery -o 3907 421 -m 0 -d 0\n");
    exit(0);
}

//TO CHANGE
__host__ void print_version(void) {
    
    fprintf(stderr,
            "MD_DTW_Classification version 0.1.0\n"
            "Copyright (C) 2016 Davide Nardone <davide.nardone@live.it>\n"
            "Originally inspired by Doruk Sart et al\n"
            "http://alumni.cs.ucr.edu/~mueen/GPU_DTW/index.html\n"
            "\n"
            "See the README file for license information.\n");
    exit(0);
}



__host__ void infoDev() {

    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    printf("Number of device: %d\n",deviceCount);
    int device;
    cudaDeviceProp deviceProp;
    //retrieving all devices
        for (device = 0; device < deviceCount; ++device)
        {
            //getting information about i-th device
            cudaGetDeviceProperties(&deviceProp, device);
            //printing information about i-th device
            printf("\n\n>>>>>>>>>>>>>>>>>>\nSelected device:%d\n<<<<<<<<<<<<<<<<<<\n\n", device);
            printf("\ndevice %d : %s\n",device,deviceProp.name);
            printf("major/minor : %d.%d compute capability\n",deviceProp.major,deviceProp.minor);
            printf("Total global mem : %lu bytes\n", deviceProp.totalGlobalMem);
            printf("Shared block mem : %lu bytes\n",deviceProp.sharedMemPerBlock);
            printf("Max memory pitch : %lu bytes\n",deviceProp.memPitch);
            printf("RegsPerBlock : %d \n", deviceProp.regsPerBlock);
            printf("WarpSize : %d \n", deviceProp.warpSize);
            printf("MaxThreadsPerBlock : %d \n", deviceProp.maxThreadsPerBlock);
            printf("TotalConstMem : %lu bytes\n", deviceProp.totalConstMem);
            printf("ClockRate : %d (kHz)\n", deviceProp.clockRate);
            printf("deviceOverlap : %d \n", deviceProp.deviceOverlap);
            printf("deviceOverlap : %d \n", deviceProp.deviceOverlap);
            printf("MultiProcessorCount: %d \n", deviceProp.multiProcessorCount);
            printf("\n");
        }
        exit(-1);
}


__host__ cudaDeviceProp getDevProp(int device) {

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, device);

    return deviceProp;

}



__host__ void initializeArray(float *array, int n, float val) {
    int i;
    for (i = 0; i < n; i++)
        array[i] = (float)i+1;
}


__host__ void initializeMatrix(float *matrix, int M, int N) {

    int i,j;
    for(i=0;i<M;i++)
        for(j=0;j<N;j++)
            matrix[i*N+j]=((float) rand()) / (float) RAND_MAX;
}


__host__ void printArray(float* array, int n) {

    int i;
    for (i = 0; i < n; i++)
        printf("val[%d]: %f\n",i,array[i]);
    printf("\n");
}


__host__ void printArrayI(int* array, int n) {

    int i;
    for (i = 0; i < n; i++)
        printf("val[%d]: %d\n",i,array[i]);
    printf("\n");
}


__host__ void printMatrix(float *matrix, int M, int N) {
    int i,j;
    for(i=0;i<M;i++) {
    for(j=0;j<N;j++)
        printf("%f\n", matrix[i*N+j]);
    printf("\n");
    }
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


__host__ void fakeK_fold(int *array, int n,int m) {
    int i,j;
    for (i = 0; i < n; i++)
        array[i] = 1; //train
    for (j = i; j < m+n; j++)
        array[i] = 0; //test
}



__host__ float min_arr(float *arr,int n,int *ind) {

    float min = FLT_MAX;
    *ind = -1;
    for (int i = 0; i < n; ++i)
    {
        if (arr[i] < min){
            min = arr[i];
            *ind = i;
        }
    }

    return min;
}


