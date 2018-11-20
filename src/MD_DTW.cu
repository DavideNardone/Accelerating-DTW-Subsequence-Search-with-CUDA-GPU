#include "header.h"

using namespace std;


int main( int argc, char** argv)
{
    struct timeval stop_CPU, start_CPU;
    cudaEvent_t start_GPU, stop_GPU;

    float time_GPU_MD_DTW_D,time_GPU_MD_DTW_I,time_GPU_rMDTW, elapsed;

    int i,j,k,f,nss; 

    int t_size,q_size;

    int read_mode=0;
    int window_size = 0;
    int dataSize = 0;
    int train_size, test_size;
    int blockSize = 0;
    int k_fold = 0;
    int flag_shuffle = 0;
    int n_feat=0;
    int device = 0;
    int gm;
    float T2;

    int num_opts;
    int flag_task = 0;
    int flag_in = 0;
    int flag_file = 0;
    int flag_opt = 0;
    int flag_device = 0;
    int flag_cross = 0;
    int flag_alg_mode = 0;
    int flag_verbose = 0;
    int verbose_mode = 1; //by default display all the outputs
    int n_file=0;
    int class_mode = 0;
    char *task, *compution_type, *distance_type;
    const char *strategy;
    int *arr_num_file;
    int* tInd = NULL;
    struct data data_struct;

    int err=0, errNR=0, *minI, minINR=-1;
    float RIA=0.0f, RA=0.0f, ER_RIA=0.0f, ER_RA=0.0f, min, minNR;

    time_t t;
    /* Intializes random number generator */
    srand((unsigned) time(&t));


    /* ******************************************* ARGUMENT PARSING ******************************************* */
    for(i = 1; i < argc; i++){

        if(flag_task){
            num_opts = 1;

            task = argv[i];

            j = 0;
            do { task[j] = toupper(task[j]); //uppercase string
            } while(task[j++]);


            if(!checkFlagOpts(argv,argc,i,num_opts)) {
                printf("The number of options is incorrect. For more information run: %s --help\n", argv[0]);
                exit(-1);
            }

            flag_task = 0;
        }
        else if(flag_in){

            compution_type = argv[i];

            j = 0;
            do { compution_type[j] = toupper(compution_type[j]); //uppercase string
            } while(compution_type[j++]);
            
            if (strcmp(compution_type,"GPU") == 0){
                num_opts = 3;

                if(!checkFlagOpts(argv,argc,i,num_opts)){
                    printf("The number of options is incorrect. For more information run: %s --help\n", argv[0]);
                    exit(-1);
                }

                n_feat = atoi(argv[i+1]);
                blockSize = atoi(argv[i+2]);
                read_mode = atoi(argv[i+3]);

                if ( blockSize > 1024 || blockSize < 0 ){
                    printf("Irregular number of threads for block. The number of threads for block has to be included in [0, 1024]\n");
                    exit(-1);
                }

                i = i + 3;
            }
            else if((strcmp(compution_type,"CPU") == 0) ){
                num_opts = 2;

                if(!checkFlagOpts(argv,argc,i,num_opts)){
                    printf("The number of options is incorrect. For more information run the execution as: %s --help\n", argv[0]);
                    exit(-1);
                }

                n_feat = atoi(argv[i+1]);
                read_mode = atoi(argv[i+2]);
                i = i + 2;
            }
            else{
                printf("The number of options is incorrect. For more information run the execution as: %s --help\n", argv[0]);
                exit(-1);
            }

            flag_in = 0;
        }
        else if(flag_file){
            if(strcmp(task,"CLASSIFICATION") == 0){

                if (read_mode == 0 || read_mode == 2){
                    n_file = 2;
                    num_opts = n_file;

                    if(!checkFlagOpts(argv,argc,i,num_opts)) {
                        printf("The number of options is incorrect. For more information run the execution as: %s --help\n", argv[0]);
                        exit(-1);
                    }
                }
                else if (read_mode == 1){
                    n_file = n_feat;
                    num_opts = n_feat;

                    if(!checkFlagOpts(argv,argc,i,num_opts)) {
                        printf("The number of options is incorrect. For more information run the execution as: %s --help\n",argv[0]);
                        exit(-1);
                    }
                }
                else{
                    printf("The number of options is incorrect. For more information run the execution as: %s --help\n", argv[0]);
                    exit(-1);                
                }
            }
            else if ( strcmp(task,"SUBSEQ_SEARCH") == 0 ){
                n_file = 2;
                num_opts = n_file;

                if(!checkFlagOpts(argv,argc,i,num_opts)) {
                    printf("The number of options is incorrect. For more information run the execution as: %s --help\n", argv[0]);
                    exit(-1);
                }
            }

            arr_num_file = (int *)malloc(n_file*sizeof(int));

            int j = 0;
            int cc = n_file;
            while(cc > 0) {
                arr_num_file[j] = i;

                i++;
                j++;
                cc--;
            }
            i--;

            flag_file = 0;
        }
        else if (flag_cross){            
            num_opts = 2;

            if(!checkFlagOpts(argv,argc,i,num_opts)) {
                printf("The number of options is incorrect. For more information run the execution as: %s --help\n", argv[0]);
                exit(-1);
            }

            k_fold = atoi(argv[i]);
            flag_shuffle = atoi(argv[i+1]);
            if (k_fold < 2) {
                printf("It's not possible to perform %d-fold-cross validation! The number of folds has to be greater than 2.\n", k_fold);
                exit(-1);
            }
            i += 1;

            flag_cross = 0;
        }        
        else if (flag_opt) {
            
            if( strcmp(task,"CLASSIFICATION") == 0 ){
                num_opts = 3;
                if (k_fold > 0) {
                    dataSize = atoi(argv[i]);
                    data_struct.tot_size = dataSize;
                    data_struct.train_size = 0;
                    data_struct.test_size = 0;
                    window_size = atoi(argv[i+1]);
                    i = i + 1;
                }
                else {
                    if(!checkFlagOpts(argv,argc,i,num_opts)) {
                        printf("The number of options is incorrect. For more information run the execution as: %s --help\n",argv[0]);
                        exit(-1);
                    }

                    train_size = atoi(argv[i]);
                    test_size = atoi(argv[i+1]);

                    data_struct.train_size = train_size;
                    data_struct.test_size = test_size;
                    dataSize = train_size + test_size;
                    data_struct.tot_size = dataSize;

                    window_size = atoi(argv[i+2]);

                    i = i + 2;
                }
            }
            else if ( strcmp(task,"SUBSEQ_SEARCH") == 0 ){
                num_opts = 2;

                if(!checkFlagOpts(argv,argc,i,num_opts)) {
                    printf("The number of options is incorrect. For more information run the execution as: %s --help\n",argv[0]);
                    exit(-1);
                }
                t_size = atoi(argv[i]);
                q_size = atoi(argv[i+1]);
                i = i + 1;
            }
            else{
                printf("The number of options is incorrect. For more information run the execution as: %s --help\n",argv[0]);
                exit(-1);
            }

            flag_opt = 0;
        }

        else if (flag_alg_mode){
            num_opts = 2;

            if(!checkFlagOpts(argv,argc,i,num_opts)) {
                printf("The number of options is incorrect. For more information run the execution as: %s --help\n",argv[0]);
                exit(-1);
            }              
         
            class_mode = atoi(argv[i]);
            if(class_mode == 0)
                strategy = "DEPENDENT";
            else if(class_mode == 1)
                strategy = "INDEPENDENT";
            else
                strategy = "ROTATION INVARIANT";

            distance_type = argv[i+1];
            i = i + 1;

            flag_alg_mode = 0;
        }
        else if (flag_device){
            num_opts = 1;
            
            if(!checkFlagOpts(argv,argc,i,num_opts)) {
                printf("The number of options is incorrect. For more information run the execution as: %s --help\n",argv[0]);
                exit(-1);
            }

            device = atoi(argv[i]);

            flag_device = 0;
        }
        else if (flag_verbose){
            num_opts = 1;

            if(!checkFlagOpts(argv,argc,i,num_opts)) {
                printf("The number of options is incorrect. For more information run the execution as: %s --help\n",argv[0]);
                exit(-1);
            }

            verbose_mode = atoi(argv[i]);

            flag_verbose = 0;

        }
        else if(!strcmp(argv[i], "-t"))
            flag_task = 1;
        else if(!strcmp(argv[i], "-i"))
            flag_in = 1;
        else if(!strcmp(argv[i], "-f"))
            flag_file = 1;        
        else if(!strcmp(argv[i], "-o"))
            flag_opt = 1;
        else if(!strcmp(argv[i], "-k"))
            flag_cross = 1;
        else if(!strcmp(argv[i], "-m"))
            flag_alg_mode = 1;
        else if(!strcmp(argv[i], "-d"))
            flag_device = 1;
        else if(!strcmp(argv[i], "-v"))
            flag_verbose = 1;
        else if(!strcmp(argv[i], "--help"))
            print_help();
        else if(!strcmp(argv[i], "--version"))
            print_version();
        else if(!strcmp(argv[i], "--infoDevice"))
            infoDev();
        else{
            printf("The number of options is incorrect. For more information run the execution as: %s --help\n",argv[0]);
            exit(-1);
        }
    }
    /* ******************************************* ARGUMENT PARSING ******************************************* */

    int suppr_verbose = 1;
    if( strcmp(task,"CLASSIFICATION") == 0 ){

        /* ***** VARIABLE WORKSPACE FOR CLASSIFICATION TASK ***** */
        unsigned long long int dataBytes = 2 * dataSize * window_size * n_feat * sizeof(float);
        int * dataLabels = (int *) malloc(dataSize * sizeof(int));
        float* data = (float*) malloc (dataBytes);
        int trainSize, testSize;
        float mean_RIA = 0.0f, mean_RA = 0.0f, mean_ER_RIA = 0.0f, mean_ER_RA = 0.0f;

        printf("Reading data...\n");
        readFile(argv, arr_num_file, n_file, read_mode, data, data_struct, window_size, dataLabels, n_feat, class_mode);

        if( k_fold < 1) //not doing K-cross validation
            k_fold = 1; // (work around to do not re-write a lot of code)
        else
            tInd = crossvalind_Kfold(dataLabels, dataSize, k_fold, flag_shuffle);


        //Setting all the variables for each k-th fold
        for (f = 0; f < k_fold; f++){
            err = 0;
            errNR = 0;

            if (k_fold > 1) { //doing K-fold cross validation
                testSize = countVal(tInd,dataSize,f);
                trainSize = dataSize - testSize;
            }
            else {
                trainSize = data_struct.train_size;
                testSize = data_struct.test_size;
            }

            /******************************* HOST MEMORY ALLOCATION *******************************/  
            unsigned long long int testBytes = testSize * window_size * n_feat * sizeof(float);
            unsigned long long int trainBytes;

            if(class_mode < 2)
                trainBytes = trainSize * window_size * n_feat * sizeof(float);
            else
                trainBytes = 2 * trainSize * window_size * n_feat * sizeof(float);
            

            float* h_train = (float*) malloc (trainBytes);
            float* h_test = (float*) malloc (testBytes);

            int * trainLabels = (int *) malloc(trainSize*sizeof(int));
            int * testLabels = (int *) malloc(testSize*sizeof(int));


            createTrainingTestingSet(data, dataLabels, dataSize, window_size, n_feat, h_train, trainLabels, trainSize, h_test, testLabels, testSize, tInd, f, class_mode);
            /******************************* HOST MEMORY ALLOCATION *******************************/

            printf("\n****************Classification w/ %s-%s using %s****************\n\n" ,strategy, distance_type, compution_type);
            printf("TRAINING SET:\tlength: %d, n_attrs: %d byte_size: %f\n",trainSize,n_feat,(float)trainBytes);
            printf("TESTING SET:\tlength: %d, n_attrs: %d, byte_size: %f\n\n",testSize,n_feat,(float)testBytes);

            minI = (int *) malloc(sizeof(int));

            if( strcmp(compution_type,"CPU") == 0) {

                float dtw_curr = 0;
                float cum_sum = 0;
                float * h_Out;

                switch (class_mode){

                    case 0:

                        h_Out = (float *) malloc(trainSize*sizeof(float));

                        gettimeofday(&start_CPU, NULL);

                        for (int k = 0; k < testSize; k++) {
                            for (int j = 0; j < trainSize; j++) {
                                if ( strcmp(distance_type,"DTW") == 0) //DTW distance
                                    h_Out[j] = short_md_dtw_c(&h_train[j*n_feat*window_size], &h_test[k*n_feat*window_size], window_size, window_size, n_feat, window_size);
                                else //Euclidean Distance
                                    h_Out[j] = short_md_ed_c(&h_train[j*n_feat*window_size], &h_test[k*n_feat*window_size], window_size, n_feat, window_size);    
                            }
                            min = min_arr(h_Out,trainSize,minI);

                            if( trainLabels[*minI] != testLabels[k] )
                                err++;

                            if(verbose_mode > 0 && verbose_mode < testSize){
                                if(k % verbose_mode == 0)
                                    printf("%d\t gt: %d\t\tRI: %d\t%3.6f\n",k , testLabels[k] , trainLabels[*minI], min);
                                else if(k==testSize)
                                    printf("%d\t gt: %d\t\tRI: %d\t%3.6f\n",k , testLabels[k] , trainLabels[*minI], min);
                            }else if(suppr_verbose == 1){
                                printf("The number of iteration is greater than testSize! Verbose mode will be suppressed for this run\n");
                                suppr_verbose = 0;
                            }
                        }
                        gettimeofday(&stop_CPU, NULL);

                        elapsed = timedifference_msec(start_CPU, stop_CPU);
                        printf("\nExecution time for %s w/ CPU using %s-%s:  %f ms\n" ,task, strategy, distance_type, elapsed);

                        RA = (float)(testSize-err)*(100.0/testSize);
                        ER_RA = (float)(testSize-(testSize-err))/(testSize);
                        printf("Regular Accuracy is %f\n",RA);
                        printf("The Error rate is %f\n",ER_RA);
                        mean_RA += RA;
                        mean_ER_RA += ER_RA;

                        free(h_train);
                        free(h_test);
                        free(h_Out);
                        printf("Memory deallocated!\n");
                    break;

                    case 1:

                        h_Out = (float*)malloc(trainSize*window_size*sizeof(float));

                        gettimeofday(&start_CPU, NULL);

                        for (int k = 0; k < testSize; k++) {
                            for (j = 0; j < trainSize; j++) {
                                cum_sum = 0.0;
                                for (int d = 0; d < n_feat; d++) {
                                    if ( strcmp(distance_type,"DTW") == 0) //DTW distance
                                        dtw_curr = short_dtw_c(&h_train[(d*window_size)+(j*n_feat*window_size)],&h_test[(k*n_feat*window_size)+(d*window_size)],window_size,window_size);
                                    else
                                        dtw_curr = short_ed_c(&h_train[(d*window_size)+(j*n_feat*window_size)],&h_test[(k*n_feat*window_size)+(d*window_size)],window_size);
                                    cum_sum += dtw_curr;
                                }
                                h_Out[j] = cum_sum;
                            }
                            min = min_arr(h_Out,trainSize,minI);

                            if( trainLabels[*minI] != testLabels[k] )
                                err++;

                            if(verbose_mode > 0 && verbose_mode < testSize){
                                if(k % verbose_mode == 0)
                                    printf("%d\t gt: %d\t\tRI: %d\t%3.6f\n",k , testLabels[k] , trainLabels[*minI], min);
                                else if(k==testSize)
                                    printf("%d\t gt: %d\t\tRI: %d\t%3.6f\n",k , testLabels[k] , trainLabels[*minI], min);
                            }else if(suppr_verbose == 1){
                                printf("The number of iteration is greater than testSize! Verbose mode will be suppressed for this run\n");
                                suppr_verbose = 0;
                            }
                        }
                        gettimeofday(&stop_CPU, NULL);

                        elapsed = timedifference_msec(start_CPU, stop_CPU);
                        printf("\nExecution time for %s w/ CPU using %s-%s:  %f ms\n" ,task, strategy, distance_type, elapsed);

                        RA = (float)(testSize-err)*(100.0/testSize);
                        ER_RA = (float)(testSize-(testSize-err))/(testSize);
                        printf("Regular Accuracy is %f\n",RA);
                        printf("The Error rate is %f\n",ER_RA);
                        mean_RA += RA;
                        mean_ER_RA += ER_RA;

                        free(h_train);
                        free(h_test);
                        free(h_Out);
                        printf("Memory deallocated!\n");
                    break;

                    case 2: //TODO: implement MD_RDTW

                    h_Out = (float *) malloc(window_size*trainSize*sizeof(float));

                    gettimeofday(&start_CPU, NULL);

                    for (int i = 0; i < testSize; i++) {
                        for (int j = 0; j < trainSize; j++) {
                            for (int k = 0; k < window_size; k++) {
                                if ( strcmp(distance_type,"DTW") == 0) //DTW distance
                                    h_Out[(j*window_size)+k] = short_md_dtw_c(&h_train[(2*j*n_feat*window_size)+k],&h_test[i*n_feat*window_size],window_size,window_size,n_feat,2*window_size);
                                else
                                    h_Out[(j*window_size)+k] = short_md_ed_c(&h_train[(2*j*n_feat*window_size)+k],&h_test[i*n_feat*window_size],window_size,n_feat,2*window_size);
                            }
                        }
                        min = 9999999999.99;

                        *minI = -1;
                        minINR = -1;
                        minNR = 99999999999.99;
                        for(int m = 0 ; m < trainSize ; m++ )
                        {
                            if (h_Out[m*window_size] < minNR )
                            {
                                minNR = h_Out[m*window_size];
                                minINR = m;
                            }
                            for(int p = 0 ; p < window_size ; p++ )
                            {
                                int t = m*window_size+p;

                                if ( h_Out[t] < min )
                                {
                                    min = h_Out[t];
                                    *minI = m;
                                }
                            }
                        }

                        if( trainLabels[*minI] != testLabels[i] )
                            err++;

                        if( trainLabels[minINR] != testLabels[i] )
                            errNR++;

                        if(verbose_mode > 0 && verbose_mode < testSize){
                            if(i % verbose_mode == 0)
                                printf("%d\t gt: %d\t\tRI: %d\t%3.6f \t\t NRI: %d\t%3.6f\n",i , testLabels[i] , trainLabels[*minI], min, trainLabels[minINR], minNR );
                            else if(i==testSize)
                                printf("%d\t gt: %d\t\tRI: %d\t%3.6f \t\t NRI: %d\t%3.6f\n",i , testLabels[i] , trainLabels[*minI], min, trainLabels[minINR], minNR );
                        }else if(suppr_verbose == 1){
                            printf("The number of iteration is greater than testSize! Verbose mode will be suppressed for this run\n");
                            suppr_verbose = 0;
                        }
                    }
                    gettimeofday(&stop_CPU, NULL);

                    elapsed = timedifference_msec(start_CPU, stop_CPU);
                    printf("\nExecution time for %s w/ CPU using %s-%s:  %f ms\n" ,task, strategy, distance_type, elapsed);

                    RIA = (float)(testSize-err)*(100.0/testSize);
                    ER_RIA = (float)(testSize-(testSize-err))/(testSize);
                    printf("Rotation Invariant Accuracy is %f\n",RIA);
                    printf("The Error rate of RI is %f\n",ER_RIA);
                    mean_RIA += RIA;
                    mean_ER_RIA += ER_RIA;

                    RA = (float)(testSize-errNR)*(100.0/testSize);
                    ER_RA = (float)(testSize-(testSize-errNR))/(testSize);
                    printf("\nRegular Accuracy is %f\n",RA);
                    printf("The Error rate of NR is %f\n",ER_RA);

                    break;
                }
            }
            else if( strcmp(compution_type,"GPU") == 0) {

                cudaSetDevice(device);

                cudaDeviceProp deviceProp;
                deviceProp = getDevProp(device);
                printf("\nDevice selected: %s\n",deviceProp.name);


                /******************************* DEVICE MEMORY ALLOCATION *******************************/
                float* d_train = 0;
                cudaMalloc((void**)&d_train, trainBytes);
                cudaMemcpy(d_train, h_train, trainBytes, cudaMemcpyHostToDevice);

                float* d_test = 0;
                float* h_Out = 0;
                cudaMalloc((void**)&d_test, n_feat*window_size*sizeof(float));

                float* d_Out = 0;
                if(class_mode < 2)
                {
                    cudaMalloc((void**)&d_Out, trainSize*sizeof(float));
                    cudaMemset(&d_Out,0,trainSize*sizeof(float));
                    h_Out = (float*)malloc(trainSize*sizeof(float));
                    memset(h_Out,0,trainSize*sizeof(float));            
                }
                else 
                {
                    cudaMalloc((void**)&d_Out, trainSize*window_size*sizeof(float));
                    cudaMemset(&d_Out,0,trainSize*window_size*sizeof(float));
                    h_Out = (float*)malloc(trainSize*window_size*sizeof(float));
                    memset(h_Out,0,trainSize*window_size*sizeof(float));            
                }


                float grid_size;
                dim3 grid;
                dim3 threads;
                /******************************* DEVICE MEMORY ALLOCATION *******************************/

                switch (class_mode)
                {
                    case 0 ://MD_DTW_D
                    {
                        T2 = (n_feat*window_size)*sizeof(float);

                        if (T2 > deviceProp.sharedMemPerBlock) {

                            printf("The T2 test timeserie: %f doesn't fit into the shared memory: %lu, so it will be allocated into the global memory\n",T2,deviceProp.sharedMemPerBlock);
                            gm = 1;
                            T2 = 0;
                        }
                        else
                            gm = 0;

                        grid_size = ceil((float)trainSize/blockSize);

                        // number of blocks (x,y) for a grid
                        grid.x = grid_size;
                        grid.y = 1;
                        // number of threads (x,y) for each block
                        threads.x = blockSize;
                        threads.y = 1;

                        printf("Grid_size_x: %d, number_of_threads_x: %d \n", grid.x,threads.x);
                        printf("Grid_size_y: %d, number_of_threads_y: %d \n\n", grid.y,threads.y);

                        cudaEventCreate(&start_GPU);
                        cudaEventCreate(&stop_GPU);
                        cudaEventRecord(start_GPU,0);

                        for (k = 0; k < testSize; k++){
                            cudaMemset(&d_test, 0, n_feat*window_size*sizeof(float));
                            cudaMemcpy(d_test, h_test + k*(n_feat*window_size) , n_feat*window_size*sizeof(float), cudaMemcpyHostToDevice);

                            if ( strcmp(distance_type,"DTW") == 0) //DTW distance
                                MD_DTW_D <<<grid, threads, T2>>> (d_train, d_test, window_size, window_size,n_feat,d_Out,trainSize,0,gm);
                            else
                                MD_ED_D <<<grid, threads, T2>>> (d_train, d_test, window_size,n_feat,d_Out,trainSize,0, gm);

                            cudaDeviceSynchronize(); //it may be avoided if there's not printf in the kernel function
                            cudaMemcpy(h_Out, d_Out, trainSize*sizeof(float), cudaMemcpyDeviceToHost);

                            min = min_arr(h_Out,trainSize,minI);

                            if( trainLabels[*minI] != testLabels[k] )
                                err++;

                            if(verbose_mode > 0 && verbose_mode < testSize){
                                if(k % verbose_mode == 0)
                                    printf("%d\t gt: %d\t\tRI: %d\t%3.6f\n",k , testLabels[k] , trainLabels[*minI], min);
                                else if(k==testSize)
                                    printf("%d\t gt: %d\t\tRI: %d\t%3.6f\n",k , testLabels[k] , trainLabels[*minI], min);
                            }else if(suppr_verbose == 1){
                                printf("The number of iteration is greater than testSize! Verbose mode will be suppressed for this run\n");
                                suppr_verbose = 0;
                            }
                        }

                        cudaEventRecord(stop_GPU,0);
                        cudaEventSynchronize(stop_GPU);
                        cudaEventElapsedTime(&time_GPU_MD_DTW_D,start_GPU,stop_GPU);
                        cudaEventDestroy(start_GPU);
                        cudaEventDestroy(stop_GPU);
                        printf("\nExecution time for %s w/ GPU using %s-%s:  %f ms\n" ,task, strategy, distance_type, time_GPU_MD_DTW_D);

                        RA = (float)(testSize-err)*(100.0/testSize);
                        ER_RA = (float)(testSize-(testSize-err))/(testSize);
                        printf("Regular Accuracy is %f\n",RA);
                        printf("The Error rate is %f\n",ER_RA);
                        mean_RA += RA;
                        mean_ER_RA += ER_RA;

                        free(h_train);
                        free(h_test);
                        free(h_Out);
                        printf("Memory deallocated!\n");
                    }
                    break;
                    case 1: //MD_DTW_I
                    {
                        printf("DTW_INDEPENDENT\n");

                        grid_size = ceil((float)(trainSize*n_feat)/blockSize);
                        float dim_row = floor((float)blockSize/n_feat);
                        float dim_col = n_feat;    

                        // number of blocks (x,y) for a grid
                        grid.x = grid_size;
                        grid.y = 1;
                        // number of threads (x,y) for each block
                        threads.x = dim_row;
                        threads.y = dim_col;


                        printf("Grid_size_x: %d, number_of_threads_x: %d \n", grid.x,threads.x);
                        printf("Grid_size_y: %d, number_of_threads_y: %d \n\n", grid.y,threads.y);

                        float sh_mem = ((threads.x*threads.y)+(n_feat*window_size))*sizeof(float);

                        cudaEventCreate(&start_GPU);
                        cudaEventCreate(&stop_GPU);
                        cudaEventRecord(start_GPU,0);

                        for (k = 0; k < testSize; k++){
                            cudaMemcpy(d_test, h_test + k*(n_feat*window_size) , n_feat*window_size*sizeof(float), cudaMemcpyHostToDevice);

                            if ( strcmp(distance_type,"DTW") == 0) //DTW distance
                                MD_DTW_I <<<grid, threads, sh_mem>>> (d_train, d_test, window_size, window_size,n_feat,d_Out,trainSize,0);
                            else
                                MD_ED_I <<<grid, threads, sh_mem>>> (d_train, d_test, window_size, n_feat,d_Out,trainSize,0);

                            cudaThreadSynchronize();
                            cudaMemcpy(h_Out, d_Out, trainSize*sizeof(float), cudaMemcpyDeviceToHost);

                            min = min_arr(h_Out,trainSize,minI);

                            if( trainLabels[*minI] != testLabels[k] )
                                err++;

                            if(verbose_mode > 0 && verbose_mode < testSize){
                                if(k % verbose_mode == 0)
                                    printf("%d\t gt: %d\t\tRI: %d\t%3.6f\n",k , testLabels[k] , trainLabels[*minI], min);
                                else if(k==testSize)
                                    printf("%d\t gt: %d\t\tRI: %d\t%3.6f\n",k , testLabels[k] , trainLabels[*minI], min);
                            }else if(suppr_verbose == 1){
                                printf("The number of iteration is greater than testSize! Verbose mode will be suppressed for this run\n");
                                suppr_verbose = 0;
                            }
                        }

                        cudaEventRecord(stop_GPU,0);
                        cudaEventSynchronize(stop_GPU);
                        cudaEventElapsedTime(&time_GPU_MD_DTW_I,start_GPU,stop_GPU);
                        cudaEventDestroy(start_GPU);
                        cudaEventDestroy(stop_GPU); 
                        printf("\nExecution time for %s w/ GPU using %s-%s:  %f ms\n" ,task, strategy, distance_type, time_GPU_MD_DTW_I);

                        RA = (float)(testSize-err)*(100.0/testSize);
                        ER_RA = (float)(testSize-(testSize-err))/(testSize);
                        printf("Regular Accuracy is %f\n",RA);
                        printf("The Error rate is %f\n",ER_RA);
                        mean_RA += RA;
                        mean_ER_RA += ER_RA;
                        free(h_train);
                        free(h_test);
                        free(h_Out);
                        printf("Memory deallocated!\n");
                    }
                    break;
                    case 2:
                    {

                        T2 = (n_feat*window_size)*sizeof(float);

                        if (T2 > deviceProp.sharedMemPerBlock) {

                            printf("The T2 test timeserie: %f doesn't fit into the shared memory: %lu, so it will be allocated into the global memory\n",T2,deviceProp.sharedMemPerBlock);
                            gm = 1;
                            T2 = 0;
                        }
                        else
                            gm = 0;

                        grid_size=ceil((float)trainSize*window_size/blockSize);

                        // number of blocks (x,y) for a grid
                        grid.x = grid_size;
                        grid.y = 1;
                        // number of threads (x,y) for each block
                        threads.x = blockSize;
                        threads.y = 1;

                        printf("Grid_size_x: %d, number_of_threads_x: %d \n", grid.x,threads.x);
                        printf("Grid_size_y: %d, number_of_threads_y: %d \n\n", grid.y,threads.y);                

                        cudaEventCreate(&start_GPU);
                        cudaEventCreate(&stop_GPU);
                        cudaEventRecord(start_GPU,0);

                        for(k = 0;k < testSize; k++) 
                        {
                            cudaMemcpy(d_test, h_test + (k*n_feat*window_size), n_feat*window_size*sizeof(float), cudaMemcpyHostToDevice);

                            if ( strcmp(distance_type,"DTW") == 0) //DTW distance                            
                                rMD_DTW_D <<<grid, threads, T2>>> (d_train, d_test, window_size, window_size,n_feat,d_Out,trainSize,gm);
                            else
                                rMD_ED_D <<<grid, threads, T2>>> (d_train, d_test, window_size, n_feat,d_Out,trainSize, gm);

                            cudaThreadSynchronize();

                            cudaMemcpy(h_Out, d_Out, trainSize*window_size*sizeof(float), cudaMemcpyDeviceToHost);

                            min = 9999999999.99;

                            *minI = -1;
                            minINR = -1; 
                            minNR = 99999999999.99;
                            i = 0;
                            for(j = 0 ; j < trainSize ; j++ ){
                                if (h_Out[j*window_size] < minNR ){
                                    minNR = h_Out[j*window_size];
                                    minINR = j;
                                }
                                for( i = 0 ; i < window_size ; i++ ){
                                    int t = j*window_size+i;
                                    if ( h_Out[t] < min ){
                                        min = h_Out[t];
                                        *minI = j;
                                    }
                                }
                            }
                            if( trainLabels[*minI] != testLabels[k] )
                                err++;

                            if( trainLabels[minINR] != testLabels[k] )
                                errNR++;

                            if(verbose_mode > 0 && verbose_mode < testSize){
                                if(i % verbose_mode == 0)
                                    printf("%d\t gt: %d\t\tRI: %d\t%3.6f \t\t NRI: %d\t%3.6f\n",i , testLabels[i] , trainLabels[*minI], min, trainLabels[minINR], minNR );
                                else if(i==testSize)
                                    printf("%d\t gt: %d\t\tRI: %d\t%3.6f \t\t NRI: %d\t%3.6f\n",i , testLabels[i] , trainLabels[*minI], min, trainLabels[minINR], minNR );
                            }else if(suppr_verbose == 1){
                                printf("The number of iteration is greater than testSize! Verbose mode will be suppressed for this run\n");
                                suppr_verbose = 0;
                            }
                        }

                        cudaEventRecord(stop_GPU,0);
                        cudaEventSynchronize(stop_GPU);
                        cudaEventElapsedTime(&time_GPU_rMDTW,start_GPU,stop_GPU);
                        cudaEventDestroy(start_GPU);
                        cudaEventDestroy(stop_GPU);
                        printf("\nExecution time for %s w/ GPU using %s-%s:  %f ms\n" ,task, strategy, distance_type, time_GPU_rMDTW);
                    
                        RIA = (float)(testSize-err)*(100.0/testSize);
                        ER_RIA = (float)(testSize-(testSize-err))/(testSize);
                        printf("Rotation Invariant Accuracy is %f\n",RIA);
                        printf("The Error rate of RI is %f\n",ER_RIA);
                        mean_RIA += RIA;
                        mean_ER_RIA += ER_RIA;

                        RA = (float)(testSize-errNR)*(100.0/testSize);
                        ER_RA = (float)(testSize-(testSize-errNR))/(testSize);
                        printf("\nRegular Accuracy is %f\n",RA);
                        printf("The Error rate of NR is %f\n",ER_RA);
                        mean_RA += RA;
                        mean_ER_RA += ER_RA;

                        free(h_train);
                        free(h_test);
                        break;
                        default:
                            printf("Error algorithm choice\n");
                    }
                    break;
                }
            }
        }
        if (class_mode < 2){
            mean_RA /= k_fold;
            mean_ER_RA /= k_fold;
            printf("\nRegular Accuracy mean is %f\n", mean_RA);
            printf("The Error rate mean of NR is %f\n", mean_ER_RA);
        }
        else {
            mean_RIA /= k_fold;
            mean_ER_RIA /= k_fold;
            printf("\nRotation Invariant Accuracy mean is %f\n", mean_RIA);
            printf("The Error rate mean of RI is %f\n", mean_ER_RIA);
        }        
    }
    else if( strcmp(task,"SUBSEQ_SEARCH") == 0 ){

        nss = t_size - q_size + 1;
        window_size = q_size;

        unsigned long long int t_bytes = t_size * n_feat * sizeof(float);
        unsigned long long int q_bytes = q_size * n_feat * sizeof(float);

        ////////////////////////////////CPU MEMORY ALLOCATION////////////////////////////////
        float *t_series = (float *) malloc (t_bytes);
        float *q_series = (float *) malloc (q_bytes);
        float* owp = (float*) malloc (nss*sizeof(float));
        memset(owp,0,nss*sizeof(float));

        readFileSubSeq(argv, arr_num_file, n_file, t_series, t_size, q_series, window_size, n_feat, read_mode);

        int *ind_min_val = (int*) malloc (sizeof(int));

        if( strcmp(compution_type,"CPU") == 0 ){
            printf("\n****************Subsequence Search with %s****************\n\n",compution_type);
            printf("Time Series T:\tlength: %d, n_feat: %d byte_size: %lu\n", t_size, n_feat, sizeof(float)*t_size);
            printf("Time Series Q:\t:length: %d, n_feat: %d, byte_size: %lu\n", q_size, n_feat, sizeof(float)*q_size);
            printf("Number of Subsequences to search: %d\n", nss);            

            float dist, min = 0.0, val_curr;

            switch (class_mode){

                case 0: {
                    printf("%s-%s %s version processing...\n" ,distance_type,strategy,compution_type);

                    gettimeofday(&start_CPU, NULL);

                    for (int i = 0; i < nss; i++) {

                        dist = 0.0;
                        if ( strcmp(distance_type,"DTW") == 0) //DTW distance
                            dist = short_md_dtw_c(&t_series[i], q_series, window_size, window_size, n_feat, t_size);
                        else
                            dist = short_md_ed_c(&t_series[i], q_series, window_size, n_feat, t_size);

                        owp[i] = dist;

                        if(verbose_mode > 0 && verbose_mode < nss){
                            if(i % verbose_mode == 0)
                                printf("curr val diff. [%d]: %f\n", i, owp[i]);
                            else if(i==nss)
                                printf("curr val diff. [%d]: %f\n", i, owp[i]);
                        }else if(suppr_verbose == 1){
                            printf("The number of iteration is greater than the number of subsequences! Verbose mode will be suppressed for this run\n");
                            suppr_verbose = 0;
                        }

                    }
                    gettimeofday(&stop_CPU, NULL);

                    elapsed = timedifference_msec(start_CPU, stop_CPU);
                    printf("\nExecution time for %s w/ CPU using %s-%s:  %f ms\n" ,task, strategy, distance_type, elapsed);

                    //computing minimum value
                    min = min_arr(owp,nss,ind_min_val);
                    printf("CPU version w/ min.index value %d, min. value: %f\n\n" ,*ind_min_val,min);
                }

                break;

                case 1: {

                    printf("%s-%s %s version processing...\n" ,distance_type,strategy,compution_type);

                    gettimeofday(&start_CPU, NULL);

                    for (int i = 0; i < nss; i++) {
                        dist = 0.0;
                        for (int k = 0; k < n_feat; k++) {
                            if ( strcmp(distance_type,"DTW") == 0) //DTW distance
                                val_curr = short_dtw_c(&t_series[(k*t_size)+i],&q_series[(k*window_size)],window_size,window_size);
                            else
                                val_curr = short_ed_c(&t_series[(k*t_size)+i],&q_series[(k*window_size)],window_size);

                            dist += val_curr;
                        }

                        owp[i] = dist;

                        if(verbose_mode > 0 && verbose_mode < nss){
                            if(i % verbose_mode == 0)
                                printf("curr val diff. [%d]: %f\n", i, owp[i]);
                            else if(i==nss)
                                printf("curr val diff. [%d]: %f\n", i, owp[i]);
                        }else if(suppr_verbose == 1){
                            printf("The number of iteration is greater than the number of subsequences! Verbose mode will be suppressed for this run\n");
                            suppr_verbose = 0;
                        }
                    }
                    gettimeofday(&stop_CPU, NULL);

                    elapsed = timedifference_msec(start_CPU, stop_CPU);
                    printf("\nExecution time for %s w/ CPU using %s-%s:  %f ms\n" ,task, strategy, distance_type, elapsed);

                    //computing minimum value
                    min = min_arr(owp,nss,ind_min_val);
                    printf("CPU version w/ min.index value %d, min. value: %f\n\n" ,*ind_min_val,min);
                }
                break;

                default:
                    printf("Error algorithm choice\n");
            }
        }
        else{ //GPU computation


            /**********************PARAMETERS DECLARATION **********************/
            float grid_size;
            dim3 grid;
            dim3 threads;

            ////////////////////////////////DTW GPU_GM ALGORITHM////////////////////////////////
            float *d_t_series = 0, *d_owp=0, *d_q_series=0;
            cudaMalloc((void**)&d_t_series, t_bytes);
            cudaMemcpy(d_t_series, t_series, t_bytes, cudaMemcpyHostToDevice);

            cudaMalloc((void**)&d_q_series, q_bytes);
            cudaMemcpy(d_q_series, q_series, q_bytes, cudaMemcpyHostToDevice);

            cudaMalloc((void**)&d_owp,nss * sizeof(float));
            cudaMemset(d_owp, 0, nss * sizeof(float));


            cudaSetDevice(device);

            cudaDeviceProp deviceProp;
            deviceProp = getDevProp(device);
            printf("\nDevice selected: %s\n",deviceProp.name);

            printf("\n****************Subsequence Search with %s****************\n\n",compution_type);
            printf("Time Series T:\tlength: %d, n_feat: %d byte_size: %lu\n", t_size, n_feat, sizeof(float)*t_size);
            printf("Time Series Q:\t:length: %d, n_feat: %d, byte_size: %lu\n", q_size, n_feat, sizeof(float)*q_size);
            printf("Number of Subsequences to search: %d\n", nss);

            switch (class_mode){

                case 0: {

                    //Setting CUDA variables and structure
                    grid_size = ceil((double)nss/blockSize);

                    // number of blocks (x,y) for a grid
                    grid.x = grid_size;
                    grid.y = 1;

                    // number of threads (x,y) for each block                    
                    threads.x = blockSize;
                    threads.y = 1;

                    T2 = (n_feat*window_size)*sizeof(float);

                    if (T2 > deviceProp.sharedMemPerBlock) {

                        printf("The T2 test timeserie: %f doesn't fit into the shared memory: %lu, so it will be allocated into the global memory\n",T2,deviceProp.sharedMemPerBlock);
                        gm = 1;
                        T2 = 0;
                    }
                    else
                        gm = 0;

                    printf("Grid_size_x: %d, number_of_threads_x: %d \n", grid.x,threads.x);
                    printf("Grid_size_y: %d, number_of_threads_y: %d \n\n", grid.y,threads.y);

                    printf("%s-%s %s version processing...\n" ,distance_type,strategy,compution_type);
                    cudaEventCreate(&start_GPU);
                    cudaEventCreate(&stop_GPU);
                    cudaEventRecord(start_GPU,0);

                    if ( strcmp(distance_type,"DTW") == 0) //DTW distance
                        MD_DTW_D <<<grid, threads, T2>>> (d_t_series, d_q_series, window_size, window_size, n_feat, d_owp, t_size, 1, gm);
                    else
                        MD_ED_D <<<grid, threads, T2>>> (d_t_series, d_q_series, window_size, n_feat, d_owp, t_size, 1, gm);

                    cudaEventRecord(stop_GPU,0);
                    cudaEventSynchronize(stop_GPU);
                    cudaEventElapsedTime(&time_GPU_MD_DTW_D,start_GPU,stop_GPU);
                    cudaEventDestroy(start_GPU);
                    cudaEventDestroy(stop_GPU);
                    printf("\nExecution time for %s w/ MD_DTW_D with GPU %f ms\n", task, time_GPU_MD_DTW_D);

                    cudaMemcpy(owp, d_owp, nss * sizeof(float), cudaMemcpyDeviceToHost);

                    for (int i = 0; i < nss; ++i)
                    {
                        if(verbose_mode > 0 && verbose_mode < nss){
                            if(i % verbose_mode == 0)
                                printf("curr val diff. [%d]: %f\n", i, owp[i]);
                            else if(i==nss)
                                printf("curr val diff. [%d]: %f\n", i, owp[i]);
                        }else if(suppr_verbose == 1){
                            printf("The number of iteration is greater than the number of subsequences! Verbose mode will be suppressed for this run\n");
                            suppr_verbose = 0;
                        }
                    }

                    min = min_arr(owp, nss, ind_min_val);
                    printf("GPU_GM version w/ min.index value %d, min. value: %f\n\n" ,*ind_min_val,min);
                }
                break;

                case 1:{

                    //Setting CUDA variables and structure
                    grid_size = ceil((float)(nss*n_feat)/blockSize);
                    float dim_row = floor((float)blockSize/n_feat);
                    float dim_col = n_feat;    

                    // number of blocks (x,y) for a grid
                    grid.x = grid_size;
                    grid.y = 1;

                    // number of threads (x,y) for each block
                    threads.x = dim_row;
                    threads.y = dim_col;

                    printf("Grid_size_x: %d, number_of_threads_x: %d \n", grid.x,threads.x);
                    printf("Grid_size_y: %d, number_of_threads_y: %d \n\n", grid.y,threads.y);

                    printf("%s-%s %s version processing...\n" ,distance_type,strategy,compution_type);
                    cudaEventCreate(&start_GPU);
                    cudaEventCreate(&stop_GPU);
                    cudaEventRecord(start_GPU,0);

                    float sh_mem = ((threads.x*threads.y) + (n_feat*t_size))*sizeof(float);
                    if ( strcmp(distance_type,"DTW") == 0) //DTW distance
                        MD_DTW_I <<<grid, threads, sh_mem>>> (d_t_series, d_q_series, window_size, window_size,n_feat,d_owp,t_size, 1);
                    else
                        MD_ED_I <<<grid, threads, sh_mem>>> (d_t_series, d_q_series, window_size, n_feat, d_owp, t_size, 1);

                    cudaThreadSynchronize();
                    cudaEventRecord(stop_GPU,0);
                    cudaEventSynchronize(stop_GPU);
                    cudaEventElapsedTime(&time_GPU_MD_DTW_I,start_GPU,stop_GPU);
                    cudaEventDestroy(start_GPU);
                    cudaEventDestroy(stop_GPU);
                    printf("\nExecution time for %s w/ MD_DTW_I with GPU %f ms\n", task, time_GPU_MD_DTW_I);

                    cudaMemcpy(owp, d_owp, nss * sizeof(float), cudaMemcpyDeviceToHost);

                    for (int i = 0; i < nss; ++i)
                    {
                        if(verbose_mode > 0 && verbose_mode < nss){
                            if(i % verbose_mode == 0)
                                printf("curr val diff. [%d]: %f\n", i, owp[i]);
                            else if(i==nss)
                                printf("curr val diff. [%d]: %f\n", i, owp[i]);
                        }else if(suppr_verbose == 1){
                            printf("The number of iteration is greater than the number of subsequences! Verbose mode will be suppressed for this run\n");
                            suppr_verbose = 0;
                        }
                    }

                    min = min_arr(owp, nss, ind_min_val);
                    printf("GPU_GM version w/ min.index value %d, min. value: %f\n\n" ,*ind_min_val,min);
                }
                break;

                default:
                    printf("Error algorithm choice\n");
            }
            cudaFree(d_t_series);
            cudaFree(d_q_series);
            cudaFree(d_owp);
        }
        free(t_series);
        free(q_series);
        free(owp);
        printf("\nMemory deallocated!\n\n");

    }
    return 0;
}