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
    int window_size = 0; //i.e. 315
    int dataSize = 0; // i.e. 440
    int train_size, test_size;
    int blockSize = 0; // i.e. 512    
    int k_fold = 0; //input data to set
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


    /*
     * argument parsing
     */
    // printf("argc: %d\n",argc);
    // printf("argv[0]: %s\n",argv[0]);
    for(i = 1; i < argc; i++) {
        // printf("argv[%d]: %s\n",i,argv[i]);

        if(flag_task){
            num_opts = 1;
            task = argv[i];
            // printf("TASK: %c\n",task[0]);

            // Upper string
            j=0;
            do { task[j] = toupper(task[j]);
            } while(task[j++]);


            if(!checkFlagOpts(argv,argc,i,num_opts)) {
                printf("The number of options is incorrect. For more information run the execution as: %s --help\n",argv[0]);
                exit(-1);
            }

            flag_task = 0;

        }
        else if(flag_in)
        {
            compution_type = argv[i];

            // Upper string
            j=0;
            do { compution_type[j] = toupper(compution_type[j]);
            } while(compution_type[j++]);
            
            if (strcmp(compution_type,"GPU") == 0){
                num_opts = 3;

                // printf("i_flag_in: %d, argc: %d\n",i,argc);

                if(!checkFlagOpts(argv,argc,i,num_opts)) {
                    printf("The number of options is incorrect. For more information run the execution as: %s --help\n",argv[0]);
                    exit(-1);
                }

                n_feat = atoi(argv[i+1]);
                blockSize = atoi(argv[i+2]);
                read_mode = atoi(argv[i+3]);
                i = i + 3;
            }
            else if( (strcmp(compution_type,"CPU") == 0) ) {
                num_opts = 2;

                // printf("i_flag_in: %d, argc: %d\n",i,argc);

                if(!checkFlagOpts(argv,argc,i,num_opts)) {
                    printf("The number of options is incorrect. For more information run the execution as: %s --help\n",argv[0]);
                    exit(-1);
                }

                n_feat = atoi(argv[i+1]);
                read_mode = atoi(argv[i+2]);
                i = i + 2;
            }
            else {
                printf("The number of options is incorrect. For more information run the execution as: %s --help\n", argv[0]);
                exit(-1);
            }

            flag_in = 0;
        }
        else if(flag_file){

            if(strcmp(task,"CLASSIFICATION") == 0){

                if (read_mode == 0 || read_mode == 2){
                    n_file = 2; //[0,1] data and label descriptors
                    num_opts = n_file;

                    // printf("i_flag_file: %d, argc: %d\n",i,argc);

                    if(!checkFlagOpts(argv,argc,i,num_opts)) {
                        printf("The number of options is incorrect. For more information run the execution as: %s --help\n",argv[0]);
                        exit(-1);
                    }
                }
                else if (read_mode == 1){
                    n_file = n_feat;
                    num_opts = n_feat;

                    // printf("i_flag_file: %d, argc: %d\n",i,argc);

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
            else { //SUB_SEQ_SEARCH TASK
                n_file = 2;
                num_opts = n_file;
                // printf("i_flag_file: %d, argc: %d\n",i,argc);

                if(!checkFlagOpts(argv,argc,i,num_opts)) {
                    printf("The number of options is incorrect. For more information run the execution as: %s --help\n",argv[0]);
                    exit(-1);
                }
            }


            if ( blockSize > 1024 || blockSize < 0 )
                printf("Irregular number of threads for block\n");

            // printf("n_feat: %d\n",n_feat);
            // printf("n_file: %d\n",n_file);
            // printf("read_mode: %d\n",read_mode);

            // i=i+4;
            // printf("i:%d\n", i);

            arr_num_file = (int *)malloc(n_file*sizeof(int));

            int j = 0;
            int cc = n_file;
            while(cc > 0) {
                // printf("file_argv[%d]: %s\n",i,argv[i]);
                arr_num_file[j] = i;

                i++;
                j++;
                cc--;

            }
            i--;

            flag_file = 0;
        }
        else if (flag_cross) {
            
            num_opts = 2;

            // printf("i_flag_cross: %d, argc: %d\n",i,argc);

            if(!checkFlagOpts(argv,argc,i,num_opts)) {
                printf("The number of options is incorrect. For more information run the execution as: %s --help\n",argv[0]);
                exit(-1);
            }

            k_fold = atoi(argv[i]);
            flag_shuffle = atoi(argv[i+1]);
            printf("k_fold: %d\n",k_fold);
            if (k_fold < 2) {
                printf("It's not possible to perform 1-cross validation!\n");
                exit(-1);
            }

            i = i + 1;

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
                    // printf("dataSize: %d\n",dataSize);
                    i = i + 1;
                }
                else {

                    // printf("i_flag_opt: %d, argc: %d\n",i,argc);

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

                    // printf("train_size: %d\n",train_size);
                    // printf("test_size: %d\n",test_size);

                    i = i + 2;
                }
            }
            else if ( strcmp(task,"SUBSEQ_SEARCH") == 0 ) { //SUBSEQ_SEARCH TASK
                // printf("SUBSEQ_SEARCH\n");
                num_opts = 2;

                if(!checkFlagOpts(argv,argc,i,num_opts)) {
                    printf("The number of options is incorrect. For more information run the execution as: %s --help\n",argv[0]);
                    exit(-1);
                }
                t_size = atoi(argv[i]);
                // printf("T_SIZE: %d\n", t_size);
                q_size = atoi(argv[i+1]);
                // printf("Q_SIZE: %d\n", q_size);
                i = i + 1;
            }
            else{
                printf("The number of options is incorrect. For more information run the execution as: %s --help\n",argv[0]);
                exit(-1);
            }
                    
            
            // if(window_size>dataSize){
            //  printf("Something went wrong! The size of the data is larger than the features space.");
            //  exit(2);
            // }
                        
            
            flag_opt=0;
        }

        else if (flag_alg_mode){
            num_opts = 2;

            // printf("i_flag_alg_mode: %d, argc: %d\n",i,argc);

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

            // printf("class_mode: %d\n",class_mode);
            distance_type = argv[i+1];
            // printf("distance type: %s\n",distance_type);
            i = i+1;

            flag_alg_mode = 0;
        }
        else if (flag_device){
            num_opts = 1;

            // printf("i_flag_device: %d, argc: %d\n",i,argc);
            
            if(!checkFlagOpts(argv,argc,i,num_opts)) {
                printf("The number of options is incorrect. For more information run the execution as: %s --help\n",argv[0]);
                exit(-1);
            }


            device = atoi(argv[i]);
            // printf("device: %d\n",device);
            flag_device = 0;

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
        else if(!strcmp(argv[i], "--help"))
            print_help();
        else if(!strcmp(argv[i], "--version"))
            print_version();
        else if(!strcmp(argv[i], "--infoDevice"))
            infoDev();
        else
        {
            printf("The number of options is incorrect. For more information run the execution as: %s --help\n",argv[0]);
            exit(-1);
        }
    }


    if( strcmp(task,"CLASSIFICATION") == 0 ){

        unsigned long long int dataBytes = 2 * dataSize * window_size * n_feat * sizeof(float);

        int * dataLabels = (int *) malloc(dataSize*sizeof(int));
        float* data = (float*) malloc (dataBytes);

        // printf("Reading data...\n");

        readFile(argv, arr_num_file, n_file, read_mode, data, data_struct, window_size, dataLabels, n_feat, class_mode);
        // printf("datasize: %d\n", dataSize);
        // printf("window_size: %d\n", window_size);
        // printArray(data,dataSize * window_size * n_feat);
        // printArrayI(dataLabels,dataSize);
        // exit(-1);


        float mean_RIA = 0.0f, mean_RA = 0.0f, mean_ER_RIA = 0.0f, mean_ER_RA = 0.0f;

        // printArrayI(dataLabels,dataSize);

        if( k_fold < 1) //not doing K-cross validation
            k_fold = 1; // (work around to do not re-write a lot of code)
        else
            tInd = crossvalind_Kfold(dataLabels,dataSize, k_fold, flag_shuffle);

        // int* tInd=(int *)malloc(dataSize*sizeof(int)); /*********** FOR TESTING SMTH ***********/

           // for (i=0; i<dataSize; i++)
           //     printf("tInd: %d\n",tInd[i]);

        int trainSize, testSize;

        for (f = 0; f < k_fold; f++){
            //SETTIN FOR EACH K_FOLD ALL THE VARIABLES 
            err = 0;
            errNR = 0;


            // i_train = 0;
            // i_test = 0;
            if (k_fold > 1) { //doing K-fold cross validation
                testSize = countVal(tInd,dataSize,f);
                trainSize = dataSize - testSize;

                // printf("DataSize: %d\n",dataSize);
                // printf("trainSize: %d\n",trainSize);
                // printf("testSize: %d\n",testSize);
                // printf("window_size: %d\n",window_size);
                // printf("n_feat: %d\n",n_feat);
                // printf("\nK_FOLD: %d\n",f+1);
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

            // printArray(h_train,trainSize * window_size * n_feat);
            // printArray(h_test,testSize * window_size * n_feat);


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
                            // printf("CPU_test[%d]\n", i+1);
                            for (int j = 0; j < trainSize; j++) {
                                // z_normalize2D(&h_train[j*dimensions*window_size],dimensions,window_size);
                                // printMatrix(&h_train[i*dimensions*window_size],dimensions,window_size);
                                if ( strcmp(distance_type,"DTW") == 0) //DTW distance
                                    h_Out[j] = short_md_dtw_c(&h_train[j*n_feat*window_size], &h_test[k*n_feat*window_size], window_size, window_size, n_feat, window_size);
                                else //Euclidean distance
                                    h_Out[j] = short_md_ed_c(&h_train[j*n_feat*window_size], &h_test[k*n_feat*window_size], window_size, n_feat, window_size);    

                                // printf("h_Out[%d]: %f\n",j,h_Out[j]);
                            }

                            // minI = (int *) malloc(sizeof(int));
                            min = min_arr(h_Out,trainSize,minI);

                            if( trainLabels[*minI] != testLabels[k] )
                                err++;

                            printf("%d\t gt: %d\t\tRI: %d\t%3.6f\n",k , testLabels[k] , trainLabels[*minI], min);
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
                                // z_normalize2D(&h_train[j*dimensions*window_size],dimensions,window_size);
                                for (int d = 0; d < n_feat; d++) {
                                    if ( strcmp(distance_type,"DTW") == 0) //DTW distance
                                        dtw_curr = short_dtw_c(&h_train[(d*window_size)+(j*n_feat*window_size)],&h_test[(k*n_feat*window_size)+(d*window_size)],window_size,window_size);
                                    else
                                        dtw_curr = short_ed_c(&h_train[(d*window_size)+(j*n_feat*window_size)],&h_test[(k*n_feat*window_size)+(d*window_size)],window_size);
                                    cum_sum += dtw_curr;
                                    // printf("dtw[%d]: %f\n",k,dtw_curr);
                                }
                                h_Out[j] = cum_sum;
                                // printf("h_Out[%d]: %f\n",j,h_Out[j]);
                            }

                            // minI = (int *) malloc(sizeof(int));
                            min = min_arr(h_Out,trainSize,minI);

                            if( trainLabels[*minI] != testLabels[k] )
                                err++;

                            printf("%d\t gt: %d\t\tRI: %d\t%3.6f\n",k , testLabels[k] , trainLabels[*minI], min);
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
                            // printf("j: %d\n",j);
                            for (int k = 0; k < window_size; k++) {
                                // short_md_dtw_c(&h_train[j*n_feat*window_size], &h_test[k*n_feat*window_size], window_size, window_size, n_feat, window_size);
                                if ( strcmp(distance_type,"DTW") == 0) //DTW distance
                                    h_Out[(j*window_size)+k] = short_md_dtw_c(&h_train[(2*j*n_feat*window_size)+k],&h_test[i*n_feat*window_size],window_size,window_size,n_feat,2*window_size);
                                else
                                    h_Out[(j*window_size)+k] = short_md_ed_c(&h_train[(2*j*n_feat*window_size)+k],&h_test[i*n_feat*window_size],window_size,n_feat,2*window_size);
                                // printf("shift-%d, dtw[%d]: %f\n",k,j,h_Out[k]);
                            }
                            // min = min_arr(h_Out,trainSize,minI);
                        }
                        // printArray(h_Out,trainSize*window_size);
                        // exit(-1);
                        min = 9999999999.99;
                        // minI = (int *) malloc(sizeof(int));

                        *minI = -1;
                        minINR = -1;
                        minNR = 99999999999.99;
                        for(int m = 0 ; m < trainSize ; m++ )
                        {
                                // printf("DTW[%d]: %f\n",window_size*j,h_Out[window_size*j]);
                            if (h_Out[m*window_size] < minNR )
                            {
                                minNR = h_Out[m*window_size];
                                minINR = m;
                            }
                                // printf("minNR: %f, minINR: %d, trainLabel: %d\n",minNR,minINR,trainLabels[minINR]);
                            for(int p = 0 ; p < window_size ; p++ )
                            {
                                int t = m*window_size+p;
                                // printf("DTW[%d]: %f\n",t,h_Out[t]);
                                if ( h_Out[t] < min )
                                {
                                    min = h_Out[t];
                                    *minI = m;
                                }
                            }
                            // printf("min: %f, minI: %d, trainLabel: %d\n",min,minI,trainLabels[minI]);
                        }
                        // exit(-1);
                        if( trainLabels[*minI] != testLabels[i] )
                            err++;

                        if( trainLabels[minINR] != testLabels[i] )
                            errNR++;

                        printf("%d\t gt: %d\t\tRI: %d\t%3.6f \t\t NRI: %d\t%3.6f\n",i , testLabels[i] , trainLabels[*minI], min, trainLabels[minINR], minNR );

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
                // cudaMemcpy(d_test, h_test, n_feat*window_size*sizeof(float), cudaMemcpyHostToDevice);


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


                // rMD_DTW_D <<<grid, threads, T2>>> (d_train, d_test, window_size, window_size,n_feat,d_Out,trainSize);
                // cudaThreadSynchronize();
                // cudaMemcpy(h_Out, d_Out, trainSize*window_size*sizeof(float), cudaMemcpyDeviceToHost);


                // printArray(h_Out,trainSize*window_size);

                // for (i = 0; i < trainSize; i++)
                //  printf("DTW[%d]: %f\n",window_size*i,h_Out[window_size*i]);
                // exit(-1);

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
                        // grid(grid_size,1);
                        // dim3 threads(blockSize,1);
                        grid.x = grid_size;
                        grid.y = 1;
                        threads.x = blockSize;
                        threads.y = 1;

                        printf("Grid_size_x: %d, number_of_threads_x: %d \n", grid.x,threads.x);
                        printf("Grid_size_y: %d, number_of_threads_y: %d \n\n", grid.y,threads.y);

                        cudaEventCreate(&start_GPU);
                        cudaEventCreate(&stop_GPU);
                        cudaEventRecord(start_GPU,0);

                        for (k = 0; k < testSize; k++){
                            // printf("GPU_test[%d]\n", i+1);
                            cudaMemset(&d_test, 0, n_feat*window_size*sizeof(float));
                            cudaMemcpy(d_test, h_test + k*(n_feat*window_size) , n_feat*window_size*sizeof(float), cudaMemcpyHostToDevice);

                            if ( strcmp(distance_type,"DTW") == 0) //DTW distance
                                // gMD_DTW_D <<<grid, threads>>> (d_train, d_test, window_size, window_size,n_feat,d_Out,trainSize,0);
                                MD_DTW_D <<<grid, threads, T2>>> (d_train, d_test, window_size, window_size,n_feat,d_Out,trainSize,0,gm);
                            else
                                MD_ED_D <<<grid, threads, T2>>> (d_train, d_test, window_size,n_feat,d_Out,trainSize,0, gm);

                            cudaDeviceSynchronize(); //it may be avoided if there's not printf in the kernel function
                            cudaMemcpy(h_Out, d_Out, trainSize*sizeof(float), cudaMemcpyDeviceToHost);
                            // printArray(h_Out,trainSize);


                            // min_arr(h_Out,trainSize,minI)
                            // float  min = 9999999999.99;
                            // minI = -1;
                            // minI = (int *) malloc(sizeof(int));
                            min = min_arr(h_Out,trainSize,minI);

                            // for (i = 0; i < trainSize; i++)
                            // {
                            //     if(h_Out[i] < min)
                            //     {
                            //         min = h_Out[i];
                            //         minI = i;
                            //     }
                            // }

                            if( trainLabels[*minI] != testLabels[k] )
                                err++;

                            printf("%d\t gt: %d\t\tRI: %d\t%3.6f\n",k , testLabels[k] , trainLabels[*minI], min);

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
                        // dim3 grid2(grid_size,1);
                        float dim_row = floor((float)blockSize/n_feat);
                        float dim_col = n_feat;    
                        // dim3 threads2(dim_row,dim_col);

                        // number of blocks (x,y) for a grid
                        grid.x = grid_size;
                        grid.y = 1;
                        // number of threads (x,y) for each block
                        threads.x = dim_row;
                        threads.y = dim_col;


                        printf("Grid_size_x: %d, number_of_threads_x: %d \n", grid.x,threads.x);
                        printf("Grid_size_y: %d, number_of_threads_y: %d \n\n", grid.y,threads.y);


                        float sh_mem = ((threads.x*threads.y)+(n_feat*window_size))*sizeof(float);
                        // float DTW_single_dim = (threads.x*threads.y)*sizeof(float);

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
                            // printArray(DTW_indep_host,trainSize);
                            // exit(-1);

                            min = min_arr(h_Out,trainSize,minI);

                            if( trainLabels[*minI] != testLabels[k] )
                                err++;

                            printf("%d\t gt: %d\t\tRI: %d\t%3.6f\n",k , testLabels[k] , trainLabels[*minI], min);
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
                        // dim3 grid(grid_size,1);
                        // dim3 threads(blockSize,1);
                        grid.x = grid_size;
                        grid.y = 1;
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
                            // minI = (int *) malloc(sizeof(int));

                            *minI = -1;
                            minINR = -1; 
                            minNR = 99999999999.99;
                            i = 0;
                            for(j = 0 ; j < trainSize ; j++ )
                            {
                                    // printf("DTW[%d]: %f\n",window_size*j,h_Out[window_size*j]);
                                if (h_Out[j*window_size] < minNR )
                                {
                                    minNR = h_Out[j*window_size];
                                    minINR = j;
                                }
                                    // printf("minNR: %f, minINR: %d, trainLabel: %d\n",minNR,minINR,trainLabels[minINR]);
                                for( i = 0 ; i < window_size ; i++ )
                                {
                                    int t = j*window_size+i;
                                    // printf("DTW[%d]: %f\n",t,h_Out[t]);
                                    if ( h_Out[t] < min )
                                    {
                                        min = h_Out[t];
                                        *minI = j;
                                    }
                                }
                                // printf("min: %f, minI: %d, trainLabel: %d\n",min,minI,trainLabels[minI]);
                            }
                            // exit(-1);
                            if( trainLabels[*minI] != testLabels[k] )
                                err++;

                            if( trainLabels[minINR] != testLabels[k] )
                                errNR++;

                            printf("%d\t gt: %d\t\tRI: %d\t%3.6f \t\t NRI: %d\t%3.6f\n",k , testLabels[k] , trainLabels[*minI], min, trainLabels[minINR], minNR );
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
                        // exit(-1);
                        break;
                        default:
                            printf("Error algorithm choice\n");
                    }
                    break;
                }
            }
        }
        if (k_fold > 1){
            mean_RA /= k_fold;
            mean_ER_RA /= k_fold;
            printf("\nRegular Accuracy mean is %f\n",mean_RA);
            printf("The Error rate mean of NR is %f\n",mean_ER_RA);
        }
        
        if (class_mode > 2) //rDTW
        {
            mean_RIA /= k_fold;
            mean_ER_RIA /= k_fold;
            printf("\nRotation Invariant Accuracy mean is %f\n",mean_RIA);
            printf("The Error rate mean of RI is %f\n",mean_ER_RIA);
        }        
    }
    else if( strcmp(task,"SUBSEQ_SEARCH") == 0 ){

        //nss=number of subsequences
        nss = t_size - q_size + 1;
        window_size = q_size; //query size
        // printf("t_size: %d\n", t_size);
        // printf("q_size/window_size: %d\n", q_size);
        // printf("nss: %d\n", nss);


        //T and Q lengths series bytes
        unsigned long long int t_bytes = t_size * n_feat * sizeof(float);
        unsigned long long int q_bytes = q_size * n_feat * sizeof(float);

        ////////////////////////////////CPU MEMORY ALLOCATION////////////////////////////////
        float *t_series = (float *) malloc (t_bytes);
        float *q_series = (float *) malloc (q_bytes);
        float* owp = (float*) malloc (nss*sizeof(float));
        memset(owp,0,nss*sizeof(float));
        // float *d_owp_copy;

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
                        // z_normalize(&t_serie[i],window_size,subseq_norm);
                        // float short_md_dtw_c(float *S, float *T,int ns, int nt,int dim)
                        // min = short_dtw_c(&t_series[i],q_series, window_size, window_size);
                        if ( strcmp(distance_type,"DTW") == 0) //DTW distance
                            dist = short_md_dtw_c(&t_series[i], q_series, window_size, window_size, n_feat, t_size);
                        else
                            dist = short_md_ed_c(&t_series[i], q_series, window_size, n_feat, t_size);

                        owp[i] = dist;
                        printf("val[%d]: %f\n", i, owp[i]);
                    }
                    gettimeofday(&stop_CPU, NULL);

                    elapsed = timedifference_msec(start_CPU, stop_CPU);
                    printf("\nExecution time for %s w/ CPU using %s-%s:  %f ms\n" ,task, strategy, distance_type, elapsed);

                    //computing minimum value
                    min = min_arr(owp,nss,ind_min_val);
                    printf("ind_min_val_CPU_version: %d, min_val_CPU_version: %f\n\n" ,*ind_min_val,min);
                }

                break;

                case 1: {

                    printf("%s-%s %s version processing...\n" ,distance_type,strategy,compution_type);

                    gettimeofday(&start_CPU, NULL);

                    for (int i = 0; i < nss; i++) {
                        dist = 0.0;
                        // z_normalize2D(&h_train[j*dimensions*window_size],dimensions,window_size);
                        for (int k = 0; k < n_feat; k++) {
                            if ( strcmp(distance_type,"DTW") == 0) //DTW distance
                                val_curr = short_dtw_c(&t_series[(k*t_size)+i],&q_series[(k*window_size)],window_size,window_size);
                            else
                                val_curr = short_ed_c(&t_series[(k*t_size)+i],&q_series[(k*window_size)],window_size);

                            dist += val_curr;
                            // printf("dtw[%d]: %f\n",k,dtw_curr);
                        }
                        owp[i] = dist;
                        // printf("min[%d]: %f\n", i, owp[i]);
                    }
                    gettimeofday(&stop_CPU, NULL);

                    elapsed = timedifference_msec(start_CPU, stop_CPU);
                    printf("\nExecution time for %s w/ CPU using %s-%s:  %f ms\n" ,task, strategy, distance_type, elapsed);

                    //computing minimum value
                    min = min_arr(owp,nss,ind_min_val);
                    printf("ind_min_val_CPU_version: %d, min_val_CPU_version: %f\n\n",*ind_min_val,min);                    
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
                        // exit(-1);
                    }
                    else
                        gm = 0;

                    printf("Grid_size_x: %d, number_of_threads_x: %d \n", grid.x,threads.x);
                    printf("Grid_size_y: %d, number_of_threads_y: %d \n\n", grid.y,threads.y);

                    printf("%s-%s %s version processing...\n" ,distance_type,strategy,compution_type);
                    cudaEventCreate(&start_GPU);
                    cudaEventCreate(&stop_GPU);
                    cudaEventRecord(start_GPU,0);

    //                 cudaMemset(&d_test, 0, n_feat*window_size*sizeof(float));
    //                 cudaMemcpy(d_test, h_test + k*(n_feat*window_size) , n_feat*window_size*sizeof(float), cudaMemcpyHostToDevice);
                    if ( strcmp(distance_type,"DTW") == 0) //DTW distance
                        MD_DTW_D <<<grid, threads, T2>>> (d_t_series, d_q_series, window_size, window_size,n_feat,d_owp,t_size,1,gm);
                    else
                        MD_ED_D <<<grid, threads, T2>>> (d_t_series, d_q_series, window_size, n_feat, d_owp,t_size,1,gm)
                    ;

                    // cudaDeviceSynchronize(); //it may be avoided if there's not printf in the kernel function
                    cudaEventRecord(stop_GPU,0);
                    cudaEventSynchronize(stop_GPU);
                    cudaEventElapsedTime(&time_GPU_MD_DTW_D,start_GPU,stop_GPU);
                    cudaEventDestroy(start_GPU);
                    cudaEventDestroy(stop_GPU);
                    printf("\nExecution time for %s w/ MD_DTW_D with GPU %f ms\n", task, time_GPU_MD_DTW_D);

                    // d_owp_copy = (float*) malloc(nss*sizeof(float));
                    cudaMemcpy(owp, d_owp, nss * sizeof(float), cudaMemcpyDeviceToHost);
                    // printArray(owp, nss);
                    min = min_arr(owp, nss, ind_min_val);
                    printf("ind_min_val_GPU_GM_version: %d, min_val_GPU_GM_version: %f\n\n",*ind_min_val,min);

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


                    // d_owp_copy = (float*) malloc(nss*sizeof(float));
                    cudaMemcpy(owp, d_owp, nss * sizeof(float), cudaMemcpyDeviceToHost);
                    // printArray(owp, nss);
                    min = min_arr(owp, nss, ind_min_val);
                    printf("ind_min_val_GPU_GM_version: %d, min_val_GPU_GM_version: %f\n\n",*ind_min_val,min);
                }
                break;

                default:
                    printf("Error algorithm choice\n");
            }
            cudaFree(d_t_series);
            cudaFree(d_q_series);
            cudaFree(d_owp);
            // free(d_owp_copy);
        }
        free(t_series);
        free(q_series);
        free(owp);
        printf("\nMemory deallocated!\n\n");

    }
    return 0;
}
