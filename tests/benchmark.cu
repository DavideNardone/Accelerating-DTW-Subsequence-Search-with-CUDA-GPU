#include "../include/header.h"

using namespace std;

void h_malloc(float **train_set, float **test_set, int **trainLabels, int **testLabels, int trainSize, int testSize, int window_size, int n_feat, int cls){

  unsigned long long int trainBytes, testBytes;


  if (cls < 2)
    trainBytes = trainSize * window_size * n_feat * sizeof(float);
  else
    trainBytes = 2 * trainSize * window_size * n_feat * sizeof(float);

  testBytes = testSize * window_size * n_feat * sizeof(float);


  *train_set = (float *)malloc(trainBytes);
  *test_set = (float *)malloc(testBytes);

  *trainLabels = (int *)malloc(trainSize * sizeof(int));
  *testLabels = (int *)malloc(testSize * sizeof(int));
  // random initialization of train data and label set
  initializeArray(*train_set, trainSize * window_size * n_feat);
  initializeArray(*test_set, testSize * window_size * n_feat);

  initializeArray(*trainLabels, trainSize);
  initializeArray(*testLabels, testSize);
}

void h_malloc(int t_size, int q_size, int n_feat, float **t_series, float **q_series, float **owp){

    int nss = t_size - q_size + 1;

    unsigned long long int t_bytes = t_size * n_feat * sizeof(float);
    unsigned long long int q_bytes = q_size * n_feat * sizeof(float);

    /* *************** CPU MEMORY ALLOCATION *************** */
    *t_series = (float *)malloc(t_bytes);
    *q_series = (float *)malloc(q_bytes);
    *owp = (float *)malloc(nss * sizeof(float));
    memset(*owp, 0, nss * sizeof(float));

    // random initialization the two sequences
    initializeArray(*t_series, t_size * n_feat);
    initializeArray(*q_series, q_size * n_feat);
}

void h_free(float **train_set, float **test_set, int **trainLabels, int **testLabels, float *h_Out){

  free(*train_set);
  free(*test_set);
  free(*trainLabels);
  free(*testLabels);
  free(h_Out);
}


void h_free(float **q_series, float **t_series, float **owp){

  free(*q_series);
  free(*t_series);
  free(*owp);
}

void run_benchmark(int nss, int t_size, int q_size, int blockSize, int n_feat, cudaDeviceProp deviceProp,  
  float *t_series, float *q_series, float *d_t_series, float *d_q_series, float *d_owp, float *owp, int task){

    int ind_min_val = 0;
    struct timeval stop_CPU, start_CPU;
    cudaEvent_t start_GPU, stop_GPU;
    char *distance_type[] = {"ED","DTW"};
    float time_cpu, time_gpu;

  switch(task){

    case 0:
      for (int i = 0; i < 2; ++i)
      {
        printf("RUNNING BENCHMARK ON MD_D-%s...\n", *(distance_type + i));
        gettimeofday(&start_CPU, NULL);
        MDD_SIM_MES_CPU(nss, t_series, q_series, t_size, q_size, n_feat, *(distance_type + i), 0, owp, &ind_min_val);
        gettimeofday(&stop_CPU, NULL);
        time_cpu = timedifference_msec(start_CPU, stop_CPU);

        cudaEventCreate(&start_GPU);
        cudaEventCreate(&stop_GPU);
        cudaEventRecord(start_GPU, 0);
        MDD_SIM_MES_GPU(nss, d_t_series, d_q_series, t_size, q_size, n_feat, blockSize, deviceProp, *(distance_type + i), 0, owp, d_owp, &ind_min_val);
        cudaEventRecord(stop_GPU, 0);
        cudaEventSynchronize(stop_GPU);
        cudaEventElapsedTime(&time_gpu, start_GPU, stop_GPU);
        cudaEventDestroy(start_GPU);
        cudaEventDestroy(stop_GPU);
        printf("CPU %f ms vs GPU  %f ms\n", time_cpu, time_gpu);
      }
    break;
    case 1:
      for (int i = 0; i < 2; ++i)
      {
        printf("RUNNING BENCHMARK ON MD_I-%s...\n", *(distance_type + i));
        gettimeofday(&start_CPU, NULL);
        MDI_SIM_MES_CPU(nss, t_series, q_series, t_size, q_size, n_feat, *(distance_type + i), 0, owp, &ind_min_val);
        gettimeofday(&stop_CPU, NULL);
        time_cpu = timedifference_msec(start_CPU, stop_CPU);

        cudaEventCreate(&start_GPU);
        cudaEventCreate(&stop_GPU);
        cudaEventRecord(start_GPU, 0);
        MDD_SIM_MES_GPU(nss, d_t_series, d_q_series, t_size, q_size, n_feat, blockSize, deviceProp, *(distance_type + i), 0, owp, d_owp, &ind_min_val);
        cudaEventRecord(stop_GPU, 0);
        cudaEventSynchronize(stop_GPU);
        cudaEventElapsedTime(&time_gpu, start_GPU, stop_GPU);
        cudaEventDestroy(start_GPU);
        cudaEventDestroy(stop_GPU);
        printf("CPU %f ms vs GPU  %f ms\n", time_cpu, time_gpu);
      }
    break;
  }

}

void run_benchmark(int trainSize, int testSize, int blockSize, int window_size, int n_feat, cudaDeviceProp deviceProp,  
  int *trainLabels, int *testLabels, float *train_set, float *test_set, float *d_train, float *d_test, float *d_Out, float *h_Out, int task){

  int ERR_CPU, ERR_GPU,ERR_NR_CPU,ERR_NR_GPU;
  struct timeval stop_CPU, start_CPU;
  cudaEvent_t start_GPU, stop_GPU;
  char *distance_type[] = {"ED","DTW"};
  float time_cpu, time_gpu;

  switch(task){

    //benchmark for Dependent-Similarity Measure Distance among CPU and GPU version
    case 0:
      for (int i = 0; i < 2; ++i)
      {
        printf("RUNNING BENCHMARK ON MD_D-%s...\n", *(distance_type + i));
        gettimeofday(&start_CPU, NULL);
        ERR_CPU = MDD_SIM_MES_CPU(trainSize, testSize, trainLabels, testLabels, train_set, test_set, window_size, n_feat, *(distance_type + i), 0);
        gettimeofday(&stop_CPU, NULL);
        time_cpu = timedifference_msec(start_CPU, stop_CPU);

        cudaEventCreate(&start_GPU);
        cudaEventCreate(&stop_GPU);
        cudaEventRecord(start_GPU, 0);
        ERR_GPU = MDD_SIM_MES_GPU(trainSize, testSize, trainLabels, testLabels, train_set, test_set, d_train, d_test, d_Out, h_Out, window_size, n_feat, 512, deviceProp, *(distance_type + i), 0);
        cudaEventRecord(stop_GPU, 0);
        cudaEventSynchronize(stop_GPU);
        cudaEventElapsedTime(&time_gpu, start_GPU, stop_GPU);
        cudaEventDestroy(start_GPU);
        cudaEventDestroy(stop_GPU);
        printf("CPU %f ms vs GPU  %f ms\n", time_cpu, time_gpu);
      }
    break;

    //benchmark for independent-Similarity Measure Distance among CPU and GPU version
    case 1:
      for (int i = 0; i < 2; ++i)
      {
        printf("RUNNING BENCHMARK ON MD_I-%s...\n", *(distance_type + i));
        gettimeofday(&start_CPU, NULL);
        ERR_CPU = MDI_SIM_MES_CPU(trainSize, testSize, trainLabels, testLabels, train_set, test_set, window_size, n_feat, *(distance_type + i), 0);
        gettimeofday(&stop_CPU, NULL);
        time_cpu = timedifference_msec(start_CPU, stop_CPU);

        cudaEventCreate(&start_GPU);
        cudaEventCreate(&stop_GPU);
        cudaEventRecord(start_GPU, 0);
        ERR_GPU = MDI_SIM_MES_GPU(trainSize, testSize, trainLabels, testLabels, train_set, test_set, d_train, d_test, d_Out, h_Out, window_size, n_feat, blockSize, deviceProp, *(distance_type + i), 0);
        cudaEventRecord(stop_GPU, 0);
        cudaEventSynchronize(stop_GPU);
        cudaEventElapsedTime(&time_gpu, start_GPU, stop_GPU);
        cudaEventDestroy(start_GPU);
        cudaEventDestroy(stop_GPU);
        printf("CPU %f ms vs GPU  %f ms\n", time_cpu, time_gpu);
        // printf("\n");
      }
    break;
    //benchmark for Rotation Dependent-Similarity Measure Distance among CPU and GPU version
    case 3:
      for (int i = 0; i < 2; ++i)
      {
        printf("RUNNING BENCHMARK ON MDR-%s...\n", *(distance_type + i));
        gettimeofday(&start_CPU, NULL);
        MDR_SIM_MES_CPU(trainSize, testSize, trainLabels,  testLabels, train_set, test_set, window_size, n_feat, *(distance_type + i), 0, &ERR_CPU, &ERR_NR_CPU);
        gettimeofday(&stop_CPU, NULL);
        time_cpu = timedifference_msec(start_CPU, stop_CPU);

        cudaEventCreate(&start_GPU);
        cudaEventCreate(&stop_GPU);
        cudaEventRecord(start_GPU, 0);
        MDR_SIM_MES_GPU(trainSize, testSize, trainLabels, testLabels, train_set, test_set, d_train, d_test, d_Out, h_Out, window_size, n_feat, blockSize, deviceProp, *(distance_type + i), 0, &ERR_GPU, &ERR_NR_GPU);
        cudaEventRecord(stop_GPU, 0);
        cudaEventSynchronize(stop_GPU);
        cudaEventElapsedTime(&time_gpu, start_GPU, stop_GPU);
        cudaEventDestroy(start_GPU);
        cudaEventDestroy(stop_GPU);
        printf("CPU %f ms vs GPU  %f ms\n", time_cpu, time_gpu);
        // printf("\n");
      }
    break;
  }
}

int main(int argc, char **argv) {

  float *train_set = 0, *test_set = 0, *d_train = 0, *d_test = 0, *d_Out = 0, *h_Out = 0;
  int *trainLabels = 0, *testLabels = 0;

  int testSize = 0;
  int trainSize = 0;
  int window_size = 0;
  int n_feat = 0;
  int blockSize = 0;

  //SETTING PARAMETERS
  int start_iter = 0;
  int end_iter = 12,i=0,j=0,k=0,l=0,p=0;

  int grid_params[12][5] = {  

     {10, 100, 15, 1,2},   
     {30, 200, 30, 3,4},   
     {50, 250, 50, 5,8},
     {70, 300, 100, 7,16},
     {100, 350, 170, 10,32},
     {150, 400, 200, 13,64},
     {200, 500, 250, 15,128},
     {250, 700, 300, 17,256},
     {300, 1000, 350, 20,512},
     {350, 1300, 400, 25,1024},
     {400, 1500, 500, 30,1024},
     {500, 2000, 1000, 50,1024}   
  };

  cudaDeviceProp deviceProp = getDevProp(0);

  //CLASSIFICATION TASK
  for (i = start_iter; i < end_iter; i++) 
  {
    testSize = grid_params[i][0];

    for (j = start_iter; j < end_iter; j++)
    {
      trainSize = grid_params[j][1];

      for (k = start_iter; k < end_iter; k++)
      {
        window_size = grid_params[k][2];

        for (l = start_iter; l < end_iter; l++)
        {
          n_feat = grid_params[l][3];

          for (p = start_iter; p < end_iter; p++)
          {

            blockSize = grid_params[p][4];

            printf("\nRunning benchmarks on classification task with trainSize[%d], testSize[%d], window_size[%d], n_feat[%d] \n", trainSize, testSize, window_size, n_feat);

            /* HOST MEMORY ALLOCATION */
            h_malloc(&train_set, &test_set, &trainLabels, &testLabels, trainSize, testSize, window_size, n_feat, 1);

            /* DEVICE MEMORY ALLOCATION */
            unsigned long long int trainBytes;

            trainBytes = trainSize * window_size * n_feat * sizeof(float);

            cudaMalloc((void **)&d_Out, trainSize * sizeof(float));
            cudaMemset(d_Out, 0, trainSize * sizeof(float));
            h_Out = (float *)malloc(trainSize * sizeof(float));
            memset(h_Out, 0, trainSize * sizeof(float));

            
            cudaMalloc((void **)&d_train, trainBytes);
            cudaMemcpy(d_train, train_set, trainBytes, cudaMemcpyHostToDevice);

            cudaMalloc((void **)&d_test, n_feat * window_size * sizeof(float));
            /* DEVICE MEMORY ALLOCATION */

            run_benchmark(trainSize, testSize, blockSize, window_size, n_feat, deviceProp, 
              trainLabels, testLabels, train_set, test_set, d_train, d_test, d_Out, h_Out, 0); 

            run_benchmark(trainSize, testSize, blockSize, window_size, n_feat, deviceProp, 
              trainLabels, testLabels, train_set, test_set, d_train, d_test, d_Out, h_Out, 1);


           // /*--------------------- Rotation Invariant ---------------------*/
            trainBytes = 2 * trainSize * window_size * n_feat * sizeof(float);

            /* HOST MEMORY ALLOCATION */

            /* DEVICE MEMORY ALLOCATION */
            cudaMalloc((void **)&d_Out, trainSize * window_size * sizeof(float));
            cudaMemset(d_Out, 0, trainSize * window_size * sizeof(float));
            h_Out = (float *)malloc(trainSize * window_size * sizeof(float));
            memset(h_Out, 0, trainSize * window_size * sizeof(float));
            cudaMalloc((void **)&d_train, trainBytes);
            cudaMemcpy(d_train, train_set, trainBytes, cudaMemcpyHostToDevice);
            cudaMalloc((void **)&d_test, n_feat * window_size * sizeof(float));
            /* DEVICE MEMORY ALLOCATION */


            run_benchmark(trainSize, testSize, blockSize, window_size, n_feat, deviceProp, 
              trainLabels, testLabels, train_set, test_set, d_train, d_test, d_Out, h_Out, 3);

            h_free(&train_set, &test_set, &trainLabels, &testLabels, h_Out);

            cudaFree(d_train);
            cudaFree(d_test);
            cudaFree(d_Out);
          }
        }
      }
    }
  }

  //SUB-SEQ SEARCH
  int t_size = 0;
  int q_size = 0;
  int nss = 0;
  float *t_series = 0, *q_series = 0, *owp = 0;

  start_iter = 0;
  end_iter = 12;
  int grid_params_2[12][4] = {  
     {50, 10, 1, 2},   
     {100, 75, 3, 4},   
     {300, 100, 5, 8},
     {500, 125, 7, 16},
     {700, 150, 10, 32},
     {800, 200, 13, 64},
     {1000, 300, 15, 128},
     {1200, 500, 17, 256},
     {1300, 600, 20, 512},
     {1500, 700, 25, 1024},
     {1800, 1000, 30, 1024},
     {2000, 1300, 50, 1024}   
  };

  for (i = start_iter; i < end_iter; i++)
  {

    t_size = grid_params_2[i][0];

    for (j = start_iter; j < end_iter; j++)
    {

      q_size = grid_params_2[j][1];

      for (k = start_iter; i < end_iter; k++)
      {
        n_feat = grid_params_2[k][2];

        for (p = start_iter; p < end_iter; p++)
        {
          blockSize = grid_params_2[p][3];

          nss = t_size - q_size + 1;

          h_malloc(t_size, q_size, n_feat, &t_series, &q_series, &owp);

          /* *************** DEVICE MEMORY ALLOCATION *************** */

          unsigned long long int t_bytes = t_size * n_feat * sizeof(float);
          unsigned long long int q_bytes = q_size * n_feat * sizeof(float);

          float *d_t_series = 0, *d_owp = 0, *d_q_series = 0;
          cudaMalloc((void **)&d_t_series, t_bytes);
          cudaMemcpy(d_t_series, t_series, t_bytes, cudaMemcpyHostToDevice);

          cudaMalloc((void **)&d_q_series, q_bytes);
          cudaMemcpy(d_q_series, q_series, q_bytes, cudaMemcpyHostToDevice);

          cudaMalloc((void **)&d_owp, nss * sizeof(float));
          cudaMemset(d_owp, 0, nss * sizeof(float));
          /* *************** DEVICE MEMORY ALLOCATION *************** */

          run_benchmark(nss, t_size, q_size, blockSize, n_feat, deviceProp, t_series, q_series, d_t_series, d_q_series, d_owp, owp, 0);


          h_free(&t_series, &q_series, &owp);

          cudaFree(d_t_series);
          cudaFree(d_q_series);
          cudaFree(d_owp);
        }
      }
    }
  }
}