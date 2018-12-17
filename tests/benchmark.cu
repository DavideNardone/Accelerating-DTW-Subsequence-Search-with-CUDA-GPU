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

void h_free(float **train_set, float **test_set, int **trainLabels, int **testLabels, float *h_Out){

  free(*train_set);
  free(*test_set);
  free(*trainLabels);
  free(*testLabels);
  free(h_Out);
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

  //SETTING PARAMETERS
  int start_iter = 0;
  int end_iter = 12;

  int grid_params[12][4] = {  
     {10, 100, 15, 1},   
     {30, 200, 30, 3},   
     {50, 250, 50, 5},
     {70, 300, 100, 7},
     {100, 350, 170, 10},
     {150, 400, 200, 13},
     {200, 500, 250, 15},
     {250, 700, 300, 17},
     {300, 1000, 350, 20},
     {350, 1300, 400, 25},
     {400, 1500, 500, 30},
     {500, 2000, 1000, 50}   
  };

  int num_iter = 12;
  for (start_iter = 0; i < end_iter; i++) 
  {
    testSize = grid_params[i][0];

    for (start_iter = 0; j < end_iter; j++)
    {
      trainSize = grid_params[j][1];

      for (start_iter = 0; k < end_iter; k++)
      {
        window_size = grid_params[k][2];

        for (start_iter = 0; l < end_iter; l++)
        {
          n_feat = grid_params[l][3];

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

          cudaDeviceProp deviceProp = getDevProp(0);

          run_benchmark(trainSize, testSize, 512, window_size, n_feat, deviceProp, 
            trainLabels, testLabels, train_set, test_set, d_train, d_test, d_Out, h_Out, 0); 

          run_benchmark(trainSize, testSize, 512, window_size, n_feat, deviceProp, 
            trainLabels, testLabels, train_set, test_set, d_train, d_test, d_Out, h_Out, 1);


         // /*--------------------- Rotation Invariant ---------------------*/
          trainBytes = 2 * trainSize * window_size * n_feat * sizeof(float);

          /* HOST MEMORY ALLOCATION */
          h_malloc(&train_set, &test_set, &trainLabels, &testLabels, trainSize, testSize, window_size, n_feat, 2);

          /* DEVICE MEMORY ALLOCATION */
          cudaMalloc((void **)&d_Out, trainSize * window_size * sizeof(float));
          cudaMemset(d_Out, 0, trainSize * window_size * sizeof(float));
          h_Out = (float *)malloc(trainSize * window_size * sizeof(float));
          memset(h_Out, 0, trainSize * window_size * sizeof(float));
          cudaMalloc((void **)&d_train, trainBytes);
          cudaMemcpy(d_train, train_set, trainBytes, cudaMemcpyHostToDevice);
          cudaMalloc((void **)&d_test, n_feat * window_size * sizeof(float));
          /* DEVICE MEMORY ALLOCATION */


          run_benchmark(trainSize, testSize, 512, window_size, n_feat, deviceProp, 
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