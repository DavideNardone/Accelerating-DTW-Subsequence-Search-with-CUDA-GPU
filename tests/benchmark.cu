#include "../include/header.h"

using namespace std;

int num_fig = 1;

void plot(float* cpu_D_MDTW, float* gpu_D_MDTW, float* cpu_I_MDTW, float* gpu_I_MDTW, int n_threads, float scf, char title[500], char dir_res[10], char suff_img_name[10]){

  // open persistent gnuplot window
  FILE* gnuplot_pipe = popen("gnuplot -persistent", "w");
  // basic settings
  float arr_min_max[4];
  int ind;
  arr_min_max[0] = max_arr(cpu_D_MDTW, n_threads, &ind);
  arr_min_max[1] = max_arr(gpu_D_MDTW, n_threads, &ind);
  arr_min_max[2] = max_arr(cpu_I_MDTW, n_threads, &ind);
  arr_min_max[3] = max_arr(gpu_I_MDTW, n_threads, &ind);  
  float max = max_arr(arr_min_max, 4, &ind);
  arr_min_max[0] = min_arr(cpu_D_MDTW, n_threads, &ind);
  arr_min_max[1] = min_arr(gpu_D_MDTW, n_threads, &ind);
  arr_min_max[2] = min_arr(cpu_I_MDTW, n_threads, &ind);
  arr_min_max[3] = min_arr(gpu_I_MDTW, n_threads, &ind); 
  float min = min_arr(arr_min_max, 4, &ind);
  fprintf(gnuplot_pipe, "set xrange [0:%d]\n", n_threads-1);
  fprintf(gnuplot_pipe, "set yrange [%f:%f]\n",min/2, 1.3*max);
  fprintf(gnuplot_pipe, "set log y\n");
  fprintf(gnuplot_pipe, "set title '%s'\n", title);
  fprintf(gnuplot_pipe, "set xtics ('2' 0, '4' 1, '8' 2,  '16' 3,  '32' 4,  '64' 5,  '128' 6,  '256' 7,  '512' 8,  '1024' 9)\n");
  fprintf(gnuplot_pipe, "set autoscale fix\n");

  //SAVE graph
  char append[5];
  char res_path[80] = "/home/davidenardone/MTSS/";
  strcat(res_path, dir_res);
  strcat(res_path, suff_img_name);
  fprintf(gnuplot_pipe, "set terminal png\n");
  fprintf(gnuplot_pipe, "set output '%s.png'\n", res_path);
  fprintf(gnuplot_pipe, "set xlabel '# of Threads'\n");
  fprintf(gnuplot_pipe, "set ylabel 'msec'\n");

  // fill it with data
  fprintf(gnuplot_pipe, "set multiplot\n");
  fprintf(gnuplot_pipe, "set key at %d,%f\n",n_threads-1, max - (1.5 + scf)*(max/4));
  fprintf(gnuplot_pipe, "plot '-' with linespoints title 'cpu_D_MDTW' pt 2 lt -1\n"); //black
  for(int i=0; i<n_threads; ++i)
    fprintf(gnuplot_pipe, "%d %f\n", i, cpu_D_MDTW[i]);
  fprintf(gnuplot_pipe, "e\n");

  fprintf(gnuplot_pipe, "set key at %d,%f\n",n_threads-1, max - (1.7 + scf)*(max/4));
  fprintf(gnuplot_pipe, "plot '-' with linespoints title 'gpu_D_MDTW' pt 3 lt 2\n");
  for(int i=0; i<n_threads; ++i)
    fprintf(gnuplot_pipe, "%d %f\n", i, gpu_D_MDTW[i]);
  fprintf(gnuplot_pipe, "e\n");

  fprintf(gnuplot_pipe, "set key at %d,%f\n",n_threads-1, max - (1.9 + scf)*(max/4));
  fprintf(gnuplot_pipe, "plot '-' with linespoints title 'cpu_I_MDTW' pt 4 lt 3\n");
  for(int i=0; i<n_threads; ++i)
    fprintf(gnuplot_pipe, "%d %f\n", i, cpu_I_MDTW[i]);
  fprintf(gnuplot_pipe, "e\n");

  fprintf(gnuplot_pipe, "set key at %d,%f\n",n_threads-1, max - (2.1 + scf)*(max/4));
  fprintf(gnuplot_pipe, "plot '-' with linespoints title 'gpu_I_MDTW' pt 5 lt 4\n");
  for(int i=0; i<n_threads; ++i)
    fprintf(gnuplot_pipe, "%d %f\n", i, gpu_I_MDTW[i]);
  fprintf(gnuplot_pipe, "e\n");

  // refresh can probably be omitted
  fprintf(gnuplot_pipe, "refresh\n");

  fflush(gnuplot_pipe);
  pclose(gnuplot_pipe);
}


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
  float *t_series, float *q_series, float *d_t_series, float *d_q_series, float *d_owp, float *owp, int task, float *time_cpu_ED, float *time_gpu_ED, float *time_cpu_MDTW, float *time_gpu_MDTW){

    int ind_min_val = 0;
    struct timeval stop_CPU, start_CPU;
    cudaEvent_t start_GPU, stop_GPU;
    char *distance_type[] = {"ED","DTW"};
    float time_cpu[2], time_gpu[2];

  switch(task){

    case 0:
      for (int i = 0; i < 2; ++i)
      {
        // printf("RUNNING BENCHMARK ON MD_D-%s...\n", *(distance_type + i));
        gettimeofday(&start_CPU, NULL);
        MDD_SIM_MES_CPU(nss, t_series, q_series, t_size, q_size, n_feat, *(distance_type + i), 0, owp, &ind_min_val);
        gettimeofday(&stop_CPU, NULL);
        time_cpu[i] = timedifference_msec(start_CPU, stop_CPU);

        cudaEventCreate(&start_GPU);
        cudaEventCreate(&stop_GPU);
        cudaEventRecord(start_GPU, 0);
        MDD_SIM_MES_GPU(nss, d_t_series, d_q_series, t_size, q_size, n_feat, blockSize, deviceProp, *(distance_type + i), 0, owp, d_owp, &ind_min_val);
        cudaEventRecord(stop_GPU, 0);
        cudaEventSynchronize(stop_GPU);
        cudaEventElapsedTime(&time_gpu[i], start_GPU, stop_GPU);
        cudaEventDestroy(start_GPU);
        cudaEventDestroy(stop_GPU);
        // printf("CPU %f ms vs GPU  %f ms\n", time_cpu, time_gpu);
      }
    break;
    case 1:
      for (int i = 0; i < 2; ++i)
      {
        // printf("RUNNING BENCHMARK ON MD_I-%s...\n", *(distance_type + i));
        gettimeofday(&start_CPU, NULL);
        MDI_SIM_MES_CPU(nss, t_series, q_series, t_size, q_size, n_feat, *(distance_type + i), 0, owp, &ind_min_val);
        gettimeofday(&stop_CPU, NULL);
        time_cpu[i] = timedifference_msec(start_CPU, stop_CPU);

        cudaEventCreate(&start_GPU);
        cudaEventCreate(&stop_GPU);
        cudaEventRecord(start_GPU, 0);
        MDD_SIM_MES_GPU(nss, d_t_series, d_q_series, t_size, q_size, n_feat, blockSize, deviceProp, *(distance_type + i), 0, owp, d_owp, &ind_min_val);
        cudaEventRecord(stop_GPU, 0);
        cudaEventSynchronize(stop_GPU);
        cudaEventElapsedTime(&time_gpu[i], start_GPU, stop_GPU);
        cudaEventDestroy(start_GPU);
        cudaEventDestroy(stop_GPU);
        // printf("CPU %f ms vs GPU  %f ms\n", time_cpu, time_gpu);
      }
    break;
  }

  *time_cpu_ED = time_cpu[0];
  *time_gpu_ED = time_gpu[0];
  *time_cpu_MDTW = time_cpu[1];
  *time_gpu_MDTW = time_gpu[1];
}

void run_benchmark(int trainSize, int testSize, int blockSize, int window_size, int n_feat, cudaDeviceProp deviceProp,  
  int *trainLabels, int *testLabels, float *train_set, float *test_set, float *d_train, float *d_test, float *d_Out, float *h_Out, int task, float *time_cpu_ED, float *time_gpu_ED, float *time_cpu_MDTW, float *time_gpu_MDTW){

  int ERR_CPU, ERR_GPU,ERR_NR_CPU,ERR_NR_GPU;
  struct timeval stop_CPU, start_CPU;
  cudaEvent_t start_GPU, stop_GPU;
  char *distance_type[] = {"ED","DTW"};
  float time_cpu[2], time_gpu[2];

  switch(task){

    //benchmark for Dependent-Similarity Measure Distance among CPU and GPU version
    case 0:
      for (int i = 0; i < 2; ++i)
      {
        // printf("RUNNING BENCHMARK ON MD_D-%s...\n", *(distance_type + i));
        gettimeofday(&start_CPU, NULL);
        ERR_CPU = MDD_SIM_MES_CPU(trainSize, testSize, trainLabels, testLabels, train_set, test_set, window_size, n_feat, *(distance_type + i), 0);
        gettimeofday(&stop_CPU, NULL);
        time_cpu[i] = timedifference_msec(start_CPU, stop_CPU);

        cudaEventCreate(&start_GPU);
        cudaEventCreate(&stop_GPU);
        cudaEventRecord(start_GPU, 0);
        ERR_GPU = MDD_SIM_MES_GPU(trainSize, testSize, trainLabels, testLabels, train_set, test_set, d_train, d_test, d_Out, h_Out, window_size, n_feat, 512, deviceProp, *(distance_type + i), 0);
        cudaEventRecord(stop_GPU, 0);
        cudaEventSynchronize(stop_GPU);
        cudaEventElapsedTime(&time_gpu[i], start_GPU, stop_GPU);
        cudaEventDestroy(start_GPU);
        cudaEventDestroy(stop_GPU);
        // printf("CPU %f ms vs GPU  %f ms\n", time_cpu[i], time_gpu[i]);
      }
    break;

    //benchmark for independent-Similarity Measure Distance among CPU and GPU version
    case 1:
      for (int i = 0; i < 2; ++i)
      {
        // printf("RUNNING BENCHMARK ON MD_I-%s...\n", *(distance_type + i));
        gettimeofday(&start_CPU, NULL);
        ERR_CPU = MDI_SIM_MES_CPU(trainSize, testSize, trainLabels, testLabels, train_set, test_set, window_size, n_feat, *(distance_type + i), 0);
        gettimeofday(&stop_CPU, NULL);
        time_cpu[i] = timedifference_msec(start_CPU, stop_CPU);

        cudaEventCreate(&start_GPU);
        cudaEventCreate(&stop_GPU);
        cudaEventRecord(start_GPU, 0);
        ERR_GPU = MDI_SIM_MES_GPU(trainSize, testSize, trainLabels, testLabels, train_set, test_set, d_train, d_test, d_Out, h_Out, window_size, n_feat, blockSize, deviceProp, *(distance_type + i), 0);
        cudaEventRecord(stop_GPU, 0);
        cudaEventSynchronize(stop_GPU);
        cudaEventElapsedTime(&time_gpu[i], start_GPU, stop_GPU);
        cudaEventDestroy(start_GPU);
        cudaEventDestroy(stop_GPU);
        // printf("CPU %f ms vs GPU  %f ms\n", time_cpu[i], time_gpu[i]);
        // printf("\n");
      }
    break;
    //benchmark for Rotation Dependent-Similarity Measure Distance among CPU and GPU version
    case 3:
      for (int i = 0; i < 2; ++i)
      {
        // printf("RUNNING BENCHMARK ON MDR-%s...\n", *(distance_type + i));
        gettimeofday(&start_CPU, NULL);
        MDR_SIM_MES_CPU(trainSize, testSize, trainLabels,  testLabels, train_set, test_set, window_size, n_feat, *(distance_type + i), 0, &ERR_CPU, &ERR_NR_CPU);
        gettimeofday(&stop_CPU, NULL);
        time_cpu[i] = timedifference_msec(start_CPU, stop_CPU);

        cudaEventCreate(&start_GPU);
        cudaEventCreate(&stop_GPU);
        cudaEventRecord(start_GPU, 0);
        MDR_SIM_MES_GPU(trainSize, testSize, trainLabels, testLabels, train_set, test_set, d_train, d_test, d_Out, h_Out, window_size, n_feat, blockSize, deviceProp, *(distance_type + i), 0, &ERR_GPU, &ERR_NR_GPU);
        cudaEventRecord(stop_GPU, 0);
        cudaEventSynchronize(stop_GPU);
        cudaEventElapsedTime(&time_gpu[i], start_GPU, stop_GPU);
        cudaEventDestroy(start_GPU);
        cudaEventDestroy(stop_GPU);
        // printf("CPU %f ms vs GPU  %f ms\n", time_cpu[i], time_gpu[i]);
        // printf("\n");
      }
    break;
  }

  *time_cpu_ED = time_cpu[0];
  *time_gpu_ED = time_gpu[0];
  *time_cpu_MDTW = time_cpu[1];
  *time_gpu_MDTW = time_gpu[1];
}


// nvcc -arch=sm_30 -Xcompiler "-O3 -Wall" ../src/module.o check_MD_DTW.o -lcheck -lm -lpthread -lrt -o unit_test
// nvcc -arch=sm_30 -Xcompiler "-O3 -Wall" ../src/module.cu benchmark.cu -lcheck -lm -lpthread -lrt -w -o benchmark
// nvcc -arch=sm_30 -Xcompiler "-O3 -Wall" -c check_MD_DTW.cu -o check_MD_DTW.o
int main(int argc, char **argv) {

  float *train_set = 0, *test_set = 0, *d_train = 0, *d_test = 0, *d_Out = 0, *h_Out = 0;
  int *trainLabels = 0, *testLabels = 0;

  int testSize = 0;
  int trainSize = 0;
  int window_size = 0;
  int n_feat = 0;
  int blockSize = 0;

  //SETTING PARAMETERS
  int start_iter = 7, end_iter = 10;
  int i=0, j=0, k=0, l=0, p=0;

  //[test size, train size, window size, n_feat] 
  int grid_params[10][4] = {
     {10, 100, 15, 1},
     {30, 200, 30, 3},
     {50, 250, 50, 5},
     {70, 300, 100, 7},
     {100, 350, 170, 10},
     {150, 400, 200, 13},
     {200, 500, 250, 50},
     {250, 700, 300, 100},
     {300, 1000, 350, 500},
     {350, 1300, 400, 1000},
  };

  int n_threads = 10;
  int arr_block_size_1[10] = {1, 2, 4, 8, 16, 32, 64, 128, 256, 1024};

  float *time_cpu_D_ED;
  float *time_gpu_D_ED;
  float *time_cpu_D_MDTW;
  float *time_gpu_D_MDTW;

  float *time_cpu_I_ED;
  float *time_gpu_I_ED;
  float *time_cpu_I_MDTW;
  float *time_gpu_I_MDTW;

  float *time_cpu_R_ED;
  float *time_gpu_R_ED;
  float *time_cpu_R_MDTW;
  float *time_gpu_R_MDTW;

  char suff_img_name[10];
  char title[500];

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

          time_cpu_D_ED = (float *)calloc(n_threads, sizeof(float));
          time_gpu_D_ED = (float *)calloc(n_threads, sizeof(float));
          time_cpu_D_MDTW = (float *)calloc(n_threads, sizeof(float));
          time_gpu_D_MDTW = (float *)calloc(n_threads, sizeof(float));

          time_cpu_I_ED = (float *)calloc(n_threads, sizeof(float));
          time_gpu_I_ED = (float *)calloc(n_threads, sizeof(float));
          time_cpu_I_MDTW = (float *)calloc(n_threads, sizeof(float));
          time_gpu_I_MDTW = (float *)calloc(n_threads, sizeof(float));

          // time_cpu_R_ED = (float *)calloc(n_threads, sizeof(float));
          // time_gpu_R_ED = (float *)scalloc(n_threads, sizeof(float));
          // time_cpu_R_MDTW = (float *)calloc(n_threads, sizeof(float));
          // time_gpu_R_MDTW = (float *)calloc(n_threads, sizeof(float));

          sprintf(suff_img_name,"%d.%d.%d",i+1,j+1,k+1);

          printf("Running benchmarks on classification task with: trainSize[%d], testSize[%d], window_size[%d], n_feat[%d]...", trainSize, testSize, window_size, n_feat);

          for (p = 0; p < n_threads; p++)
          {

            blockSize = arr_block_size_1[p];
            printf("%d.", blockSize);

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
              trainLabels, testLabels, train_set, test_set, d_train, d_test, d_Out, h_Out, 0, &time_cpu_D_ED[p], &time_gpu_D_ED[p], &time_cpu_D_MDTW[p], &time_gpu_D_MDTW[p]);

            run_benchmark(trainSize, testSize, blockSize, window_size, n_feat, deviceProp, 
              trainLabels, testLabels, train_set, test_set, d_train, d_test, d_Out, h_Out, 1, &time_cpu_I_ED[p], &time_gpu_I_ED[p], &time_cpu_I_MDTW[p], &time_gpu_I_MDTW[p]);

           // It's pointless to compute the execution time for the rotation invariant version
           // since its purpose is to improve the accuracy result for some scenarios! 

           // /*--------------------- Rotation Invariant ---------------------*/
            // trainBytes = 2 * trainSize * window_size * n_feat * sizeof(float);
            /* HOST MEMORY ALLOCATION */

            /* DEVICE MEMORY ALLOCATION */
            // cudaMalloc((void **)&d_Out, trainSize * window_size * sizeof(float));
            // cudaMemset(d_Out, 0, trainSize * window_size * sizeof(float));
            // h_Out = (float *)malloc(trainSize * window_size * sizeof(float));
            // memset(h_Out, 0, trainSize * window_size * sizeof(float));
            // cudaMalloc((void **)&d_train, trainBytes);
            // cudaMemcpy(d_train, train_set, trainBytes, cudaMemcpyHostToDevice);
            // cudaMalloc((void **)&d_test, n_feat * window_size * sizeof(float));
            // /* DEVICE MEMORY ALLOCATION */


            // run_benchmark(trainSize, testSize, blockSize, window_size, n_feat, deviceProp, 
            //   trainLabels, testLabels, train_set, test_set, d_train, d_test, d_Out, h_Out, 3, &time_cpu_R_ED[p], &time_gpu_R_ED[p], &time_cpu_R_MDTW[p], &time_gpu_R_MDTW[p]);

            h_free(&train_set, &test_set, &trainLabels, &testLabels, h_Out);

            cudaFree(d_train);
            cudaFree(d_test);
            cudaFree(d_Out);
          }
          printf("\n");
          sprintf(title, "Execution Time on [Tr:%d, Ts:%d, ws:%d, n_feats: %d]", trainSize, testSize, window_size, n_feat);
          plot(time_cpu_D_MDTW, time_gpu_D_MDTW, time_cpu_I_MDTW, time_gpu_I_MDTW, n_threads, 0.2, title, "res/cls/", suff_img_name);
        }
      }
    }
  }
  free(time_cpu_D_ED);
  free(time_gpu_D_ED);
  free(time_cpu_I_MDTW);
  free(time_gpu_I_MDTW);

  printf("DONE BENCHMARK ON CLASSIFICATION\n");

  ////////////////////////////////////// SUB-SEQ SEARCH //////////////////////////////////////
  int t_size = 0;
  int q_size = 0;
  int nss = 0;
  float *t_series = 0, *q_series = 0, *owp = 0;

  start_iter = 7;
  end_iter = 10;

  int grid_params_2[10][4] = {  
     {50, 10, 1},   
     {100, 75, 3},   
     {300, 100, 5},
     {500, 125, 7},
     {700, 150, 10},
     {800, 200, 13},
     {1000, 300, 50},
     {1200, 500, 100},
     {1300, 600, 300},
     {1800, 1000, 1024}
  };

  n_threads = 10;
  int arr_block_size_2[10] = {1, 2, 4, 8, 16, 32, 64, 128, 256, 1024};

  for (i = start_iter; i < end_iter; i++)
  {
    t_size = grid_params_2[i][0];

    for (j = start_iter; j < end_iter; j++)
    {
      q_size = grid_params_2[j][1];

      if( t_size < q_size)
        break;

      for (k = start_iter; k < end_iter; k++)
      {
        n_feat = grid_params_2[k][2];;

        time_cpu_D_ED = (float *)calloc(n_threads, sizeof(float));
        time_gpu_D_ED = (float *)calloc(n_threads, sizeof(float));
        time_cpu_D_MDTW = (float *)calloc(n_threads, sizeof(float));
        time_gpu_D_MDTW = (float *)calloc(n_threads, sizeof(float));

        time_cpu_I_ED = (float *)calloc(n_threads, sizeof(float));
        time_gpu_I_ED = (float *)calloc(n_threads, sizeof(float));
        time_cpu_I_MDTW = (float *)calloc(n_threads, sizeof(float));
        time_gpu_I_MDTW = (float *)calloc(n_threads, sizeof(float));

        sprintf(suff_img_name,"%d.%d.%d",i+1,j+1,k+1);

        printf("Running benchmarks on sub_sequence_search task with: blockSize[%d], t_size[%d], q_size[%d], nss[%d], n_feat[%d]...", blockSize, t_size, q_size, nss, n_feat);

        for (p = 0; p < n_threads; p++)
        {
          nss = t_size - q_size + 1;

          blockSize = arr_block_size_2[p];
          printf("%d.", blockSize);

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

          run_benchmark(nss, t_size, q_size, blockSize, n_feat, deviceProp, t_series, q_series, d_t_series, d_q_series, d_owp, owp, 0, &time_cpu_D_ED[p], &time_gpu_D_ED[p], &time_cpu_D_MDTW[p], &time_gpu_D_MDTW[p]);

          run_benchmark(nss, t_size, q_size, blockSize, n_feat, deviceProp, t_series, q_series, d_t_series, d_q_series, d_owp, owp, 0, &time_cpu_I_ED[p], &time_gpu_I_ED[p], &time_cpu_I_MDTW[p], &time_gpu_I_MDTW[p]);


          h_free(&t_series, &q_series, &owp);

          cudaFree(d_t_series);
          cudaFree(d_q_series);
          cudaFree(d_owp);
        }
        printf("\n");
        sprintf(title, "Execution Time on [t_size:%d, q_size:%d, nss:%d, n_feat: %d]", t_size, q_size, nss, n_feat);
        plot(time_cpu_D_MDTW, time_gpu_D_MDTW, time_cpu_I_MDTW, time_gpu_I_MDTW, n_threads, 0.3, title, "res/sub_seq/", suff_img_name);
      }
    }
  }
  free(time_cpu_D_ED);
  free(time_gpu_D_ED);
  free(time_cpu_I_MDTW);
  free(time_gpu_I_MDTW);

  printf("DONE BENCHMARK ON SUB_SEQUENCE_SEARCH\n");
}