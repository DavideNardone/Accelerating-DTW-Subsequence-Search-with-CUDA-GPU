#include "../include/header.h"

using namespace std;

/* ******************************************* MAIN
 * ******************************************* */
int main(int argc, char **argv) {
  struct timeval stop_CPU, start_CPU;
  cudaEvent_t start_GPU, stop_GPU;

  float time_GPU_MD_DTW_D, time_GPU_MD_DTW_I, time_GPU_rMDTW, elapsed=0.0;

  int i, j, f, nss;
  int t_size = 0, q_size = 0;

  int read_mode = 0;
  int window_size = 0;
  int dataSize = 0;
  int trainSize = 0, testSize = 0;
  int blockSize = 0;
  int k_fold = 0;
  int flag_shuffle = 0;
  int n_feat = 0;
  int device = 0;

  int num_opts;
  int flag_task = 0;
  int flag_in = 0;
  int flag_file = 0;
  int flag_opt = 0;
  int flag_device = 0;
  int flag_cross = 0;
  int flag_alg_mode = 0;
  int flag_verbose = 0;
  int verbose_mode = 1; // by default display all the outputs
  int n_file = 0;
  int class_mode = 0;
  char *task = NULL, *compution_type = NULL, *distance_type = NULL;
  const char *strategy;
  int *arr_num_file = NULL;
  int *tInd = NULL;
  struct data data_struct;

  int err = 0, errNR = 0;
  float RIA = 0.0f, RA = 0.0f, ER_RIA = 0.0f, ER_RA = 0.0f;

  time_t t;
  /* Intializes random number generator */
  srand((unsigned)time(&t));

  /* ******************************************* ARGUMENT PARSING
   * ******************************************* */
  for (i = 1; i < argc; i++) {

    if (flag_task) {
      num_opts = 1;

      task = argv[i];

      j = 0;
      do {
        task[j] = toupper(task[j]); // uppercase string
      } while (task[j++]);

      if (!checkFlagOpts(argv, argc, i, num_opts)) {
        printf("The number of options is incorrect. For more information run: "
               "%s --help\n",
               argv[0]);
        exit(-1);
      }

      flag_task = 0;
    } else if (flag_in) {

      compution_type = argv[i];

      j = 0;
      do {
        compution_type[j] = toupper(compution_type[j]); // uppercase string
      } while (compution_type[j++]);

      if (strcmp(compution_type, "GPU") == 0) {
        num_opts = 3;

        if (!checkFlagOpts(argv, argc, i, num_opts)) {
          printf("The number of options is incorrect. For more information "
                 "run: %s --help\n",
                 argv[0]);
          exit(-1);
        }

        n_feat = atoi(argv[i + 1]);
        blockSize = atoi(argv[i + 2]);
        read_mode = atoi(argv[i + 3]);

        i = i + 3;
      } else if ((strcmp(compution_type, "CPU") == 0)) {
        num_opts = 2;

        if (!checkFlagOpts(argv, argc, i, num_opts)) {
          printf("The number of options is incorrect. For more information run "
                 "the execution as: %s --help\n",
                 argv[0]);
          exit(-1);
        }

        n_feat = atoi(argv[i + 1]);
        read_mode = atoi(argv[i + 2]);
        i = i + 2;
      } else {
        printf("The number of options is incorrect. For more information run "
               "the execution as: %s --help\n",
               argv[0]);
        exit(-1);
      }

      flag_in = 0;
    } else if (flag_file) {
      if (strcmp(task, "CLASSIFICATION") == 0) {

        if (read_mode == 0 || read_mode == 2) {
          n_file = 2;
          num_opts = n_file;

          if (!checkFlagOpts(argv, argc, i, num_opts)) {
            printf("The number of options is incorrect. For more information "
                   "run the execution as: %s --help\n",
                   argv[0]);
            exit(-1);
          }
        } else if (read_mode == 1) {
          n_file = n_feat;
          num_opts = n_feat;

          if (!checkFlagOpts(argv, argc, i, num_opts)) {
            printf("The number of options is incorrect. For more information "
                   "run the execution as: %s --help\n",
                   argv[0]);
            exit(-1);
          }
        } else {
          printf("The number of options is incorrect. For more information run "
                 "the execution as: %s --help\n",
                 argv[0]);
          exit(-1);
        }
      } else if (strcmp(task, "SUBSEQ_SEARCH") == 0) {
        n_file = 2;
        num_opts = n_file;

        if (!checkFlagOpts(argv, argc, i, num_opts)) {
          printf("The number of options is incorrect. For more information run "
                 "the execution as: %s --help\n",
                 argv[0]);
          exit(-1);
        }
      }

      arr_num_file = (int *)malloc(n_file * sizeof(int));

      int j = 0;
      int cc = n_file;
      while (cc > 0) {
        arr_num_file[j] = i;

        i++;
        j++;
        cc--;
      }
      i--;

      flag_file = 0;
    } else if (flag_cross) {
      num_opts = 2;

      if (!checkFlagOpts(argv, argc, i, num_opts)) {
        printf("The number of options is incorrect. For more information run "
               "the execution as: %s --help\n",
               argv[0]);
        exit(-1);
      }

      k_fold = atoi(argv[i]);
      flag_shuffle = atoi(argv[i + 1]);
      if (k_fold < 2) {
        printf("It's not possible to perform %d-fold-cross validation! The "
               "number of folds has to be greater than 2.\n",
               k_fold);
        exit(-1);
      }
      i += 1;

      flag_cross = 0;
    } else if (flag_opt) {

      if (strcmp(task, "CLASSIFICATION") == 0) {
        num_opts = 3;
        if (k_fold > 0) {
          dataSize = atoi(argv[i]);
          data_struct.tot_size = dataSize;
          data_struct.train_size = 0;
          data_struct.test_size = 0;
          window_size = atoi(argv[i + 1]);
          i = i + 1;
        } else {
          if (!checkFlagOpts(argv, argc, i, num_opts)) {
            printf("The number of options is incorrect. For more information "
                   "run the execution as: %s --help\n",
                   argv[0]);
            exit(-1);
          }

          trainSize = atoi(argv[i]);
          testSize = atoi(argv[i + 1]);

          data_struct.train_size = trainSize;
          data_struct.test_size = testSize;
          dataSize = trainSize + testSize;
          data_struct.tot_size = dataSize;

          window_size = atoi(argv[i + 2]);

          i = i + 2;
        }

      } else if (strcmp(task, "SUBSEQ_SEARCH") == 0) {
        num_opts = 2;

        if (!checkFlagOpts(argv, argc, i, num_opts)) {
          printf("The number of options is incorrect. For more information run "
                 "the execution as: %s --help\n",
                 argv[0]);
          exit(-1);
        }
        t_size = atoi(argv[i]);
        q_size = atoi(argv[i + 1]);
        i = i + 1;
      } else {
        printf("The number of options is incorrect. For more information run "
               "the execution as: %s --help\n",
               argv[0]);
        exit(-1);
      }

      flag_opt = 0;
    }

    else if (flag_alg_mode) {
      num_opts = 2;

      if (!checkFlagOpts(argv, argc, i, num_opts)) {
        printf("The number of options is incorrect. For more information run "
               "the execution as: %s --help\n",
               argv[0]);
        exit(-1);
      }

      class_mode = atoi(argv[i]);
      if (class_mode == 0)
        strategy = "DEPENDENT";
      else if (class_mode == 1)
        strategy = "INDEPENDENT";
      else
        strategy = "ROTATION INVARIANT";

      distance_type = argv[i + 1];
      i = i + 1;

      flag_alg_mode = 0;
    } else if (flag_device) {
      num_opts = 1;

      if (!checkFlagOpts(argv, argc, i, num_opts)) {
        printf("The number of options is incorrect. For more information run "
               "the execution as: %s --help\n",
               argv[0]);
        exit(-1);
      }

      device = atoi(argv[i]);

      cudaSetDevice(device);

      flag_device = 0;
    } else if (flag_verbose) {
      num_opts = 1;

      if (!checkFlagOpts(argv, argc, i, num_opts)) {
        printf("The number of options is incorrect. For more information run "
               "the execution as: %s --help\n",
               argv[0]);
        exit(-1);
      }

      verbose_mode = atoi(argv[i]);

      flag_verbose = 0;

    } else if (!strcmp(argv[i], "-t"))
      flag_task = 1;
    else if (!strcmp(argv[i], "-i"))
      flag_in = 1;
    else if (!strcmp(argv[i], "-f"))
      flag_file = 1;
    else if (!strcmp(argv[i], "-o"))
      flag_opt = 1;
    else if (!strcmp(argv[i], "-k"))
      flag_cross = 1;
    else if (!strcmp(argv[i], "-m"))
      flag_alg_mode = 1;
    else if (!strcmp(argv[i], "-d"))
      flag_device = 1;
    else if (!strcmp(argv[i], "-v"))
      flag_verbose = 1;
    else if (!strcmp(argv[i], "--help"))
      print_help();
    else if (!strcmp(argv[i], "--version"))
      print_version();
    else if (!strcmp(argv[i], "--infoDevice"))
      infoDev();
    else {
      printf("The number of options is incorrect. For more information run the "
             "execution as: %s --help\n",
             argv[0]);
      exit(-1);
    }
  }
  /* ******************************************* ARGUMENT PARSING
   * ******************************************* */

  cudaDeviceProp deviceProp = getDevProp(device);
  checkGPU_prop(compution_type, deviceProp, "maxThreadsPerBlock", blockSize);

  if(verbose_mode == 0){
    printf("\nThe number of iteration is greater than testSize! "
      "Verbose mode will be suppressed for this run\n");
  }

  if (strcmp(task, "CLASSIFICATION") == 0) {

    /* ***** VARIABLE WORKSPACE FOR CLASSIFICATION TASK ***** */
    unsigned long long int dataBytes =
        2 * dataSize * window_size * n_feat * sizeof(float);
    int *dataLabels = (int *)malloc(dataSize * sizeof(int));
    float *data = (float *)malloc(dataBytes);
    // int trainSize, testSize;
    float mean_RIA = 0.0f, mean_RA = 0.0f, mean_ER_RIA = 0.0f,
          mean_ER_RA = 0.0f;

    float exec_time_mean = 0.0f;

    printf("Reading data...\n");
    printf("Dataset size: [%d,%d,%d]\n", dataSize, window_size, n_feat);
    if(k_fold < 1){
        printf("\tTrain set size: [%d,%d,%d]\n", trainSize, window_size, n_feat);
        printf("\tTest set size: [%d,%d,%d]\n\n", testSize, window_size, n_feat);
    }
    printf("\nClassification w/ %s-%s using " "%s\n\n", strategy, distance_type, compution_type);

    readFile(argv, arr_num_file, n_file, read_mode, data, data_struct,
             window_size, dataLabels, n_feat, class_mode);

    if (k_fold < 1) // not doing K-cross validation
      k_fold = 1; // (work around to do not re-write a lot of code)
    else
      tInd = crossvalind_Kfold(dataLabels, dataSize, k_fold, flag_shuffle);

    // Setting all the variables for each k-th fold
    for (f = 0; f < k_fold; f++) {

      err = 0;
      errNR = 0;

      if (k_fold > 1) { // doing K-fold cross validation
        testSize = countVal(tInd, dataSize, f);
        trainSize = dataSize - testSize;
      }

      /* *************** HOST MEMORY ALLOCATION *************** */
      unsigned long long int testBytes =
          testSize * window_size * n_feat * sizeof(float);
      unsigned long long int trainBytes;

      if (class_mode < 2)
        trainBytes = trainSize * window_size * n_feat * sizeof(float);
      else
        trainBytes = 2 * trainSize * window_size * n_feat * sizeof(float);

      float *h_train = (float *)malloc(trainBytes);
      float *h_test = (float *)malloc(testBytes);
      float *h_Out = 0;

      int *trainLabels = (int *)malloc(trainSize * sizeof(int));
      int *testLabels = (int *)malloc(testSize * sizeof(int));

      createTrainingTestingSet(data, dataLabels, dataSize, window_size, n_feat,
                               h_train, trainLabels, trainSize, h_test,
                               testLabels, testSize, tInd, f, class_mode);
      /* *************** HOST MEMORY ALLOCATION *************** */

      if(verbose_mode > 0){
        printf("Running %d/%d fold...\n", f+1, k_fold);
        printf("\tTrain set size: [%d,%d,%d] = %llu bytes\n", trainSize, window_size, n_feat, trainBytes);
        printf("\tTest set size: [%d,%d,%d] = %llu bytes\n\n", testSize, window_size, n_feat, testBytes);
      }

      if (strcmp(compution_type, "CPU") == 0) {

        switch (class_mode) {

          case 0: // MD_DTW_D
          {
            gettimeofday(&start_CPU, NULL);

            err = MDD_SIM_MES_CPU(trainSize, testSize, trainLabels, testLabels, h_train, h_test, window_size, n_feat, distance_type, verbose_mode);

            gettimeofday(&stop_CPU, NULL);

            elapsed = timedifference_msec(start_CPU, stop_CPU);
            RA = (float)(testSize - err) * (100.0 / testSize);
            ER_RA = (float)(testSize - (testSize - err)) / (testSize);

            if(verbose_mode > 0){
              printf("\n\tExecution time: %f ms\n", elapsed);
              printf("\tRegular Accuracy: %f\n", RA);
              printf("\tError rate: %f\n\n", ER_RA*100);
            }
            mean_RA += RA;
            mean_ER_RA += ER_RA;
            exec_time_mean += elapsed;
          } break;

          case 1: // MD_DTW_I
          {
            gettimeofday(&start_CPU, NULL);

            err = MDI_SIM_MES_CPU(trainSize, testSize, trainLabels, testLabels, h_train, h_test, window_size, n_feat, distance_type, verbose_mode);

            gettimeofday(&stop_CPU, NULL);

            elapsed = timedifference_msec(start_CPU, stop_CPU);
            RA = (float)(testSize - err) * (100.0 / testSize);
            ER_RA = (float)(testSize - (testSize - err)) / (testSize);

            if(verbose_mode > 0){
              printf("\n\tExecution time: %f ms\n", elapsed);
              printf("\tRegular Accuracy: %f\n", RA);
              printf("\tError rate: %f\n\n", ER_RA*100);
            }
            mean_RA += RA;
            mean_ER_RA += ER_RA;
            exec_time_mean += elapsed;            
          } break;

          case 2: // MD_RDTW_I
          {
            gettimeofday(&start_CPU, NULL);

            MDR_SIM_MES_CPU(trainSize, testSize, trainLabels,  testLabels, h_train, h_test, window_size, n_feat, distance_type, verbose_mode, &err, &errNR);

            gettimeofday(&stop_CPU, NULL);

            elapsed = timedifference_msec(start_CPU, stop_CPU);
            RIA = (float)(testSize - err) * (100.0 / testSize);
            ER_RIA = (float)(testSize - (testSize - err)) / (testSize);
            RA = (float)(testSize - errNR) * (100.0 / testSize);
            ER_RA = (float)(testSize - (testSize - errNR)) / (testSize);

            if(verbose_mode > 0){
              printf("\n\tNRI Execution time: %f ms\n", elapsed);
              printf("\tNRI Regular Accuracy: %f\n", RA);
              printf("\tNRI Error rate: %f\n\n", ER_RA*100);
              printf("\tRI Accuracy: %f\n", RIA);
              printf("\tRI Error rate: %f\n\n", ER_RIA*100);
            }
            mean_RIA += RIA;
            mean_ER_RIA += ER_RIA;
            mean_RA += RA;
            mean_ER_RA += ER_RA;
            exec_time_mean += elapsed;
          } break;

          default: printf("Error algorithm choice\n");
        }
      } else if (strcmp(compution_type, "GPU") == 0) {

        /* *************** DEVICE MEMORY ALLOCATION *************** */
        float *d_train = 0;
        cudaMalloc((void **)&d_train, trainBytes);
        cudaMemcpy(d_train, h_train, trainBytes, cudaMemcpyHostToDevice);

        float *d_test = 0;
        cudaMalloc((void **)&d_test, n_feat * window_size * sizeof(float));

        float *d_Out = 0;
        if (class_mode < 2) {
          cudaMalloc((void **)&d_Out, trainSize * sizeof(float));
          cudaMemset(d_Out, 0, trainSize * sizeof(float));
          h_Out = (float *)malloc(trainSize * sizeof(float));
          memset(h_Out, 0, trainSize * sizeof(float));
        } else {
          cudaMalloc((void **)&d_Out, trainSize * window_size * sizeof(float));
          cudaMemset(d_Out, 0, trainSize * window_size * sizeof(float));
          h_Out = (float *)malloc(trainSize * window_size * sizeof(float));
          memset(h_Out, 0, trainSize * window_size * sizeof(float));
        }
        /* *************** DEVICE MEMORY ALLOCATION *************** */

        switch (class_mode) {

          case 0: // MD_DTW_D
          {
            cudaEventCreate(&start_GPU);
            cudaEventCreate(&stop_GPU);
            cudaEventRecord(start_GPU, 0);

            int err = MDD_SIM_MES_GPU(trainSize, testSize, trainLabels, testLabels, h_train, h_test, d_train, d_test, d_Out, h_Out, window_size, n_feat, blockSize, deviceProp, distance_type, verbose_mode);

            cudaEventRecord(stop_GPU, 0);
            cudaEventSynchronize(stop_GPU);
            cudaEventElapsedTime(&time_GPU_MD_DTW_D, start_GPU, stop_GPU);
            cudaEventDestroy(start_GPU);
            cudaEventDestroy(stop_GPU);

            RA = (float)(testSize - err) * (100.0 / testSize);
            ER_RA = (float)(testSize - (testSize - err)) / (testSize);

            if(verbose_mode > 0){
              printf("\n\tExecution time: %f ms\n", time_GPU_MD_DTW_D);
              printf("\tRegular Accuracy: %f\n", RA);
              printf("\tThe Error rate: %f\n\n", ER_RA*100);
            }
            mean_RA += RA;
            mean_ER_RA += ER_RA;
            exec_time_mean += time_GPU_MD_DTW_D;
          } break;

          case 1: // MD_DTW_I
          {
            cudaEventCreate(&start_GPU);
            cudaEventCreate(&stop_GPU);
            cudaEventRecord(start_GPU, 0);

            int err = MDI_SIM_MES_GPU(trainSize, testSize, trainLabels, testLabels, h_train, h_test, d_train, d_test, d_Out, h_Out, window_size, n_feat, blockSize, deviceProp, distance_type, verbose_mode);

            cudaEventRecord(stop_GPU, 0);
            cudaEventSynchronize(stop_GPU);
            cudaEventElapsedTime(&time_GPU_MD_DTW_I, start_GPU, stop_GPU);
            cudaEventDestroy(start_GPU);
            cudaEventDestroy(stop_GPU);

            RA = (float)(testSize - err) * (100.0 / testSize);
            ER_RA = (float)(testSize - (testSize - err)) / (testSize);

            if(verbose_mode > 0){
              printf("\n\tExecution time: %f ms\n", time_GPU_MD_DTW_I);
              printf("\tRegular Accuracy: %f\n", RA);
              printf("\tError rate: %f\n\n", ER_RA*100);
            }
            mean_RA += RA;
            mean_ER_RA += ER_RA;
            exec_time_mean += time_GPU_MD_DTW_I;
          } break;

          case 2: // MD_RDTW_I
          {
            cudaEventCreate(&start_GPU);
            cudaEventCreate(&stop_GPU);
            cudaEventRecord(start_GPU, 0);

            MDR_SIM_MES_GPU(trainSize, testSize, trainLabels, testLabels, h_train, h_test, d_train, d_test, d_Out, h_Out, window_size, n_feat, blockSize, deviceProp, distance_type, verbose_mode, &err, &errNR);

            cudaEventRecord(stop_GPU, 0);
            cudaEventSynchronize(stop_GPU);
            cudaEventElapsedTime(&time_GPU_rMDTW, start_GPU, stop_GPU);
            cudaEventDestroy(start_GPU);
            cudaEventDestroy(stop_GPU);

            RIA = (float)(testSize - err) * (100.0 / testSize);
            ER_RIA = (float)(testSize - (testSize - err)) / (testSize);
            RA = (float)(testSize - errNR) * (100.0 / testSize);
            ER_RA = (float)(testSize - (testSize - errNR)) / (testSize);

            if(verbose_mode > 0){
              printf("\n\tNRI Execution time: %f ms\n", time_GPU_rMDTW);
              printf("\tNRI Regular Accuracy: %f\n", RA);
              printf("\tNRI Error rate: %f\n\n", ER_RA*100);
              printf("\tRI Accuracy: %f\n", RIA);
              printf("\tRI Error rate: %f\n\n", ER_RIA*100);
            }
            mean_RIA += RIA;
            mean_ER_RIA += ER_RIA;
            mean_RA += RA;
            mean_ER_RA += ER_RA;
            exec_time_mean += time_GPU_rMDTW;
          } break;

          default: printf("Error algorithm choice\n");
        }
        cudaFree(d_train);
        cudaFree(d_Out);
        cudaFree(d_Out);
      }
      free(h_train);
      free(h_test);
      free(h_Out);
    }
    if (class_mode < 2) {
      mean_RA /= k_fold;
      mean_ER_RA /= k_fold;
      printf("\nNRI Regular Accuracy mean: %f\n", mean_RA);
      printf("NRI Error rate mean: %f\n\n", mean_ER_RA*100);
      printf("NRI Execution time mean: %f ms\n", exec_time_mean);
    } else {
      mean_RIA /= k_fold;
      mean_ER_RIA /= k_fold;
      printf("\nRI Accuracy mean: %f\n", mean_RIA);
      printf("RI Error rate mean: %f\n\n", mean_ER_RIA*100);
      printf("RI Execution time mean: %f ms\n", exec_time_mean);
    }
  } else if (strcmp(task, "SUBSEQ_SEARCH") == 0) {

    nss = t_size - q_size + 1;
    // window_size = q_size;

    unsigned long long int t_bytes = t_size * n_feat * sizeof(float);
    unsigned long long int q_bytes = q_size * n_feat * sizeof(float);

    /* *************** CPU MEMORY ALLOCATION *************** */
    float *t_series = (float *)malloc(t_bytes);
    float *q_series = (float *)malloc(q_bytes);
    float *owp = (float *)malloc(nss * sizeof(float));
    memset(owp, 0, nss * sizeof(float));

    printf("Reading data...\n");
    printf("Number of Subsequences to search: %d\n", nss);
    printf("Time Series T: [%d,%d]\n", t_size, n_feat);
    printf("Time Series Q: [%d,%d]\n", q_size, n_feat);
    printf("Subsequence Search w/ %s-%s using " "%s\n\n", strategy, distance_type, compution_type);

    readFileSubSeq(argv, arr_num_file, n_file, t_series, t_size, q_series,
                   q_size, n_feat, read_mode);

    float min = 9999.99;
    int ind_min_val = 0;

    if (strcmp(compution_type, "CPU") == 0) {

      switch (class_mode) {

        case 0: // MD_DTW_D
        {
          gettimeofday(&start_CPU, NULL);

          elapsed = timedifference_msec(start_CPU, stop_CPU);

          min = MDD_SIM_MES_CPU(nss, t_series, q_series, t_size, q_size, n_feat, distance_type, verbose_mode, owp, &ind_min_val);

          printf("\tMin. index value %d, min. value: %f\n\n", ind_min_val, min);
          printf("\n\tExecution time: %f ms\n", elapsed);
        } break;

        case 1: // MD_DTW_I
        {
          gettimeofday(&start_CPU, NULL);

          min =  MDI_SIM_MES_CPU(nss, t_series, q_series, t_size, q_size, n_feat, distance_type, verbose_mode, owp, &ind_min_val);

          gettimeofday(&stop_CPU, NULL);

          printf("\tMin. index value %d, min. value: %f\n\n", ind_min_val, min);
          printf("\n\tExecution time: %f ms\n", elapsed);
        } break;

        default:
          printf("Error algorithm choice\n");
      }
    } else { // GPU computation

      /* *************** DEVICE MEMORY ALLOCATION *************** */
      float *d_t_series = 0, *d_owp = 0, *d_q_series = 0;
      cudaMalloc((void **)&d_t_series, t_bytes);
      cudaMemcpy(d_t_series, t_series, t_bytes, cudaMemcpyHostToDevice);

      cudaMalloc((void **)&d_q_series, q_bytes);
      cudaMemcpy(d_q_series, q_series, q_bytes, cudaMemcpyHostToDevice);

      cudaMalloc((void **)&d_owp, nss * sizeof(float));
      cudaMemset(d_owp, 0, nss * sizeof(float));
      /* *************** DEVICE MEMORY ALLOCATION *************** */

      cudaSetDevice(device);

      cudaDeviceProp deviceProp;
      deviceProp = getDevProp(device);

      switch (class_mode) {

        case 0: // MD_DTW_D
        {
          cudaEventCreate(&start_GPU);
          cudaEventCreate(&stop_GPU);
          cudaEventRecord(start_GPU, 0);

          min = MDD_SIM_MES_GPU(nss, d_t_series, d_q_series, t_size, q_size, n_feat, blockSize, deviceProp, distance_type, verbose_mode, owp, d_owp, &ind_min_val);

          cudaEventRecord(stop_GPU, 0);
          cudaEventSynchronize(stop_GPU);
          cudaEventElapsedTime(&time_GPU_MD_DTW_D, start_GPU, stop_GPU);
          cudaEventDestroy(start_GPU);
          cudaEventDestroy(stop_GPU);

          printf("\n\tExecution time:  %f ms\n", time_GPU_MD_DTW_D);
          printf("\tMin. index value %d, min. value: %f\n\n", ind_min_val, min);
        } break;

        case 1: // MD_DTW_I
        {
          cudaEventCreate(&start_GPU);
          cudaEventCreate(&stop_GPU);
          cudaEventRecord(start_GPU, 0);

          //TODO: change the order of the parameters: ..... d_q_series, t_size, q_size, n_feat ...
          min = MDI_SIM_MES_GPU(nss, d_t_series, d_q_series, t_size, q_size, n_feat, blockSize, deviceProp, distance_type,  verbose_mode, owp, d_owp, &ind_min_val);

          cudaThreadSynchronize();
          cudaEventRecord(stop_GPU, 0);
          cudaEventSynchronize(stop_GPU);
          cudaEventElapsedTime(&time_GPU_MD_DTW_I, start_GPU, stop_GPU);
          cudaEventDestroy(start_GPU);
          cudaEventDestroy(stop_GPU);

          printf("\tMin. index value %d, min. value: %f\n\n", ind_min_val, min);
          printf("\n\tExecution time: %f ms\n", time_GPU_MD_DTW_I);
        } break;

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