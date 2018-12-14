#include <check.h>               
#include "../include/header.h"

using namespace std;

int test_cls = 0;
int test_sub_seq = 0;

START_TEST(test_classification_task) {

    int testSize = 50;
    int trainSize = 100;
    int window_size = 50;
    int n_feat = 3;

    unsigned long long int trainBytes;

    if(test_cls < 2)
      trainBytes = trainSize * window_size * n_feat * sizeof(float);
    else
      trainBytes = 2 * trainSize * window_size * n_feat * sizeof(float);

    unsigned long long int testBytes = testSize * window_size * n_feat * sizeof(float);

    float *train_set = (float *)malloc(trainBytes);
    float *test_set = (float *)malloc(testBytes);

    int *trainLabels = (int *)malloc(trainSize * sizeof(int));
    int *testLabels = (int *)malloc(testSize * sizeof(int));

    // random initialization of train data and label set
    initializeArray(train_set, trainSize * window_size * n_feat);
    initializeArray(test_set, testSize * window_size * n_feat);

    initializeArray(trainLabels, trainSize);
    initializeArray(testLabels, testSize);


    cudaDeviceProp deviceProp = getDevProp(0);

    /* *************** DEVICE MEMORY ALLOCATION *************** */
    float *d_train = 0;
    cudaMalloc((void **)&d_train, trainBytes);
    cudaMemcpy(d_train, train_set, trainBytes, cudaMemcpyHostToDevice);

    float *d_test = 0;
    float *h_Out = 0;
    cudaMalloc((void **)&d_test, n_feat * window_size * sizeof(float));

    float *d_Out = 0;
    if (test_cls < 2) {
      cudaMalloc((void **)&d_Out, trainSize * sizeof(float));
      cudaMemset(d_Out, 0, trainSize * sizeof(float));
      h_Out = (float *)malloc(trainSize * sizeof(float));
      memset(h_Out, 0, trainSize * sizeof(float));
    }
    else {
      cudaMalloc((void **)&d_Out, trainSize * window_size * sizeof(float));
      cudaMemset(d_Out, 0, trainSize * window_size * sizeof(float));
      h_Out = (float *)malloc(trainSize * window_size * sizeof(float));
      memset(h_Out, 0, trainSize * window_size * sizeof(float));
    }
    /* *************** DEVICE MEMORY ALLOCATION *************** */

    int ED_ERR_CPU, ED_ERR_NR_CPU, DTW_ERR_CPU, DTW_ERR_NR_CPU, ED_ERR_GPU, ED_ERR_NR_GPU, DTW_ERR_GPU, DTW_ERR_NR_GPU;
    char distance_type_1[] = "ED";
    char distance_type_2[] = "DTW";

    switch(test_cls) {

      case 0: { // testing MD_MES_DEPENDENT

        ED_ERR_CPU = MDD_SIM_MES_CPU(trainSize, testSize, trainLabels, testLabels, train_set, test_set, window_size, n_feat, distance_type_1, 0);
        ED_ERR_GPU = MDD_SIM_MES_GPU(trainSize, testSize, trainLabels, testLabels, train_set, test_set, d_train, d_test, d_Out, h_Out, window_size, n_feat, 512, deviceProp, distance_type_1, 0);

        DTW_ERR_CPU = MDD_SIM_MES_CPU(trainSize, testSize, trainLabels, testLabels, train_set, test_set, window_size, n_feat, distance_type_2, 0);
        DTW_ERR_GPU = MDD_SIM_MES_GPU(trainSize, testSize, trainLabels, testLabels, train_set, test_set, d_train, d_test, d_Out, h_Out, window_size, n_feat, 512, deviceProp, distance_type_2, 0);

      } break;

      case 1: { // testing MD_MES_INDEPENDENT

        ED_ERR_CPU = MDI_SIM_MES_CPU(trainSize, testSize, trainLabels, testLabels, train_set, test_set, window_size, n_feat, distance_type_1, 0);
        ED_ERR_GPU = MDI_SIM_MES_GPU(trainSize, testSize, trainLabels, testLabels, train_set, test_set, d_train, d_test, d_Out, h_Out, window_size, n_feat, 512, deviceProp, distance_type_1, 0);

        DTW_ERR_CPU = MDI_SIM_MES_CPU(trainSize, testSize, trainLabels, testLabels, train_set, test_set, window_size, n_feat, distance_type_2, 0);
        DTW_ERR_GPU = MDI_SIM_MES_GPU(trainSize, testSize, trainLabels, testLabels, train_set, test_set, d_train, d_test, d_Out, h_Out, window_size, n_feat, 512, deviceProp, distance_type_2, 0);

      }break;

      case 2: {

        MDR_SIM_MES_CPU(trainSize, testSize, trainLabels,  testLabels, train_set, test_set, window_size, n_feat, distance_type_1, 0, &ED_ERR_CPU, &ED_ERR_NR_CPU);
        MDR_SIM_MES_GPU(trainSize, testSize, trainLabels, testLabels, train_set, test_set, d_train, d_test, d_Out, h_Out, window_size, n_feat, 512, deviceProp, distance_type_1, 0, &ED_ERR_GPU, &ED_ERR_NR_GPU);

        MDR_SIM_MES_CPU(trainSize, testSize, trainLabels,  testLabels, train_set, test_set, window_size, n_feat, distance_type_1, 0, &DTW_ERR_CPU, &DTW_ERR_NR_CPU);
        MDR_SIM_MES_GPU(trainSize, testSize, trainLabels, testLabels, train_set, test_set, d_train, d_test, d_Out, h_Out, window_size, n_feat, 512, deviceProp, distance_type_1, 0, &DTW_ERR_GPU, &DTW_ERR_NR_GPU);   

      } break;

      default: printf("Error\n");
    }

    ck_assert_int_eq(ED_ERR_CPU, ED_ERR_GPU);
    ck_assert_int_eq(DTW_ERR_CPU, DTW_ERR_GPU);

    if(test_cls > 1){
      ck_assert_int_eq(ED_ERR_NR_CPU, ED_ERR_NR_GPU);
      ck_assert_int_eq(DTW_ERR_NR_CPU, DTW_ERR_NR_GPU);
    }

    free(train_set);
    free(test_set);
    free(trainLabels);
    free(testLabels);

    cudaFree(d_Out);
    cudaFree(d_test);

    test_cls++;
} END_TEST

START_TEST(test_sub_seq_task) {

    int t_size = 1000;
    int q_size = 10;
    int n_feat = 3;

    int nss = t_size - q_size + 1;

    unsigned long long int t_bytes = t_size * n_feat * sizeof(float);
    unsigned long long int q_bytes = q_size * n_feat * sizeof(float);

    /* *************** CPU MEMORY ALLOCATION *************** */
    float *t_series = (float *)malloc(t_bytes);
    float *q_series = (float *)malloc(q_bytes);
    float *owp = (float *)malloc(nss * sizeof(float));
    memset(owp, 0, nss * sizeof(float));

    // random initialization the two sequences
    initializeArray(t_series, t_size * n_feat);
    initializeArray(q_series, q_size * n_feat);

    cudaDeviceProp deviceProp = getDevProp(0);

    /* *************** DEVICE MEMORY ALLOCATION *************** */
    float *d_t_series = 0, *d_owp = 0, *d_q_series = 0;
    cudaMalloc((void **)&d_t_series, t_bytes);
    cudaMemcpy(d_t_series, t_series, t_bytes, cudaMemcpyHostToDevice);

    cudaMalloc((void **)&d_q_series, q_bytes);
    cudaMemcpy(d_q_series, q_series, q_bytes, cudaMemcpyHostToDevice);

    cudaMalloc((void **)&d_owp, nss * sizeof(float));
    cudaMemset(d_owp, 0, nss * sizeof(float));
    /* *************** DEVICE MEMORY ALLOCATION *************** */

    int ED_MIN_CPU = 0, DTW_MIN_CPU = 0, ED_MIN_GPU = 0, DTW_MIN_GPU = 0, ED_IND_MIN_VAL_CPU = 0, ED_IND_MIN_VAL_GPU = 0, DTW_IND_MIN_VAL_CPU = 0, DTW_IND_MIN_VAL_GPU = 0;
    char distance_type_1[] = "ED";
    char distance_type_2[] = "DTW";

    switch(test_sub_seq) {

      case 0:{

        ED_MIN_CPU = MDD_SIM_MES_CPU(nss, t_series, q_series, t_size, q_size, n_feat, distance_type_1, 0, owp, &ED_IND_MIN_VAL_CPU);
        ED_MIN_GPU = MDD_SIM_MES_GPU(nss, d_t_series, d_q_series, t_size, q_size, n_feat, 512, deviceProp, distance_type_1, 0, owp, d_owp, &ED_IND_MIN_VAL_GPU);

        DTW_MIN_CPU = MDD_SIM_MES_CPU(nss, t_series, q_series, t_size, q_size, n_feat, distance_type_2, 0, owp, &DTW_IND_MIN_VAL_CPU);
        DTW_MIN_GPU = MDD_SIM_MES_GPU(nss, d_t_series, d_q_series, t_size, q_size, n_feat, 512, deviceProp, distance_type_2, 0, owp, d_owp, &DTW_IND_MIN_VAL_GPU);

      }break;

      case 1:{

        ED_MIN_CPU = MDI_SIM_MES_CPU(nss, t_series, q_series, t_size, q_size, n_feat, distance_type_1, 0, owp, &ED_IND_MIN_VAL_CPU);
        ED_MIN_GPU = MDI_SIM_MES_GPU(nss, d_t_series, d_q_series, t_size, q_size, n_feat, 512, deviceProp, distance_type_1, 0, owp, d_owp, &ED_IND_MIN_VAL_GPU);

        DTW_MIN_CPU = MDI_SIM_MES_CPU(nss, t_series, q_series, t_size, q_size, n_feat, distance_type_2, 0, owp, &ED_IND_MIN_VAL_CPU);
        DTW_MIN_GPU = MDI_SIM_MES_GPU(nss, d_t_series, d_q_series, t_size, q_size, n_feat, 512, deviceProp, distance_type_2, 0, owp, d_owp, &ED_IND_MIN_VAL_GPU);

      }break;

      default: printf("Error\n");
    }

    ck_assert_int_eq(ED_MIN_CPU, ED_MIN_GPU);
    ck_assert_int_eq(DTW_MIN_CPU, DTW_MIN_GPU);

    free(t_series);
    free(q_series);
    free(owp);

    cudaFree(d_t_series);
    cudaFree(d_q_series);
    cudaFree(d_owp);

} END_TEST

Suite *classification_task(void) {
  Suite *s;
  TCase *tc_core;

  s = suite_create("classification_task");
  tc_core = tcase_create("Core");

  tcase_add_test(tc_core, test_classification_task);
  tcase_add_test(tc_core, test_classification_task);
  tcase_add_test(tc_core, test_classification_task);
  suite_add_tcase(s, tc_core);

  return s;
}

Suite *sub_sequence_task(void) {
  Suite *s;
  TCase *tc_core;

  s = suite_create("sub_sequence_task");
  tc_core = tcase_create("Core");

  tcase_add_test(tc_core, test_sub_seq_task);
  tcase_add_test(tc_core, test_sub_seq_task);
  tcase_add_test(tc_core, test_sub_seq_task);
  suite_add_tcase(s, tc_core);

  return s;
}

int main(void) {

  int no_failed = 0;                   
  Suite *s1,*s2; 
  SRunner *runner1,*runner2;                     

  s1 = classification_task();                   
  runner1 = srunner_create(s1);

  s2 = sub_sequence_task();          
  runner2 = srunner_create(s2); 

  //runner_1
  srunner_run_all(runner1, CK_NORMAL);  
  no_failed = srunner_ntests_failed(runner1); 
  srunner_free(runner1);

  //runner_2
  srunner_run_all(runner2, CK_NORMAL);  
  no_failed = srunner_ntests_failed(runner2); 
  srunner_free(runner2);

  return (no_failed == 0) ? EXIT_SUCCESS : EXIT_FAILURE;  
}