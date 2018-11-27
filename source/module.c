#include "../include/header.h"

using namespace std;


/**
 * \brief The callback function `printArrayDev` print all the elements contained into `array` on the std out.
 *
 * The following function print all the element contained into `array` on the std out of the CUDA device.
 * \param *array Vector of float elements to print
 * \param n The size of the vector
 */
__device__ void printArrayDev(float *array, int n) {

  int i;
  for (i = 0; i < n; i++)
    printf("arr[%d]: %f \n", i, array[i]);
}

/**
 * \brief The callback function `printMatrixDev` print all the elements contained into `matrix` on the std out.
 *
 * The following function print all the elements contained into `matrix` on the std out of the CUDA device.
 * \param *matrix Matrix of float elements to print
 * \param n The size of the matrix
 */
__device__ void printMatrixDev(float *matrix, int M, int N) {

  int i, j;
  for (i = 0; i < M; i++) {
    for (j = 0; j < N; j++)
      printf("%f ", matrix[i * N + j]);
    printf("\n");
  }
}

/**
 * \brief The callback function `stdDev` compute the `standard deviation` of a given vector allocated on the CUDA device.
 *
 * The following function computes the `standard deviation` of a given input vector.
 * \param *data Input vector
 * \param n Size of the vector
 * \param *avg `Mean` computed on the input vector
 * \return `Standard deviation` computed on the input vector
 */
__device__ float stdDev(float *data, int n, float *avg) {
  printf("N_SAMPLE: %d\n", n);
  printf("DATA_SIZE: %d\n", sizeof(data));
  float mean = 0.0, sum_deviation = 0.0;
  int i;
  for (i = 0; i < n; ++i) {
    mean += data[i];
  }
  mean = mean / n;
  *avg = mean;
  for (i = 0; i < n; ++i)
    sum_deviation += (data[i] - mean) * (data[i] - mean);

  return sqrt(sum_deviation / (n - 1));
}

/**
 * \brief The kernel function `MD_ED_D` computes the `Dependent-Multi Dimensional Euclidean` distance (D-MDE).
 *
 * The following kernel function computes the D-MDE taking advantage of the GPU, by using a specific number of threads for block.
    It considers the comparison of many Multivariate Time Series (MTS) stored into the unrolled vector `*S` against the only unrolled vector `*T`.
    By exploiting the CUDA threads, this computation can be done very fast.
    For more information about how it's computed, refer to the following link: http://stats.stackexchange.com/questions/184977/multivariate-time-series-euclidean-distance

 * \param *S Unrolled vector containing `trainSize` number of MTS 
 * \param *T Unrolled vector representing the second time Series to compare against `*S`
 * \param window_size Length of the two given MTS
 * \param dimensions Number of variables for the two MTS
 * \param *data_out Vector containing the results achieved by comparing `*T` against `*S`
 * \param *trainSize Number of MTS contained in the vector `T`
 * \param task Integer discriminating the task to perform (e.g., 0: CLASSIFICATION, 1:SUBSEQUENCE SEARCH)
 * \param gm Integer indicating where to store the unrolled vector `*T` (e.g., 0:shared memory, 1: global memory)
 */
__global__ void MD_ED_D(float *S, float *T, int window_size, int dimensions,
                        float *data_out, int trainSize, int task, int gm) {

  long long int i, j, p;
  float sumErr = 0, dd = 0;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (gm == 0) {
    extern __shared__ float T2[];

    int t, offset;
    if (task == 0) {

      offset = window_size;

      int wind = dimensions * WS;
      t = idx * wind;
      if ((idx * wind) + wind >
          trainSize * wind) // CHANGE FORMULA 120=train_size
        return;

      if (threadIdx.x == 0) {
        for (i = 0; i < dimensions; i++)
          for (j = 0; j < window_size; j++)
            T2[window_size * i + j] = T[window_size * i + j];
      }

      __syncthreads();
    } else {

      // in this case 'trainSize' is the number of subsequence to search 'nss',
      // that is, the length of dataset to perform on
      offset = trainSize;

      t = idx;
      if ((idx + WS) > trainSize)
        return;

      if (threadIdx.x == 0) {
        for (i = 0; i < dimensions; i++)
          for (j = 0; j < window_size; j++)
            T2[window_size * i + j] = T[window_size * i + j];
      }
      __syncthreads();
    }

    for (j = 0; j < window_size; j++) {
      dd = 0;
      for (p = 0; p < dimensions; p++)
        dd += (S[(t + p * offset) + j] - T2[(p * window_size) + j]) *
              (S[(t + p * offset) + j] - T2[(p * window_size) + j]);

      sumErr += dd;
    }
    data_out[idx] = sqrt(sumErr);
  } else {

    int t, offset;
    if (task == 0) {

      offset = window_size;

      int wind = dimensions * WS;
      t = idx * wind;
      if ((idx * wind) + wind > trainSize * wind)
        return;
    } else {

      // in this case 'trainSize' is the number of subsequence to search 'nss',
      // that is, the length of dataset to perform on
      offset = trainSize;

      t = idx;
      if ((idx + WS) > trainSize)
        return;
    }

    for (j = 0; j < window_size; j++) {
      dd = 0;
      for (p = 0; p < dimensions; p++)
        dd += (S[(t + p * offset) + j] - T[(p * window_size) + j]) *
              (S[(t + p * offset) + j] - T[(p * window_size) + j]);

      sumErr += dd;
    }
    data_out[idx] = sqrt(sumErr);
  }
}

/**
 * \brief The kernel function `MD_ED_I` computes the `Independent-Multi Dimensional Euclidean` distance (I-MDE).
 *
 * The following kernel function computes the I-MDE taking advantage of the GPU, by using a specific number of threads for block.
    It considers the comparison of many Multivariate Time Series (MTS) stored into the unrolled vector `*S` against the only unrolled vector `*T`.
    By exploiting the CUDA threads, this computation can be done very fast.
    For more information about how it's computed, refer to the following link: http://stats.stackexchange.com/questions/184977/multivariate-time-series-euclidean-distance

 * \param *S Unrolled vector containing `trainSize` number of MTS 
 * \param *T Unrolled vector representing the second time Series to compare against `*S`
 * \param window_size Length of the two given MTS
 * \param dimensions Nnumber of variables for the two MTS
 * \param *data_out Vector containing the results achieved by comparing `*T` against `*S`
 * \param *trainSize Number of MTS contained in the vector `T`
 * \param task Integer discriminating the task to perform (e.g., 0: CLASSIFICATION, 1:SUBSEQUENCE SEARCH)
 */
__global__ void MD_ED_I(float *S, float *T, int window_size, int dimensions,
                        float *data_out, int trainSize, int task) {

  int idx, offset_x;
  float sumErr = 0;
  long long int i, j;

  extern __shared__ float sh_mem[];

  float *T2 = (float *)sh_mem;
  float *DTW_single_dim =
      (float *)&sh_mem[dimensions * window_size]; // offset on the shared memory
                                                  // for the segment T2

  if (task == 0) {
    idx = threadIdx.x * dimensions + threadIdx.y;
    offset_x = ((blockDim.x * blockDim.y * window_size) * blockIdx.x) +
               idx * window_size;

    if (((blockDim.x * blockDim.y * blockIdx.x) + idx) >=
        trainSize * dimensions) // 120=train_size
      return;

  } else { // SUBSEQ_SEARCH

    idx = threadIdx.x * dimensions + threadIdx.y;
    offset_x =
        (blockDim.x * blockIdx.x) +
        ((threadIdx.y * trainSize) +
         threadIdx.x); // use blockIdx and other measure to set well the offset

    if ((idx + WS) > trainSize)
      return;
  }

  if (idx == 0) {
    for (i = 0; i < dimensions; i++)
      for (j = 0; j < window_size; j++)
        *(T2 + (window_size * i + j)) = T[window_size * i + j];
  }
  __syncthreads();

  for (j = 0; j < window_size; j++)
    sumErr += (S[offset_x + j] - T2[window_size * threadIdx.y + j]) *
              (S[offset_x + j] - T2[window_size * threadIdx.y + j]);

  DTW_single_dim[idx] = sqrt(sumErr);

  __syncthreads();

  if (idx == 0) {
    for (i = 0; i < blockDim.x; i++) {
      data_out[(blockIdx.x * blockDim.x) + i] = 0.0;
      for (j = 0; j < blockDim.y; j++) {
        data_out[(blockIdx.x * blockDim.x) + i] +=
            DTW_single_dim[i * dimensions + j]; // rivedere formula!
      }
    }
  }
}

/**
 * \brief The kernel function `rMD_ED_D` computes the `Rotation Dependent-Multi Dimensional Euclidean` distance (rD-MDE).
 *
 * The following kernel function computes the rD-MDE taking advantage of the GPU, by using a specific number of threads for block.
    It considers the comparison of all the possible `punctual rotation` of the Multivariate Time Series (MTS) stored into the unrolled vector `*S` against the only unrolled vector `*T`.
    By exploiting the CUDA threads, this computation can be done very fast.

 * \param *S Unrolled vector containing `trainSize` number of MTS 
 * \param *T Unrolled vector representing the second time Series to compare against `*S`
 * \param window_size Length of the two given MTS
 * \param dimensions Number of variables for the two MTS
 * \param *data_out Vector containing the results achieved by comparing `*T` against `*S`
 * \param *trainSize Number of MTS contained in the vector `T`
 * \param gm Integer indicating where to store the unrolled vector `*T` (e.g., 0:shared memory, 1: global memory)
 */
__global__ void rMD_ED_D(float *S, float *T, int window_size, int dimensions,
                         float *data_out, int trainSize, int gm) {

  long long int i, j, p;
  float sumErr = 0, dd = 0;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (gm == 0) {

    extern __shared__ float T2[];

    // offset training set
    int s = dimensions * 2 * WS * (idx / WS);
    int t = s + idx % WS;

    if (idx >= (trainSize * window_size)) //
      return;

    if (threadIdx.x == 0) {
      for (i = 0; i < dimensions; i++)
        for (j = 0; j < window_size; j++)
          T2[window_size * i + j] = T[window_size * i + j];
    }
    __syncthreads();

    for (j = 0; j < window_size; j++) {
      dd = 0;
      for (p = 0; p < dimensions; p++)
        dd += (S[(t + p * 2 * window_size) + j] - T2[(p * window_size) + j]) *
              (S[(t + p * 2 * window_size) + j] - T2[(p * window_size) + j]);

      sumErr += dd;
    }
    data_out[idx] = sqrt(sumErr);
  } else {

    int s = dimensions * 2 * WS * (idx / WS);
    int t = s + idx % WS;

    if (idx >= (trainSize * window_size))
      return;

    for (j = 0; j < window_size; j++) {
      dd = 0;
      for (p = 0; p < dimensions; p++)
        dd += (S[(t + p * 2 * window_size) + j] - T[(p * window_size) + j]) *
              (S[(t + p * 2 * window_size) + j] - T[(p * window_size) + j]);

      sumErr += dd;
    }
    data_out[idx] = sqrt(sumErr);
  }
}

/**
 * \brief The kernel function `MD_DTW_D` computes the `Dependent-Multi Dimensional Dynamic Time Warping` distance (D-MDDTW).
 *
 * The following kernel function computes the D-MDDTW taking advantage of the GPU, by using a specific number of threads for block.
    It considers the comparison of many Multivariate Time Series (MTS) stored into the unrolled vector `*S` against the only unrolled vector `*T`.
    By exploiting the CUDA threads, this computation can be done very fast.
    For more information about how it's computed, refer to the following link: http://stats.stackexchange.com/questions/184977/multivariate-time-series-euclidean-distance

 * \param *S Unrolled vector containing `trainSize` number of MTS 
 * \param *T Unrolled vector representing the second time Series to compare against `*S`
 * \param window_size Length of the two given MTS
 * \param dimensions Number of variables for the two MTS
 * \param *data_out Vector containing the results achieved by comparing `*T` against `*S`
 * \param *trainSize Number of MTS contained in the vector `T`
 * \param task Integer discriminating the task to perform (e.g., 0: CLASSIFICATION, 1:SUBSEQUENCE SEARCH)
 * \param gm Integer indicating where to store the unrolled vector `*T` (e.g., 0:shared memory, 1: global memory)
 */
__global__ void MD_DTW_D(float *S, float *T, int ns, int nt, int dimensions,
                         float *data_out, int trainSize, int task, int gm) {

  long long int k, l, g;

  long long int i, j, p;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  float min_nb = 0;
  float array[WS][2];

  if (gm == 0) {

    // query timeseries
    extern __shared__ float T2[];

    int t, offset;
    if (task == 0) {

      offset = ns;

      int wind = dimensions * WS;
      t = idx * wind;
      if ((idx * wind) + wind > trainSize * wind)
        return;

      if (threadIdx.x == 0) {
        for (i = 0; i < dimensions; i++)
          for (j = 0; j < nt; j++)
            T2[nt * i + j] = T[nt * i + j];
      }
      __syncthreads();
    } else {

      offset = trainSize;

      t = idx;
      if ((idx + WS) > trainSize)
        return;

      if (threadIdx.x == 0) {
        for (i = 0; i < dimensions; i++)
          for (j = 0; j < nt; j++)
            T2[nt * i + j] = T[nt * i + j];
      }
      __syncthreads();
    }

    k = 0;
    l = 1;

    // computing first row (instace versus query)
    for (i = 0; i < nt; i++) {
      array[i][k] = 0.0;
      for (p = 0; p < dimensions; p++) {
        if (i == 0)
          array[i][k] += pow((S[t + p * offset] - T2[p * nt]), 2);
        else
          array[i][k] += pow((S[t + p * offset] - T2[p * nt + i]), 2);
      }
      if (i != 0)
        array[i][k] += array[i - 1][k];
    }

    k = 1;
    l = 0;
    for (j = 1; j < ns; j++) {

      i = 0;
      array[i][k] = 0.0;

      for (p = 0; p < dimensions; p++)
        array[i][k] += pow((S[t + p * offset + j] - T2[p * nt + i]), 2);

      array[i][k] += array[i][l];
      for (i = 1; i < nt; i++) {

        array[i][k] = 0.0;
        float a = array[i - 1][l];
        float b = array[i][l];
        float c = array[i - 1][k];

        min_nb = fminf(a, b);
        min_nb = fminf(c, min_nb);

        for (p = 0; p < dimensions; p++)
          array[i][k] += pow((S[t + p * offset + j] - T2[p * nt + i]), 2);

        array[i][k] += min_nb;
      }
      g = k;
      k = l;
      l = g;
    }
    data_out[idx] = array[nt - 1][g];
  } else {

    int t, offset;
    if (task == 0) {

      offset = ns;

      int wind = dimensions * WS;
      t = idx * wind;
      if ((idx * wind) + wind > trainSize * wind)
        return;
    } else {

      offset = trainSize;

      t = idx;
      if ((idx + WS) > trainSize)
        return;
    }

    k = 0;
    l = 1;

    // computing first row (instace versus query)
    for (i = 0; i < nt; i++) {
      array[i][k] = 0.0;
      for (p = 0; p < dimensions; p++) {
        if (i == 0)
          array[i][k] += pow((S[t + p * offset] - T[p * nt]), 2);
        else
          array[i][k] += pow((S[t + p * offset] - T[p * nt + i]), 2);
      }
      if (i != 0)
        array[i][k] += array[i - 1][k];
    }

    k = 1;
    l = 0;
    for (j = 1; j < ns; j++) {

      i = 0;
      array[i][k] = 0.0;

      for (p = 0; p < dimensions; p++)
        array[i][k] += pow((S[t + p * offset + j] - T[p * nt + i]), 2);

      array[i][k] += array[i][l];
      for (i = 1; i < nt; i++) {

        array[i][k] = 0.0;
        float a = array[i - 1][l];
        float b = array[i][l];
        float c = array[i - 1][k];

        min_nb = fminf(a, b);
        min_nb = fminf(c, min_nb);

        for (p = 0; p < dimensions; p++)
          array[i][k] += pow((S[t + p * offset + j] - T[p * nt + i]), 2);

        array[i][k] += min_nb;
      }
      g = k;
      k = l;
      l = g;
    }
    data_out[idx] = array[nt - 1][g];
  }
}

/**
 * \brief The kernel function `MD_ED_I` computes the `Independent Multi Dimensional-Dynamic Time Warping` distance (I-MDDTW).
 *
 * The following kernel function computes the I-MDDTW taking advantage of the GPU, by using a specific number of threads for block.
    It considers the comparison of many Multivariate Time Series (MTS) stored into the unrolled vector `*S` against the only unrolled vector `*T`.
    By exploiting the CUDA threads, this computation can be done very fast.
    For more information about how it's computed, refer to the following link: http://stats.stackexchange.com/questions/184977/multivariate-time-series-euclidean-distance

 * \param *S Unrolled vector containing `trainSize` number of MTS 
 * \param *T Unrolled vector representing the second time Series to compare against `*S`
 * \param window_size Length of the two given MTS
 * \param dimensions Number of variables for the two MTS
 * \param *data_out Vector containing the results achieved by comparing `*T` against `*S`
 * \param *trainSize Number of MTS contained in the vector `T`
 * \param task Integer discriminating the task to perform (e.g., 0: CLASSIFICATION, 1:SUBSEQUENCE SEARCH)
 */
__global__ void MD_DTW_I(float *S, float *T, int ns, int nt, int dimensions,
                         float *data_out, int trainSize, int task) {

  int idx, offset_x;
  long long int i, j;
  long long int k, l, g;
  float min_nb = 0;
  float array[WS][2];

  extern __shared__ float sh_mem[];

  float *T2 = (float *)sh_mem;
  float *DTW_single_dim =
      (float *)&sh_mem[dimensions *
                       nt]; // offset on the shared memory for the segment T2

  if (task == 0) {
    idx = threadIdx.x * dimensions + threadIdx.y;
    offset_x = ((blockDim.x * blockDim.y * ns) * blockIdx.x) + idx * ns;

    if (((blockDim.x * blockDim.y * blockIdx.x) + idx) >=
        trainSize * dimensions) // 120=train_size
      return;

  } else { // SUBSEQ_SEARCH

    idx = threadIdx.x * dimensions + threadIdx.y;
    offset_x =
        (blockDim.x * blockIdx.x) +
        ((threadIdx.y * trainSize) +
         threadIdx.x); // use blockIdx and other measure to set well the offset

    if ((idx + WS) > trainSize)
      return;
  }

  if (idx == 0) {
    for (i = 0; i < dimensions; i++)
      for (j = 0; j < nt; j++)
        *(T2 + (nt * i + j)) = T[nt * i + j];
  }
  __syncthreads();

  k = 0;
  l = 1;
  for (i = 0; i < nt; i++) {
    if (i == 0)
      array[i][k] = pow((S[offset_x] - T2[nt * threadIdx.y]), 2);
    else
      array[i][k] =
          pow((S[offset_x] - T2[nt * threadIdx.y + i]), 2) + array[i - 1][k];
  }

  k = 1;
  l = 0;
  for (j = 1; j < ns; j++) {
    i = 0;
    array[i][k] =
        pow((S[offset_x + j] - T2[nt * threadIdx.y + i]), 2) + array[i][l];

    for (i = 1; i < nt; i++) {
      double a = array[i - 1][l];
      double b = array[i][l];
      double c = array[i - 1][k];

      min_nb = fminf(a, b);
      min_nb = fminf(c, min_nb);

      array[i][k] =
          pow((S[offset_x + j] - T2[nt * threadIdx.y + i]), 2) + min_nb;
    }
    g = k;
    k = l;
    l = g;
  }
  DTW_single_dim[idx] = array[WS - 1][g];

  __syncthreads();

  if (idx == 0) {
    for (i = 0; i < blockDim.x; i++) {
      data_out[(blockIdx.x * blockDim.x) + i] = 0.0;
      for (j = 0; j < blockDim.y; j++) {
        data_out[(blockIdx.x * blockDim.x) + i] +=
            DTW_single_dim[i * dimensions + j];
      }
    }
  }
}

/**
 * \brief The kernel function `rMD_DTW_D` computes the `Rotation Dependent-Multi Dimensional Dynamic Time Warping` distance (rD-MDDTW).
 *
 * The following kernel function computes the rD-MDDTW taking advantage of the GPU, by using a specific number of threads for block.
    It considers the comparison of all the possible `punctual rotation` of the Multivariate Time Series (MTS) stored into the unrolled vector `*S` against the only unrolled vector `*T`.
    By exploiting the CUDA threads, this computation can be done very fast.

 * \param *S Unrolled vector containing `trainSize` number of MTS 
 * \param *T Unrolled vector representing the second time Series to compare against `*S`
 * \param window_size Length of the two given MTS
 * \param dimensions Nnumber of variables for the two MTS
 * \param *data_out Vector containing the results achieved by comparing `*T` against `*S`
 * \param *trainSize Number of MTS contained in the vector `T`
 * \param gm Integer indicating where to store the unrolled vector `*T` (e.g., 0:shared memory, 1: global memory)
 */
__global__ void rMD_DTW_D(float *S, float *T, int ns, int nt, int dimensions,
                          float *data_out, int trainSize, int gm) {

  long long int k, l, g;
  long long int i, j, p;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  float min_nb = 0;
  float array[WS][2];

  if (gm == 0) {

    extern __shared__ float T2[];

    // offset training set
    int s = dimensions * 2 * WS * (idx / WS);
    int t = s + idx % WS;

    if (idx >= (trainSize * ns)) //
      return;

    if (threadIdx.x == 0) {
      for (i = 0; i < dimensions; i++)
        for (j = 0; j < nt; j++)
          T2[nt * i + j] = T[nt * i + j];
    }
    __syncthreads();

    k = 0;
    l = 1;
    for (i = 0; i < nt; i++) {
      array[i][k] = 0.0;
      for (p = 0; p < dimensions; p++) {
        if (i == 0)
          array[i][k] += pow((S[t + p * 2 * ns] - T2[p * nt]), 2);
        else
          array[i][k] += pow((S[t + p * 2 * ns] - T2[p * nt + i]), 2);
      }
      if (i != 0)
        array[i][k] += array[i - 1][k];
    }

    k = 1;
    l = 0;
    for (j = 1; j < ns; j++) {
      i = 0;
      array[i][k] = 0.0;

      for (p = 0; p < dimensions; p++) {
        array[i][k] += pow((S[t + p * 2 * ns + j] - T2[p * nt + i]), 2);
      }

      array[i][k] += array[i][l];

      for (i = 1; i < nt; i++) {
        array[i][k] = 0.0;
        float a = array[i - 1][l];
        float b = array[i][l];
        float c = array[i - 1][k];

        min_nb = fminf(a, b);
        min_nb = fminf(c, min_nb);

        for (p = 0; p < dimensions; p++)
          array[i][k] += pow((S[t + p * 2 * ns + j] - T2[p * nt + i]), 2);

        array[i][k] += min_nb;
      }
      g = k;
      k = l;
      l = g;
    }

    data_out[idx] = array[nt - 1][g];
  } else {

    // offset training set
    int s = dimensions * 2 * WS * (idx / WS);
    int t = s + idx % WS;

    if (idx >= (trainSize * ns)) //
      return;

    k = 0;
    l = 1;

    // computing first row (instace versus query)
    for (i = 0; i < nt; i++) {
      array[i][k] = 0.0;
      for (p = 0; p < dimensions; p++) {
        if (i == 0)
          array[i][k] += pow((S[t + p * 2 * ns] - T[p * nt]), 2);
        else
          array[i][k] += pow((S[t + p * 2 * ns] - T[p * nt + i]), 2);
      }
      if (i != 0)
        array[i][k] += array[i - 1][k];
    }

    k = 1;
    l = 0;
    for (j = 1; j < ns; j++) {

      i = 0;
      array[i][k] = 0.0;

      for (p = 0; p < dimensions; p++) {
        array[i][k] += pow((S[t + p * 2 * ns + j] - T[p * nt + i]), 2);
      }

      array[i][k] += array[i][l];

      for (i = 1; i < nt; i++) {

        array[i][k] = 0.0;
        float a = array[i - 1][l];
        float b = array[i][l];
        float c = array[i - 1][k];

        min_nb = fminf(a, b);
        min_nb = fminf(c, min_nb);

        for (p = 0; p < dimensions; p++)
          array[i][k] += pow((S[t + p * 2 * ns + j] - T[p * nt + i]), 2);

        array[i][k] += min_nb;
      }
      g = k;
      k = l;
      l = g;
    }
    data_out[idx] = array[nt - 1][g];
  }
}

/**
 * \brief The callback function `checkFlagOpts` check out the correctness of the parameters for a given flag.
 *
 * The following function check out the correctness of the parameters for a given flag by counting the number of parameters.

 * \param **input_args Vector containing all the command line parameters passed to the program
 * \param num_args Vector containing the number of arguments passed to the program
 * \param ind Current index parsed on `**input_args`
 * \param num_opts Number of parameters to parse for the current flag stored into input_args[ind]
 * \return Integer (0,1) indicating the corretness of the number of parameters for the current flag
 */
__host__ int checkFlagOpts(char **input_args, int num_args, int ind,
                           int num_opts) {

  int count = 0;
  char *pch = NULL;

  if (ind + num_opts < num_args) { // it means a wrong number of options params
                                   // and that there's no other flag option

    while (pch == NULL && count <= num_opts) {
      pch = strchr(input_args[ind], '-');
      ind++;
      count++;
    }

    if (count - 1 != num_opts)
      return 0;
    else
      return 1;
  } else if (ind + num_opts > num_args)
    return 0;

  else
    return 1;
}

/**
 * \brief The callback function `readFileSubSeq` allows to read several file formats for the `SUBSEQUENCE SEARCH` task.
 *
 * The following function allows to read several file format for the `SUBSEQUENCE SEARCH` task by providing in input several parameters.

 * \param **file_name Vector containing the absolute paths for the files to read
 * \param *ind_files  Vector containing parsed indices for the file to read
 * \param n_file Number of file to read
 * \param *t_series Vector that will contain the time series `*t`
 * \param *q_series Vector that will contain the time series `*q`
 * \param windows_size Length of both time series
 * \param n_feat Number of variables for both time series
 * \param read_mode Integer for handling different input file formats (for more information, refer to README)
 */
__host__ void readFileSubSeq(char **file_name, int *ind_files, int n_file,
                             float *t_series, int t_size, float *q_series,
                             int window_size, int n_feat, int read_mode) {

  int i, k;

  FILE **inputFile = NULL;

  inputFile = (FILE **)malloc(n_file * sizeof(FILE *));

  for (i = 0; i < n_file; i++) {
    char *curr_file = file_name[ind_files[i]];
    inputFile[i] = fopen(curr_file, "r");

    if ((inputFile[i]) == NULL) {
      fprintf(stderr, "Failed to open file: %s\n", curr_file);
      exit(2);
    }
  }

  float *tmp;

  // dimension on x axis (columns) and time on y axis (rows)
  if (read_mode == 0) {
    tmp = (float *)malloc(n_feat * sizeof(float));

    // reading t_series file
    for (i = 0; i < t_size; i++) {
      for (k = 0; k < n_feat; k++) {
        fscanf(inputFile[0], "%f", &tmp[k]);
        t_series[(k * t_size) + i] = tmp[k];
      }
    }

    // reading q_series file
    for (i = 0; i < window_size; i++) {
      for (k = 0; k < n_feat; k++) {
        fscanf(inputFile[1], "%f", &tmp[k]);
        q_series[(k * window_size) + i] = tmp[k];
      }
    }
  }
  // time on x axis (row) and dimensions on y axis (columns)
  else if (read_mode == 1) {

    tmp = (float *)malloc(t_size * sizeof(float));

    for (k = 0; k < n_feat; k++) {
      for (i = 0; i < t_size; i++) {
        fscanf(inputFile[0], "%f", &tmp[i]);
        t_series[(k * window_size) + i] = tmp[i];
      }
    }
    free(tmp);

    tmp = (float *)malloc(window_size * sizeof(float));

    for (k = 0; k < n_feat; k++) {
      for (i = 0; i < window_size; i++) {
        fscanf(inputFile[1], "%f", &tmp[i]);
        q_series[(k * window_size) + i] = tmp[i];
      }
    }
  }
}

/**
 * \brief The callback function `readFile` allows to read several file formats for the `CLASSIFICATION` task.
 *
 * The following function allows to read several file format for the `CLASSIFICATION` task by providing in input several parameters.

 * \param **file_name Vector containing the absolute paths for the files to read
 * \param *ind_files  Vector containing parsed indices for the file to read
 * \param read_mode Integer for handling different input file formats (for more information, refer to README)
 * \param *data Vector for storing all the data read contained in the file
 * \param data_struct Struct containing some information about the data (e.g., dataset size, train size, ect.)
 * \param windows_size Length for the time series to be stored into `*data`
 * \param *dataLabels Vector for storing all the label information contained in the file
 * \param n_feat Number of variables for both time series
 * \param class_alg Integer for handling different reading modes which depends on the the type of algorithm picked
 */
__host__ void readFile(char **file_name, int *ind_files, int n_file,
                       int read_mode, float *data, struct data data_struct,
                       int window_size, int *dataLabels, int n_feat,
                       int class_alg) {

  FILE **inputFile = NULL;

  inputFile = (FILE **)malloc(n_file * sizeof(FILE *));

  for (int i = 0; i < n_file; i++) {
    char *curr_file = file_name[ind_files[i]];
    inputFile[i] = fopen(curr_file, "r");

    if ((inputFile + i) == NULL) {
      fprintf(stderr, "Failed to open file: %s\n", curr_file);
      exit(2);
    }
  }

  int i, j, k;
  float label = 0;

  // reading data from 1 big file
  if (read_mode == 0) { // read_mode=0

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
    } else {
      lab_ind = 0;
      data_ind = 1;
    }

    float tmp = 0;

    // DIMENSION ON THE ROWS AND LENGTH ON COLS
    for (i = 0; i < data_struct.tot_size; i++) {

      fscanf(inputFile[lab_ind], "%f", &label);

      dataLabels[i] = (int)label;

      for (k = 0; k < n_feat; k++) {
        for (j = 0; j < window_size; j++) {
          fscanf(inputFile[data_ind], "%f", &tmp);

          // MDT_D or MDT_I
          if (class_alg < 2)
            data[(n_feat * i * window_size) + (k * window_size) + j] = tmp;
          else {

            data[(n_feat * 2 * i * window_size) + (2 * k * window_size) + j] =
                tmp;
            data[(n_feat * 2 * i * window_size) +
                 ((2 * k * window_size) + window_size) + j] = tmp;
          }
        }
      }
    }
  }

  // reading from k-files
  else if (read_mode == 1) {

    float *tmp = (float *)malloc(n_feat * sizeof(float));

    for (i = 0; i < data_struct.tot_size; i++) {

      // reading labels
      for (k = 0; k < n_feat; k++)
        fscanf(inputFile[k], "%f", &label);

      dataLabels[i] = (int)label;

      for (j = 0; j < window_size; j++) {
        for (k = 0; k < n_feat; k++)
          fscanf(inputFile[k], "%f", &tmp[k]);

        for (k = 0; k < n_feat; k++) {

          // MDT_D or MDT_I
          if (class_alg < 2)
            data[(n_feat * i * window_size) + (k * window_size) + j] = tmp[k];
          else {
            data[(n_feat * 2 * i * window_size) + (2 * k * window_size) + j] =
                tmp[k];
            data[(n_feat * 2 * i * window_size) +
                 ((2 * k * window_size) + window_size) + j] = tmp[k];
          }
        }
      }
    }
  } else {

    float *tmp = (float *)malloc(window_size * sizeof(float));

    int i = 0;
    int size_arr[2] = {data_struct.train_size, data_struct.test_size};

    for (int ll = 0; ll < n_file; ll++) {
      for (int inn = 0; inn < size_arr[ll]; inn++) {

        // reading data
        for (k = 0; k < n_feat; k++) {

          // reading labels from either train or test set
          fscanf(inputFile[ll], "%f", &label);

          dataLabels[i] = (int)label;

          for (j = 0; j < window_size; j++) {

            fscanf(inputFile[ll], "%f", &tmp[j]); // fd=0 data descript

            // MDT_D or MDT_I
            if (class_alg < 2)
              data[(n_feat * i * window_size) + (k * window_size) + j] = tmp[j];
            else {
              data[(n_feat * 2 * i * window_size) + (2 * k * window_size) + j] =
                  tmp[j];
              data[(n_feat * 2 * i * window_size) +
                   ((2 * k * window_size) + window_size) + j] = tmp[j];
            }
          }
        }
        i++;
      }
    }
  } // END ELSE

  // Closing and deallocatin all files
  for (k = 0; k < n_file; k++)
    fclose(inputFile[k]);

  free(inputFile);
}

/**
 * \brief The callback function `createTrainingTestingSet` splits the dataset information into random train and test subsets.
 *
 * The following function splits the `data` and `label` information into random two different train and test subsets.

 * \param *data Vector containing the data
 * \param *dataLabels Vector containing the label
 * \param dataSize Number of time series stored in the '*data'
 * \param windows_size Length for the time series stored into '*data'
 * \param n_feat Number of variables for the time series stored into '*data'
 * \param *h_train Vector containing the data for the train set
 * \param *trainLabels Vector containing the labels for the train set
 * \param trainSize Number of time series to be stored in the train set
 * \param *h_test Vector containing the data for the test set
 * \param *testLabels Vector containing the labels for the test set
 * \param testSize Number of time series to be stored in the test set
 * \param *tInd Vector providing train and test indices to split data in train test sets
 * \param k_th_fold Number of folds. Must be at least 2
 * \param class_mode Integer for handling different reading modes which depends on the the type of algorithm picked.
 */
__host__ void createTrainingTestingSet(
    float *data, int *dataLabels, int dataSize, int window_size, int n_feat,
    float *h_train, int *trainLabels, int trainSize, float *h_test,
    int *testLabels, int testSize, int *tInd, int k_th_fold, int class_mode) {

  int i, j, k, i_train = 0, i_test = 0;

  if (tInd != NULL) {
    /* Creating Training and Testing set */
    for (i = 0; i < dataSize; i++) {

      // training set
      if (tInd[i] != k_th_fold) {
        trainLabels[i_train] = dataLabels[i];

        for (j = 0; j < window_size; j++) {
          for (k = 0; k < n_feat; k++) {
            if (class_mode < 2) {
              h_train[(n_feat * i_train * window_size) + (k * window_size) +
                      j] =
                  data[(n_feat * i * window_size) + (k * window_size) + j];
            } else {
              h_train[(n_feat * 2 * i_train * window_size) +
                      (2 * k * window_size) + j] =
                  data[(n_feat * 2 * i * window_size) + (2 * k * window_size) +
                       j];
              h_train[(n_feat * 2 * i_train * window_size) +
                      ((2 * k * window_size) + window_size) + j] =
                  data[(n_feat * 2 * i * window_size) +
                       ((2 * k * window_size) + window_size) + j];
            }
          }
        }
        i_train++;
      }
      // testing set
      else {
        testLabels[i_test] = dataLabels[i];

        for (j = 0; j < window_size; j++) {

          for (k = 0; k < n_feat; k++) {
            if (class_mode < 2)
              h_test[(window_size * n_feat * i_test) + window_size * k + j] =
                  data[(n_feat * i * window_size) + (k * window_size) + j];
            else
              h_test[(window_size * n_feat * i_test) + window_size * k + j] =
                  data[(n_feat * 2 * i * window_size) + (2 * k * window_size) +
                       j];
          }
        }
        i_test++;
      }
    }
  } else {

    int i = 0;

    for (int i_train = 0; i < trainSize; i++) {

      trainLabels[i_train] = dataLabels[i];

      for (j = 0; j < window_size; j++) {

        // reading data
        for (k = 0; k < n_feat; k++) {
          if (class_mode < 2)
            h_train[(n_feat * i_train * window_size) + (k * window_size) + j] =
                data[(n_feat * i * window_size) + (k * window_size) + j];
          else {
            h_train[(n_feat * 2 * i_train * window_size) +
                    (2 * k * window_size) + j] =
                data[(n_feat * 2 * i * window_size) + (2 * k * window_size) +
                     j];
            h_train[(n_feat * 2 * i_train * window_size) +
                    ((2 * k * window_size) + window_size) + j] =
                data[(n_feat * 2 * i * window_size) +
                     ((2 * k * window_size) + window_size) + j];
          }
        }
      }
      i_train++;
    }

    for (int i_test = 0; i_test < testSize; i++) {

      testLabels[i_test] = dataLabels[i];

      for (j = 0; j < window_size; j++) {

        for (k = 0; k < n_feat; k++) {
          if (class_mode < 2)
            h_test[(window_size * n_feat * i_test) + window_size * k + j] =
                data[(n_feat * i * window_size) + (k * window_size) + j];
          else
            h_test[(window_size * n_feat * i_test) + window_size * k + j] =
                data[(n_feat * 2 * i * window_size) + (2 * k * window_size) +
                     j];
        }
      }
      i_test++;
    }
  }
}

/**
 * \brief The callback function `cmpfunc` is an utiliy function for sorting vector values
 
 * \param *a Integer value
 * \param *b Integer value
 * \return Difference betwen `*a` and `*b`
 */
__host__ int cmpfunc(const void *a, const void *b) {
  return (*(int *)a - *(int *)b);
}

/**
 * \brief The callback function `generateArray` fills an input array from a desidered starting point.
 
 * \param size Size of the vector
 * \param *arrayG Vector to fill
 * \param *offset Offset from where to start to fill `*arrayG`
 */
__host__ void generateArray(int size, int *arrayG, int offset) {

  int i, j = 0;

  if(offset > size - 1){
    printf("The offset has to be smaller than the size of the array\n");
    exit(-1);
  }

  for (i = offset; size > 0; i++) {
    arrayG[j++] = i;
    size--;
  }
}

/**
 * \brief The callback function `findInd` fill an array with incremental value whether a desiderd value exist into an input array.
 
 * \param *array Vector where to search into
 * \param size Size of the vector
 * \param *arrayG Vector to fill with incremental value
 * \param g Value to find in the `*array`
 */
__host__ void findInd(int *array, int size, int *arrayG, int g) {

  int i, j = 0;
  for (i = 0; i < size; i++) {
    if (array[i] == g) {
      arrayG[j++] = i;
    }
  }
}

/**
 * \brief The callback function `unique_val` look for unique value into an array
 
 * \param *array Vector where to search into
 * \param size Size of the vector
 * \param *arrayG Vector to fill with incremental value
 * \return Number of unique value found into `*array`
 */
__host__ int unique_val(int *array, int size) {

  int i;

  qsort(array, size, sizeof(int), cmpfunc);

  int unique = 1; // incase we have only one element; it is unique!

  for (i = 0;
       i < size - 1 /*since we don't want to compare last element with junk*/;
       i++) {
    if (array[i] == array[i + 1]) {
      continue;
    } else {
      unique++;
    }
  }
  return unique;
}

/**
 * \brief The callback function `accumarray` is an utility function for the k-fold cross validation.
 
 * \param *array Vector where to search into
 * \param size Size of the vector
 * \param *val Value to find
 * \return Utility array
 */
__host__ int *accumarray(int *array, int size, int *val) {

  int i, j = 0;
  int u_val = unique_val(array, size);
  int *nS = (int *)malloc(u_val * sizeof(int));
  memset(nS, 0, u_val * sizeof(int));

  for (i = 0; i < size; i++) {
    if (array[i] == array[i + 1]) {
      nS[j]++;
      continue;
    } else {
      val[j] = array[i];
      nS[j]++;
      j++;
    }
  }
  return nS;
}

/**
 * \brief The callback function `shuffle` is function for shuffling the data contained into an array.
 
 * \param *array Vector to shuffle
 * \param array_size Size of the vector
 * \param shuff_size Shuffle factor size
 */
__host__ void shuffle(int *array, size_t array_size, size_t shuff_size) {

  if (array_size > 1) {
    size_t i;
    for (i = 0; i < shuff_size - 1; i++) {
      size_t j = i + rand() / (RAND_MAX / (array_size - i) + 1);
      int t = array[j];
      array[j] = array[i];
      array[i] = t;
    }
  }
}

/**
 * \brief The callback function `idAssign`is an utility function for the k-fold cross validation.
 
 * \param *perm Vector of permutations
 * \param size_perm Size of the permutations
 * \param *group Support vector
 * \param size_group Size of the support vector
 * \param *rand_ind Vector of random value
 * \param *h Supprt vector
 * \param *tInd Vector of indices values for splitting the dataset into train and test set
 */
__host__ void idAssign(int *perm, int size_perm, int *group, int size_group,
                       int *rand_ind, int *h, int *tInd) {

  int i;
  int group_perm;
  for (i = 0; i < size_group; i++) {
    group_perm = perm[group[i]];
    tInd[h[rand_ind[i]]] = group_perm;
  }
}

/**
 * \brief The callback function `checkCUDAError` display on the standard output more information about a type of CUDA error.
 
 * \param *msg Message to display along with the error information provided by CUDA
 */
__host__ void checkCUDAError(const char *msg) {

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    fprintf(stderr, "Cuda error: %s %s\n", msg, cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
}

/**
 * \brief The callback function `crossvalind_Kfold` generates Cross-Validation indices for splitting the dataset into train and test set.

 * \param *label Vector of labels
 * \param N Size of the vector `*label`
 * \param K Number of fold to generate
 * \param flag_shuffle
 * \return Vector containing 1s for observations that belong to the training set and 0s for observations that belong to the test (evaluation) set.
 */
__host__ int *crossvalind_Kfold(int *label, int N, int K, int flag_shuffle) {

  int *label_copy = (int *)malloc(N * sizeof(int));
  memcpy(label_copy, label, N * sizeof(int));

  // output
  int *tInd = (int *)malloc(N * sizeof(int));
  memset(tInd, 0, N * sizeof(int));

  int ul = unique_val(label_copy, N);

  int *arr_val = (int *)malloc(ul * sizeof(int));

  int *nS = accumarray(label_copy, N, arr_val);

  int i, j;
  int *pq = (int *)malloc(K * sizeof(int));
  generateArray(K, pq, 0);

  for (i = 0; i < ul; i++) {

    int *randInd = (int *)malloc(nS[i] * sizeof(int));
    generateArray(nS[i], randInd, 0);

    int *q = (int *)malloc(nS[i] * sizeof(int));
    int *h = (int *)malloc(nS[i] * sizeof(int));

    findInd(label, N, h, arr_val[i]);

    for (j = 0; j < nS[i]; j++) {
      float val = (float)(K * (j + 1)) / nS[i]; // j+1 because we need no zero
                                                // values; MATLAB: q =
                                                // ceil(K*(1:nS(g))/nS(g));
      q[j] = (int)ceil(val) - 1; // C indices start from 0
    }

    if (flag_shuffle == 1) {

      shuffle(pq, K, K);
      shuffle(randInd, nS[i], nS[i]);
    }

    idAssign(pq, K, q, nS[i], randInd, h, tInd);

    free(randInd);
    free(q);
    free(h);
  }

  return tInd;
}

/**
 * \brief The callback function `countVal` count the number of occurences found for a desidered value stored into an array.

 * \param *data Vector where to search
 * \param N Size of the vector `*data`
 * \param key Desidered value to search into `*data`
 * \return Number of occurences found for the `key` into `*data`
 */
__host__ int countVal(int *data, int N, int key) {

  int i, cnt = 0;
  for (i = 0; i < N; i++) {
    if (data[i] == key)
      cnt++;
  }
  return cnt;
}

/**
 * \brief The callback function `standard_deviation` compute the `standard deviation` of a given vector.
 *
 * The following function computes the `standard deviation` of a given input vector.
 * \param *data Input vector
 * \param n Size of the vector
 * \param *avg `Mean` computed on the input vector
 * \return `Standard deviation` computed on the input vector
 */
__host__ float standard_deviation(float *data, int n, float *avg) {
  float mean = 0.0, sum_deviation = 0.0;
  int i;
  for (i = 0; i < n; ++i) {
    mean += data[i];
  }
  mean = mean / n;
  *avg = mean;
  for (i = 0; i < n; ++i)
    sum_deviation += (data[i] - mean) * (data[i] - mean);
  return sqrt(sum_deviation / (n - 1));
}

/**
 * \brief The callback function `z_normalize2D` z-normalize an input vector.
 *
 * The following function calculate the z score of each value into a vector, relative to the sample mean and standard deviation.

 * The following function computes the `standard deviation` of a given input vector.
 * \param *M Input matrix
 * \param nrow number of rows
 * \param ncol number of columns
 */
__host__ void z_normalize2D(float *M, int nrow, int ncol) {

  int i;
  float std_dev = 0;
  float *mean = (float *)malloc(sizeof(float));

  for (i = 0; i < nrow; i++) {
    std_dev = 0;
    *mean = 0;

    std_dev = standard_deviation(&M[i * ncol], ncol, mean);
    for (int k = 0; k < ncol; k++)
      M[i * ncol + k] = (M[i * ncol + k] - (*mean)) / std_dev;
  }
  free(mean);
}


/**
 * \brief The function `short_ed_c` computes the `mono-dimensional Euclidean` distance.
 *
 *  It considers the calculation of the Euclidean distance for two mono-dimensional time series stored, rispectively into the vectors `*T` and `*S`

 * \param *S Vector containing the first time series
 * \param *T Vector containing the second time series
 * \param window_size Length of the two given time series
 * \return ED distance among the two time series 
 */
__host__ float short_ed_c(float *T, float *S, int window_size) {

  float sumErr = 0;

  for (int i = 0; i < window_size; i++)
    sumErr += (T[i] - S[i]) * (T[i] - S[i]);

  return sqrt(sumErr);
}

/**
 * \brief The function `short_dtw_c` computes the `mono-dimensional Dynamic Time Warping` distance (DTW).
 *
 *  It considers the calculation of the DTW distance for two mono-dimensional time series stored, rispectively into the vectors `*instance` and `*query`

 * \param *S instance containing the first time series
 * \param *query Vector containing the time series to compare against `*instance`
 * \param ns Length of the `*instance`
 * \param nt Length of the `*query`
 * \return DTW distance among the two time series 
 */
__host__ float short_dtw_c(float *instance, float *query, int ns, int nt) {

  int k, l, g;
  long long int i, j;
  float **array;
  float min_nb;

  // create array
  array = (float **)malloc((nt) * sizeof(float *));
  for (i = 0; i < nt; i++) {
    array[i] = (float *)malloc((2) * sizeof(float));
  }

  k = 0;
  l = 1;
  for (i = 0; i < nt; i++) {
    if (i == 0)
      array[i][k] = pow((instance[0] - query[i]),
                        2); // squared difference (ins[0]-query[0])^2
    else
      array[i][k] = pow((instance[0] - query[i]), 2) + array[i - 1][k];
  }

  k = 1;
  l = 0;

  // computing DTW
  for (j = 1; j < ns; j++) {
    i = 0;
    array[i][k] = pow((instance[j] - query[i]), 2) + array[i][l];

    for (i = 1; i < nt; i++) {
      float a = array[i - 1][l];
      float b = array[i][l];
      float c = array[i - 1][k];

      min_nb = fminf(a, b);
      min_nb = fminf(c, min_nb);

      array[i][k] = pow((instance[j] - query[i]), 2) + min_nb;
    }
    g = k;
    k = l;
    l = g;
  }

  float min = array[nt - 1][g];

  for (i = 0; i < ns; i++)
    free(array[i]);
  free(array);

  return min;
}

/**
 * \brief The function `short_md_ed_c` computes the `Multi-Dimensional Euclidean` distance (MD-E).
 *
 *  It considers the calculation of the MD-E distance for two multivariate time series (MTS) stored, rispectively into the vectors `*T` and `*S`

 * \param *S Vector containing the first time series
 * \param *T Vector containing the second time series
 * \param window_size Length of the two given time series
 * \param dimensions Number of variables for the two MTS
 * \param offset Integer used for computing the rotation invariant euclidean distance (It's usually equal to window_size)
 * \return Euclidean distance among the two MTS 
 */
__host__ float short_md_ed_c(float *T, float *S, int window_size,
                             int dimensions, int offset) {

  float sumErr = 0, dd = 0;

  for (int i = 0; i < window_size; i++) {
    dd = 0;
    for (int p = 0; p < dimensions; p++)
      dd += (T[(p * offset) + i] - S[(p * window_size) + i]) *
            (T[(p * offset) + i] - S[(p * window_size) + i]);

    sumErr += dd;
  }

  return sqrt(sumErr);
}

/**
 * \brief The function `short_md_dtw_c` computes the `Multi-Dimensional Dynamic Time Warping` distance (MD-DTW).
 *
 *  It considers the calculation of the MD-DTW distance for two multivariate time series (MTS) stored, rispectively into the vectors `*S` and `*T`

 * \param *S instance containing the first time series
 * \param *T Vector containing the time series to compare against `*instance`
 * \param ns Length of the `*instance`
 * \param nt Length of the `*query
 * \param dim Number of variables for the two MTS
 * \param offset Integer used for computing the rotation invariant euclidean distance (It's usually equal to window_size)
 * \return Dynamic Time Warping distance among the two MTS 
 */
__host__ float short_md_dtw_c(float *S, float *T, int ns, int nt, int dim,
                              int offset) {

  int k, l, g;
  long long int i, j;
  float **array;
  float min_nb;

  array = (float **)malloc((nt) * sizeof(float *));
  for (i = 0; i < nt; i++) {
    array[i] = (float *)malloc((2) * sizeof(float));
  }

  k = 0;
  l = 1;
  for (i = 0; i < nt; i++) {
    array[i][k] = 0.0;
    for (int p = 0; p < dim; p++) {
      if (i == 0)
        array[i][k] += pow((S[p * offset + i] - T[p * nt + i]), 2);
      else
        array[i][k] += pow((S[p * offset + 0] - T[p * nt + i]), 2);
    }
    if (i != 0)
      array[i][k] += array[i - 1][k];
  }

  k = 1;
  l = 0;

  for (j = 1; j < ns; j++) {
    i = 0;
    array[i][k] = 0.0;
    for (int p = 0; p < dim; p++)
      array[i][k] += pow((S[p * offset + j] - T[p * nt + i]), 2);

    array[i][k] += array[i][l];

    for (i = 1; i < nt; i++) {
      array[i][k] = 0.0;
      float a = array[i - 1][l];
      float b = array[i][l];
      float c = array[i - 1][k];

      min_nb = fminf(a, b);
      min_nb = fminf(c, min_nb);

      for (int p = 0; p < dim; p++)
        array[i][k] += pow((S[p * offset + j] - T[p * nt + i]), 2);

      array[i][k] += min_nb;
    }
    g = k;
    k = l;
    l = g;
  }

  float min = array[nt - 1][g];

  return min;
}

/**
 * \brief The function `print_help` print on the standard output several information about the input parameters to feed to the software.
*/
__host__ void print_help(void) {

  fprintf(stderr,
          "\nUsage: MTSS [OPTIONS]\n"
          "Multivariate Time Serie Software (MTSS) using Multivariate Dynamic "
          "Time Warping\n"
          "\n"
          "OPTIONS:\n"
          "-t Task                \t\tParameters\n"
          "String value           \t\tThis param. represents the kind of task "
          "you want to perform (CLASSIFICATION or SUBSEQ_SEARCH)\n\n"
          "-i Input               \t\tParameters\n"
          "String value           \t\t This param. is used to pick up the CPU "
          "or GPU version\n"
          "Integer value          \t\tThis param. represents the "
          "dimensionality of MTS (TS) (e.g., 1,2,3, ect)\n"
          "Integer values         \t\tThe second/third argument (depending on "
          "the first param.) represents either the desired number of threads "
          "with whom the kernel will be executed (e.g., 64,128,...,1024) or "
          "the read mode. For more information refer to the README.\n\n"
          "-f Files               \t\tParameter\n"
          "String value           \t\tFollow two or more text file "
          "representing the data format (fore more information about the "
          "structure of these files see the README file provided with the "
          "software)\n\n"
          "-k Cross Validation    \t\tParameter\n"
          "Integer value          \t\tThis param. specify the number of K-fold "
          "to use int he K-cross validation step\n\n"
          "Integer value          \t\tSetting this param. to 1 does not allow "
          "the reproducibility of the results on the same dataset among the "
          "GPU and CPU versions\n\n"
          "-o Option Parameters   \t\tParameter.\n"
          "Integer value          \t\tThis param. represents the size of the "
          "dataset (number of sample)\n"
          "Integer value          \t\tThis param. represents the window size "
          "of the MTS\n\n"
          "-m Algorithm Mode      \t\tParameters\n"
          "Integer value          \t\tThis param. represents the type of MTSS "
          "algorithm to use in the tasks (for more information see the README "
          "file)\n\n"
          "-d Device Choice       \t\tParameters\n"
          "Integer value          \t\tThis param. specify the GPU device (on "
          "your machine) you want to use to execute the MTSS\n\n"
          "-v Verbose Mode        \t\tParameters\n"
          "Integer value          \t\tThis param. specify the verbosity "
          "outputs for the software.The value 0 means no verbosity\n\n"
          "--version              \t\tDisplay version information.\n"
          "--help                 \t\tDisplay help information.\n"
          "\n"
          "e.g.\n"
          "./mdtwObj -t CLASSIFICATION -i CPU 3 1 -f "
          "data/classification/rm_1/X_MAT data/classification/rm_1/Y_MAT "
          "data/classification/rm_1/Z_MAT -k 10 0 -o 1000 152 -m 0 DTW\n"
          "./mdtwObj -t CLASSIFICATION -i GPU 3 512 0 -f "
          "data/classification/rm_0/DATA data/classification/rm_0/LABEL -k 10 "
          "0 -o 1000 152 -m 0 DTW -d 0\n"
          "./mdtwObj -t SUBSEQ_SEARCH -i CPU 1 0 -f ECGseries ECGquery -o 3907 "
          "421 -m 0 -d 0\n"
          "./mdtwObj -t SUBSEQ_SEARCH -i GPU 3 512 0 -f "
          "data/subseq_search/T_series data/subseq_search/Q_series -o 3907 421 "
          "-m 1 DTW -d 0\n");
  exit(0);
}

/**
 * \brief The function `print_version` print on the standard output the software version.
*/
__host__ void print_version(void) {

  fprintf(stderr, "Multivariate Time Series Software version 1.0.0\n"
                  "Copyright (C) 2016 Davide Nardone <davide.nardone@live.it>\n"
                  "Originally inspired by Doruk Sart et al\n"
                  "See the README file for license information.\n");
  exit(0);
}

/**
 * \brief The function `infoDev` print on the standard output several information abou the available GPUs.
*/
__host__ void infoDev() {

  int deviceCount;
  cudaGetDeviceCount(&deviceCount);
  printf("Number of device: %d\n", deviceCount);
  int device;
  cudaDeviceProp deviceProp;
  // retrieving all devices
  for (device = 0; device < deviceCount; ++device) {
    // getting information about i-th device
    cudaGetDeviceProperties(&deviceProp, device);
    // printing information about i-th device
    printf("\n\n>>>>>>>>>>>>>>>>>>\nSelected device:%d\n<<<<<<<<<<<<<<<<<<\n\n",
           device);
    printf("\ndevice %d : %s\n", device, deviceProp.name);
    printf("major/minor : %d.%d compute capability\n", deviceProp.major,
           deviceProp.minor);
    printf("Total global mem : %lu bytes\n", deviceProp.totalGlobalMem);
    printf("Shared block mem : %lu bytes\n", deviceProp.sharedMemPerBlock);
    printf("Max memory pitch : %lu bytes\n", deviceProp.memPitch);
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

/**
 * \brief The function `getDevProp` return an object `deviceProp` containing all the information about a specific GPU device.
 
 * \return `deviceProp` object.
 */
__host__ cudaDeviceProp getDevProp(int device) {

  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, device);

  return deviceProp;
}

/**
 * \brief The callback function `initializeArray` fills an input array with a desidered value.
 
 * \param *array Vector to fill
 * \param n Size of the vector
 * \param val Value to fill the array with
 */
__host__ void initializeArray(float *array, int n, float val) {
  int i;
  for (i = 0; i < n; i++)
    array[i] = val;
}

/**
 * \brief The callback function `initializeMatrix` fills an input matrix with random values.
 
 * \param *matrix Matrix to fill
 * \param M Number of rows
 * \param N Number of columns
 */
__host__ void initializeMatrix(float *matrix, int M, int N) {

  int i, j;
  for (i = 0; i < M; i++)
    for (j = 0; j < N; j++)
      matrix[i * N + j] = ((float)rand()) / (float)RAND_MAX;
}

/**
 * \brief The callback function `printArray` print on the standard output an input array of float values.
 
 * \param *array array 
 * \param n Size of the vector
 */
__host__ void printArray(float *array, int n) {

  int i;
  for (i = 0; i < n; i++)
    printf("val[%d]: %f\n", i, array[i]);
  printf("\n");
}

/**
 * \brief The callback function `printArrayI` print on the standard output an input array of integer values.
 
 * \param *array array 
 * \param n Size of the vector
 */
__host__ void printArrayI(int *array, int n) {

  int i;
  for (i = 0; i < n; i++)
    printf("val[%d]: %d\n", i, array[i]);
  printf("\n");
}

/**
 * \brief The callback function `printMatrix` print on the standard output an input matrix of float values.
 
 * \param *array array 
 * \param M Number of rows
 * \param N Number of columns
 */
__host__ void printMatrix(float *matrix, int M, int N) {
  int i, j;
  for (i = 0; i < M; i++) {
    for (j = 0; j < N; j++)
      printf("%f\n", matrix[i * N + j]);
    printf("\n");
  }
}

/**
 * \brief The callback function `equalArray` check whether the host and device result are the same 
 
 * \param *a array host
 * \param *b array device
 * \param n Size of the two vector
 */
__host__ void equalArray(float *a, float *b, int n) {

  int i = 0;

  while (a[i] == b[i])
    i++;
  if (i < n) {
    printf("I risultati dell'host e del device sono diversi\n");
    printf("CPU[%d]: %f, GPU[%d]: %f \n", i, a[i], i, b[i]);
  } else
    printf("I risultati dell'host e del device coincidono\n");
}

/**
 * \brief The callback function `equalArray` print on the standard output both the host and device array
 
 * \param *a array host
 * \param *b array device
 * \param n Size of the two vector
 */
__host__ void compareArray(float *a, float *b, int n) {

  int i = 0;

  for (i = 0; i < n; ++i) {
    if (a[i] != b[i])
      printf("CPU[%d]: %f, GPU[%d]: %f \n", i, a[i], i, b[i]);
  }
}

/**
 * \brief The callback function `min_arr` computes the minimum value of an input array.
 * \param *arr array
 * \param n Size of the two vector
 * \param *ind Index of the minimum value found into the array `*arr`
 * \return minimum value found into the array `*arr`
 */
__host__ float min_arr(float *arr, int n, int *ind) {

  float min = FLT_MAX;
  *ind = -1;
  for (int i = 0; i < n; ++i) {
    if (arr[i] < min) {
      min = arr[i];
      *ind = i;
    }
  }

  return min;
}

/**
 * \brief The callback function `timedifference_msec` computes the time difference among `t0` and `t1`.
 * \param t0 structure containing time took at `t0` 
 * \param t0 structure containing time took at `t1` 
 * \return Elapsed time among `t0` and `t1`
 */
float timedifference_msec(struct timeval t0, struct timeval t1) {
  return (t1.tv_sec - t0.tv_sec) * 1000.0f +
         (t1.tv_usec - t0.tv_usec) / 1000.0f;
}