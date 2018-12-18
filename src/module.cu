#include "../include/header.h"

using namespace std;


/**
 * \brief The function `stdDev` compute the `standard deviation` of a given vector allocated on the CUDA device.
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
 * \brief The kernel function `MD_DTW_D` computes the `Dependent-Multi Dimensional Dynamic Time Warping` distance (D-MDDTW).
 *
 * The following kernel function computes the D-MDDTW taking advantage of the GPU, by using a specific number of threads for block.
    It considers the comparison of many Multivariate Time Series (MTS) stored into the unrolled vector `*S` against the only unrolled vector `*T`.
    By exploiting the CUDA threads, this computation can be done very fast.
    For more information about how it's computed, refer to the following link: http://stats.stackexchange.com/questions/184977/multivariate-time-series-euclidean-distance

 * \param *S Unrolled vector containing `trainSize` number of MTS 
 * \param *T Unrolled vector representing the second time Series to compare against `*S`
 * \param *trainSize Number of MTS contained in the vector `T`
 * \param window_size Length of the two given MTS
 * \param dimensions Number of variables for the two MTS
 * \param *data_out Vector containing the results achieved by comparing `*T` against `*S`
 * \param task Integer discriminating the task to perform (e.g., 0: CLASSIFICATION, 1:SUBSEQUENCE SEARCH)
 * \param gm Integer indicating where to store the unrolled vector `*T` (e.g., 0:shared memory, 1: global memory)
 */
template<int WS>
__global__ void MD_DTW_D(float *S, float *T, int trainSize, int window_size, int dimensions,
                         float *data_out, int task, int gm) {

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

      offset = window_size;

      int wind = dimensions * window_size;
      t = idx * wind;
      if ((idx * wind) + wind > trainSize * wind)
        return;

      if (threadIdx.x == 0) {
        for (i = 0; i < dimensions; i++)
          for (j = 0; j < window_size; j++)
            T2[window_size * i + j] = T[window_size * i + j];
      }
      __syncthreads();
    } else {

      offset = trainSize;

      t = idx;
      if ((idx + window_size) > trainSize)
        return;

      if (threadIdx.x == 0) {
        for (i = 0; i < dimensions; i++)
          for (j = 0; j < window_size; j++)
            T2[window_size * i + j] = T[window_size * i + j];
      }
      __syncthreads();
    }

    k = 0;
    l = 1;

    // computing first row (instace versus query)
    for (i = 0; i < window_size; i++) {
      array[i][k] = 0.0;
      for (p = 0; p < dimensions; p++) {
        if (i == 0)
          array[i][k] += pow((S[t + p * offset] - T2[p * window_size]), 2);
        else
          array[i][k] += pow((S[t + p * offset] - T2[p * window_size + i]), 2);
      }
      if (i != 0)
        array[i][k] += array[i - 1][k];
    }

    k = 1;
    l = 0;
    for (j = 1; j < window_size; j++) {

      i = 0;
      array[i][k] = 0.0;

      for (p = 0; p < dimensions; p++)
        array[i][k] += pow((S[t + p * offset + j] - T2[p * window_size + i]), 2);

      array[i][k] += array[i][l];
      for (i = 1; i < window_size; i++) {

        array[i][k] = 0.0;
        float a = array[i - 1][l];
        float b = array[i][l];
        float c = array[i - 1][k];

        min_nb = fminf(a, b);
        min_nb = fminf(c, min_nb);

        for (p = 0; p < dimensions; p++)
          array[i][k] += pow((S[t + p * offset + j] - T2[p * window_size + i]), 2);

        array[i][k] += min_nb;
      }
      g = k;
      k = l;
      l = g;
    }
    data_out[idx] = array[window_size - 1][g];
  } else {

    int t, offset;
    if (task == 0) {

      offset = window_size;

      int wind = dimensions * window_size;
      t = idx * wind;
      if ((idx * wind) + wind > trainSize * wind)
        return;
    } else {

      offset = trainSize;

      t = idx;
      if ((idx + window_size) > trainSize)
        return;
    }

    k = 0;
    l = 1;

    // computing first row (instace versus query)
    for (i = 0; i < window_size; i++) {
      array[i][k] = 0.0;
      for (p = 0; p < dimensions; p++) {
        if (i == 0)
          array[i][k] += pow((S[t + p * offset] - T[p * window_size]), 2);
        else
          array[i][k] += pow((S[t + p * offset] - T[p * window_size + i]), 2);
      }
      if (i != 0)
        array[i][k] += array[i - 1][k];
    }

    k = 1;
    l = 0;
    for (j = 1; j < window_size; j++) {

      i = 0;
      array[i][k] = 0.0;

      for (p = 0; p < dimensions; p++)
        array[i][k] += pow((S[t + p * offset + j] - T[p * window_size + i]), 2);

      array[i][k] += array[i][l];
      for (i = 1; i < window_size; i++) {

        array[i][k] = 0.0;
        float a = array[i - 1][l];
        float b = array[i][l];
        float c = array[i - 1][k];

        min_nb = fminf(a, b);
        min_nb = fminf(c, min_nb);

        for (p = 0; p < dimensions; p++)
          array[i][k] += pow((S[t + p * offset + j] - T[p * window_size + i]), 2);

        array[i][k] += min_nb;
      }
      g = k;
      k = l;
      l = g;
    }
    data_out[idx] = array[window_size - 1][g];
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
 * \param *trainSize Number of MTS contained in the vector `T`
 * \param window_size Length of the two given MTS
 * \param dimensions Number of variables for the two MTS
 * \param *data_out Vector containing the results achieved by comparing `*T` against `*S`
 * \param task Integer discriminating the task to perform (e.g., 0: CLASSIFICATION, 1:SUBSEQUENCE SEARCH)
 */
template<int WS>
__global__ void MD_DTW_I(float *S, float *T, int trainSize, int window_size, int dimensions,
                         float *data_out, int task) {

  int idx, offset_x;
  long long int i, j;
  long long int k, l, g;
  float min_nb = 0;
  float array[WS][2];

  extern __shared__ float sh_mem[];

  float *T2 = (float *)sh_mem;
  float *DTW_single_dim =
      (float *)&sh_mem[dimensions *
                       window_size]; // offset on the shared memory for the segment T2

  if (task == 0) {
    idx = threadIdx.x * dimensions + threadIdx.y;
    offset_x = ((blockDim.x * blockDim.y * window_size) * blockIdx.x) + idx * window_size;

    if (((blockDim.x * blockDim.y * blockIdx.x) + idx) >=
        trainSize * dimensions) // 120=train_size
      return;

  } else { // SUBSEQ_SEARCH

    idx = threadIdx.x * dimensions + threadIdx.y;
    offset_x =
        (blockDim.x * blockIdx.x) +
        ((threadIdx.y * trainSize) +
         threadIdx.x); // use blockIdx and other measure to set well the offset

    if ((idx + window_size) > trainSize)
      return;
  }

  if (idx == 0) {
    for (i = 0; i < dimensions; i++)
      for (j = 0; j < window_size; j++)
        *(T2 + (window_size * i + j)) = T[window_size * i + j];
  }
  __syncthreads();

  k = 0;
  l = 1;
  for (i = 0; i < window_size; i++) {
    if (i == 0)
      array[i][k] = pow((S[offset_x] - T2[window_size * threadIdx.y]), 2);
    else
      array[i][k] =
          pow((S[offset_x] - T2[window_size * threadIdx.y + i]), 2) + array[i - 1][k];
  }

  k = 1;
  l = 0;
  for (j = 1; j < window_size; j++) {
    i = 0;
    array[i][k] =
        pow((S[offset_x + j] - T2[window_size * threadIdx.y + i]), 2) + array[i][l];

    for (i = 1; i < window_size; i++) {
      double a = array[i - 1][l];
      double b = array[i][l];
      double c = array[i - 1][k];

      min_nb = fminf(a, b);
      min_nb = fminf(c, min_nb);

      array[i][k] =
          pow((S[offset_x + j] - T2[window_size * threadIdx.y + i]), 2) + min_nb;
    }
    g = k;
    k = l;
    l = g;
  }
  DTW_single_dim[idx] = array[window_size - 1][g];

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
 * \param *trainSize Number of MTS contained in the vector `T`
 * \param window_size Length of the two given MTS
 * \param dimensions Nnumber of variables for the two MTS
 * \param *data_out Vector containing the results achieved by comparing `*T` against `*S`
 * \param gm Integer indicating where to store the unrolled vector `*T` (e.g., 0:shared memory, 1: global memory)
 */
template<int WS>
__global__ void rMD_DTW_D(float *S, float *T, int trainSize, int window_size, int dimensions,
                          float *data_out, int gm) {

  long long int k, l, g;
  long long int i, j, p;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  float min_nb = 0;
  float array[WS][2];

  if (gm == 0) {

    extern __shared__ float T2[];

    // offset training set
    int s = dimensions * 2 * window_size * (idx / window_size);
    int t = s + idx % window_size;

    if (idx >= (trainSize * window_size)) //
      return;

    if (threadIdx.x == 0) {
      for (i = 0; i < dimensions; i++)
        for (j = 0; j < window_size; j++)
          T2[window_size * i + j] = T[window_size * i + j];
    }
    __syncthreads();

    k = 0;
    l = 1;
    for (i = 0; i < window_size; i++) {
      array[i][k] = 0.0;
      for (p = 0; p < dimensions; p++) {
        if (i == 0)
          array[i][k] += pow((S[t + p * 2 * window_size] - T2[p * window_size]), 2);
        else
          array[i][k] += pow((S[t + p * 2 * window_size] - T2[p * window_size + i]), 2);
      }
      if (i != 0)
        array[i][k] += array[i - 1][k];
    }

    k = 1;
    l = 0;
    for (j = 1; j < window_size; j++) {
      i = 0;
      array[i][k] = 0.0;

      for (p = 0; p < dimensions; p++) {
        array[i][k] += pow((S[t + p * 2 * window_size + j] - T2[p * window_size + i]), 2);
      }

      array[i][k] += array[i][l];

      for (i = 1; i < window_size; i++) {
        array[i][k] = 0.0;
        float a = array[i - 1][l];
        float b = array[i][l];
        float c = array[i - 1][k];

        min_nb = fminf(a, b);
        min_nb = fminf(c, min_nb);

        for (p = 0; p < dimensions; p++)
          array[i][k] += pow((S[t + p * 2 * window_size + j] - T2[p * window_size + i]), 2);

        array[i][k] += min_nb;
      }
      g = k;
      k = l;
      l = g;
    }

    data_out[idx] = array[window_size - 1][g];
  } else {

    // offset training set
    int s = dimensions * 2 * window_size * (idx / window_size);
    int t = s + idx % window_size;

    if (idx >= (trainSize * window_size)) //
      return;

    k = 0;
    l = 1;

    // computing first row (instace versus query)
    for (i = 0; i < window_size; i++) {
      array[i][k] = 0.0;
      for (p = 0; p < dimensions; p++) {
        if (i == 0)
          array[i][k] += pow((S[t + p * 2 * window_size] - T[p * window_size]), 2);
        else
          array[i][k] += pow((S[t + p * 2 * window_size] - T[p * window_size + i]), 2);
      }
      if (i != 0)
        array[i][k] += array[i - 1][k];
    }

    k = 1;
    l = 0;
    for (j = 1; j < window_size; j++) {

      i = 0;
      array[i][k] = 0.0;

      for (p = 0; p < dimensions; p++) {
        array[i][k] += pow((S[t + p * 2 * window_size + j] - T[p * window_size + i]), 2);
      }

      array[i][k] += array[i][l];

      for (i = 1; i < window_size; i++) {

        array[i][k] = 0.0;
        float a = array[i - 1][l];
        float b = array[i][l];
        float c = array[i - 1][k];

        min_nb = fminf(a, b);
        min_nb = fminf(c, min_nb);

        for (p = 0; p < dimensions; p++)
          array[i][k] += pow((S[t + p * 2 * window_size + j] - T[p * window_size + i]), 2);

        array[i][k] += min_nb;
      }
      g = k;
      k = l;
      l = g;
    }
    data_out[idx] = array[window_size - 1][g];
  }
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
 * \param *trainSize Number of MTS contained in the vector `T`
 * \param window_size Length of the two given MTS
 * \param dimensions Number of variables for the two MTS
 * \param *data_out Vector containing the results achieved by comparing `*T` against `*S`
 * \param task Integer discriminating the task to perform (e.g., 0: CLASSIFICATION, 1:SUBSEQUENCE SEARCH)
 * \param gm Integer indicating where to store the unrolled vector `*T` (e.g., 0:shared memory, 1: global memory)
 */
__global__ void MD_ED_D(float *S, float *T, int trainSize, int window_size, int dimensions,
                        float *data_out, int task, int gm) {

  long long int i, j, p;
  float sumErr = 0, dd = 0;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (gm == 0) {
    extern __shared__ float T2[];

    int t, offset;
    if (task == 0) {

      offset = window_size;

      int wind = dimensions * window_size;
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
      if ((idx + window_size) > trainSize)
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

      int wind = dimensions * window_size;
      t = idx * wind;
      if ((idx * wind) + wind > trainSize * wind)
        return;
    } else {

      // in this case 'trainSize' is the number of subsequence to search 'nss',
      // that is, the length of dataset to perform on
      offset = trainSize;

      t = idx;
      if ((idx + window_size) > trainSize)
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
 * \param *trainSize Number of MTS contained in the vector `T`
 * \param window_size Length of the two given MTS
 * \param dimensions Nnumber of variables for the two MTS
 * \param *data_out Vector containing the results achieved by comparing `*T` against `*S`
 * \param task Integer discriminating the task to perform (e.g., 0: CLASSIFICATION, 1:SUBSEQUENCE SEARCH)
 */
__global__ void MD_ED_I(float *S, float *T, int trainSize, int window_size, int dimensions,
                        float *data_out, int task) {

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

    if ((idx + window_size) > trainSize)
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
    int s = dimensions * 2 * window_size * (idx / window_size);
    int t = s + idx % window_size;

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

    int s = dimensions * 2 * window_size * (idx / window_size);
    int t = s + idx % window_size;

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
 * \brief The function `checkFlagOpts` check out the correctness of the parameters for a given flag.
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
 * \brief The function `readFileSubSeq` allows to read several file formats for the `SUBSEQUENCE SEARCH` task.
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

    if ( access(curr_file, F_OK ) == -1 ) {
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
        if( fscanf(inputFile[0], "%f", &tmp[k]) < 1 ){
          fprintf(stderr, "File reading error!\n");
          exit(2);
        }
        t_series[(k * t_size) + i] = tmp[k];
      }
    }

    // reading q_series file
    for (i = 0; i < window_size; i++) {
      for (k = 0; k < n_feat; k++) {
        if( fscanf(inputFile[1], "%f", &tmp[k]) < 1){
          fprintf(stderr, "File reading error!\n");
          exit(2);
        }
        q_series[(k * window_size) + i] = tmp[k];
      }
    }
  }
  // time on x axis (row) and dimensions on y axis (columns)
  else if (read_mode == 1) {

    tmp = (float *)malloc(t_size * sizeof(float));

    for (k = 0; k < n_feat; k++) {
      for (i = 0; i < t_size; i++) {
        if( fscanf(inputFile[0], "%f", &tmp[i]) < 0){
          fprintf(stderr, "File reading error!\n");
          exit(2);
        }
        t_series[(k * window_size) + i] = tmp[i];
      }
    }
    free(tmp);

    tmp = (float *)malloc(window_size * sizeof(float));

    for (k = 0; k < n_feat; k++) {
      for (i = 0; i < window_size; i++) {
        if( fscanf(inputFile[1], "%f", &tmp[i]) < 0){
          fprintf(stderr, "File reading error!\n");
          exit(2);
        }
        q_series[(k * window_size) + i] = tmp[i];
      }
    }
  }
}

/**
 * \brief The function `readFile` allows to read several file formats for the `CLASSIFICATION` task.
 *
 * The following function allows to read several file format for the `CLASSIFICATION` task by providing in input several parameters.

 * \param **file_name Vector containing the absolute paths for the files to read
 * \param *ind_files  Vector containing parsed indices for the file to read
 * \param read_mode Integer for handling different input file formats (for more information, refer to README)
 * \param *data Vector for storing all the data read contained in the file
 * \param data_struct Struct containing some information about the data (e.g., dataset size, train size, ect.)
 * \param window_size Length for the time series to be stored into `*data`
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

    if ( access(curr_file, F_OK ) == -1 ) {
      fprintf(stderr, "File reading error!\n");
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

      if( fscanf(inputFile[lab_ind], "%f", &label) < 1){
          fprintf(stderr, "File reading error!\n");
          exit(2);
      }

      dataLabels[i] = (int)label;

      for (k = 0; k < n_feat; k++) {
        for (j = 0; j < window_size; j++) {
          if( fscanf(inputFile[data_ind], "%f", &tmp) < 1){
            fprintf(stderr, "File reading error!\n");
            exit(2);            
          }

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
        if( fscanf(inputFile[k], "%f", &label) < 1){
          fprintf(stderr, "File reading error!\n");
          exit(2);
        }

      dataLabels[i] = (int)label;

      for (j = 0; j < window_size; j++) {
        for (k = 0; k < n_feat; k++)
          if( fscanf(inputFile[k], "%f", &tmp[k]) < 1){
            fprintf(stderr, "File reading error!\n");
            exit(2);
          }

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
          if( fscanf(inputFile[ll], "%f", &label) < 1){
            fprintf(stderr, "File reading error!\n");
            exit(2);
          }

          dataLabels[i] = (int)label;

          for (j = 0; j < window_size; j++) {

            if( fscanf(inputFile[ll], "%f", &tmp[j]) < 1){ // fd=0 data descript
              fprintf(stderr, "File reading error!\n");
              exit(2);              
            }

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
 * \brief The function `createTrainingTestingSet` splits the dataset information into random train and test subsets.
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
 * \brief The function `cmpfunc` is an utiliy function for sorting vector values
 
 * \param *a Integer value
 * \param *b Integer value
 * \return Difference betwen `*a` and `*b`
 */
__host__ int cmpfunc(const void *a, const void *b) {
  return (*(int *)a - *(int *)b);
}

/**
 * \brief The function `generateArray` fills an input array from a desidered starting point.
 
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
 * \brief The function `findInd` fill an array with incremental value whether a desiderd value exist into an input array.
 
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
 * \brief The function `unique_val` look for unique value into an array
 
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
 * \brief The function `accumarray` is an utility function for the k-fold cross validation.
 
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
 * \brief The function `shuffle` is function for shuffling the data contained into an array.
 
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
 * \brief The function `idAssign`is an utility function for the k-fold cross validation.
 
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
 * \brief The function `checkCUDAError` display on the standard output more information about a type of CUDA error.
 
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
 * \brief The function `crossvalind_Kfold` generates Cross-Validation indices for splitting the dataset into train and test set.

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
 * \brief The function `countVal` count the number of occurences found for a desidered value stored into an array.

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
 * \brief The function `standard_deviation` compute the `standard deviation` of a given vector.
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
 * \brief The function `z_normalize2D` z-normalize an input vector.
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

  int k = 0, l = 0, g = 0;
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

  int k = 0, l = 0, g = 0;
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
 
 * \return `deviceProp` CUDA object containing several information about its own device.
 */
__host__ cudaDeviceProp getDevProp(int device) {

  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, device);

  return deviceProp;
}

/**
 * \brief The function `checkGPU_prop` check whether a GPU property for its own device is correct.

  * \param *compution_type vector used for trigger the check only on for GPU execution
  * \param *deviceProp CUDA object containing several information about its own device
  * \param *prop_in GPU property to check
  * \param *prop_GPU_in GPU property value to check
 */
__host__ void checkGPU_prop(char *compution_type, cudaDeviceProp deviceProp, const char *prop_in, int prop_GPU_in){

  if (strcmp(compution_type, "GPU") == 0) {

    if ( (strcmp(prop_in, "maxThreadsPerBlock") == 0) && (prop_GPU_in < 0 || prop_GPU_in > deviceProp.maxThreadsPerBlock) ) {

      printf(" %d is an irregular #threads for block for the device %s.\n The number of threads "
         "for block has to be included in [0, %d]\n", prop_GPU_in, deviceProp.name, deviceProp.maxThreadsPerBlock);
      exit(-2);
    }
  }
}

/**
 * \brief The function `initializeArray` fills an input array with random values.
 
 * \param *array Vector to fill
 * \param n Size of the vector
 * \param val Value to fill the array with
 */
__host__ void initializeArray(float *array, int n) {
  int i;
  for (i = 0; i < n; i++)
    array[i] = ((float)rand()) / (float)RAND_MAX;
}

__host__ void initializeArray(int *array, int n) {
  int i;
  for (i = 0; i < n; i++)
    array[i] = ((int)rand()) / (int)RAND_MAX;
}

/**
 * \brief The function `initializeMatrix` fills an input matrix with random values.
 
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
 * \brief The function `printArray` print on the standard output an input array of float values.
 
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
 * \brief The function `printArrayI` print on the standard output an input array of integer values.
 
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
 * \brief The function `printMatrix` print on the standard output an input matrix of float values.
 
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
 * \brief The function `equalArray` check whether the host and device result are the same 
 
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
 * \brief The function `equalArray` print on the standard output both the host and device array
 
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
 * \brief The function `min_arr` computes the minimum value of an input array.
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
 * \brief The function `timedifference_msec` computes the time difference among `t0` and `t1`.
 * \param t0 structure containing time took at `t0` 
 * \param t0 structure containing time took at `t1` 
 * \return Elapsed time among `t0` and `t1`
 */
float timedifference_msec(struct timeval t0, struct timeval t1) {
  return (t1.tv_sec - t0.tv_sec) * 1000.0f +
         (t1.tv_usec - t0.tv_usec) / 1000.0f;
}

/**
 * \brief The function `foldit` implements the switch statement for a range of values.
 * \param ws Length for both the time series 
 */
__host__ int foldit (int ws) {
  
    if (ws <= 0) return -1;
    else if (ws > 0 and ws <= 64) return 0;
    else if (ws > 64 and ws <= 128) return 1;
    else if (ws > 128 and ws <= 256) return 2;
    else if (ws > 256 and ws <= 512) return 3;
    else if (ws > 512 and ws <= 1024) return 4;
    else if (ws > 1024 and ws <= 2048) return 5;
    else if (ws > 2048 and ws <= 4096) return 6;
    else if (ws > 4096 and ws <= 8192) return 7;
    else if (ws > 8192 and ws <= 16384) return 8;
    else return 999;   // triggers the default part of the switch
}

/**
 * \brief The function `MDD_SIM_MES_CPU` is a wrapper function used for computing the CPU dependent multidimensional similary measure for the classification task.

 * \param trainSize Number of MTS contained into the train set
 * \param testSize Number of MTS contained into the test set
 * \param *trainLabels Vector containing the labels for the train set
 * \param *testLabels Vector containing the labels for the test set
 * \param *h_train Vector containing the data for the train set
 * \param *h_test Vector containing the data for the test set
 * \param window_size Length for the time series to be stored into `*data`
 * \param n_feat Number of variables for the time series stored into both train and test set
 * \param *distance_type Type of similarity measure to adopt for performing the classification task
 * \param verbose_mode Flag used to increase/reduce the verbosity of the output results
 * \return the number of misclassification  
 */
__host__ float MDD_SIM_MES_CPU(int trainSize, int testSize, int *trainLabels, int *testLabels, float *h_train, float *h_test, int window_size, int n_feat, char *distance_type, int verbose_mode){

  int *minI = (int *)malloc(sizeof(int));
  float *h_Out = (float *)malloc(trainSize * sizeof(float));
  int err = 0;
  float min = 0;

  for (int k = 0; k < testSize; k++) {
    for (int j = 0; j < trainSize; j++) {
      if (strcmp(distance_type, "DTW") == 0) // DTW distance
        h_Out[j] = short_md_dtw_c(&h_train[j * n_feat * window_size],
                                  &h_test[k * n_feat * window_size],
                                  window_size, window_size, n_feat,
                                  window_size);
      else // Euclidean Distance
        h_Out[j] = short_md_ed_c(&h_train[j * n_feat * window_size],
                                 &h_test[k * n_feat * window_size],
                                 window_size, n_feat, window_size);
    }
    min = min_arr(h_Out, trainSize, minI);

    if (trainLabels[*minI] != testLabels[k])
      err++;

    if (verbose_mode > 0 && verbose_mode < testSize) {
      if (k % verbose_mode == 0)
        printf("\t%d\t gt: %d\t\tRI: %d\t%3.6f\n", k, testLabels[k],
               trainLabels[*minI], min);
      else if (k == testSize-1)
        printf("\t%d\t gt: %d\t\tRI: %d\t%3.6f\n", k, testLabels[k],
               trainLabels[*minI], min);
    }
  }
  free(minI);

  return err;
}

/**
 * \brief The function `MDD_SIM_MES_CPU` is a wrapper function used for computing the CPU multidimensional similary measure for the sub-sequence similarity search task.

 * \param nss Number of sub-sequences to search
 * \param *t_series Vector containing the first time series
 * \param *q_series Vector containing the time series to compare against `*instance`
 * \param t_size Length of the time series `*t_series`
 * \param q_size Length of the time series `*q_series`
 * \param n_feat Number of variables for the two MTS
 * \param *distance_type Type of similarity measure to adopt for performing the classification task
 * \param verbose_mode Flag used to increase/reduce the verbosity of the output results
 * \param *owp Support vector containing all the comparing
 * \param *ind_min_val Index containing the minimum value obtained by comparing `*q_series` over `*t_series`
 * \return minimum value obtained by comparing `*q_series` over `*t_series`
 */
__host__ float MDD_SIM_MES_CPU(int nss, float *t_series, float *q_series, int t_size, int q_size, int n_feat, char *distance_type, int verbose_mode, float *owp, int *ind_min_val){

  float min = 9999.99, dist;

  for (int i = 0; i < nss; i++) {

    dist = 0.0;
    if (strcmp(distance_type, "DTW") == 0) // DTW distance
      dist = short_md_dtw_c(&t_series[i], q_series, q_size,
                            q_size, n_feat, t_size);
    else
      dist = short_md_ed_c(&t_series[i], q_series, q_size, n_feat,
                           t_size);

    owp[i] = dist;

    if (verbose_mode > 0 && verbose_mode < nss) {
      if (i % verbose_mode == 0)
        printf("\tCurr val diff. [%d]: %f\n", i, owp[i]);
      else if (i == nss)
        printf("\tCurr val diff. [%d]: %f\n", i, owp[i]);
    }
  }

  // computing minimum value
  min = min_arr(owp, nss, ind_min_val);

  return min;
}

/**
 * \brief The function `MDI_SIM_MES_CPU` is a wrapper function used for computing the CPU independent multidimensional similary measure for the classification task.

 * \param trainSize Number of MTS contained into the train set
 * \param testSize Number of MTS contained into the test set
 * \param *trainLabels Vector containing the labels for the train set
 * \param *testLabels Vector containing the labels for the test set
 * \param *h_train Vector containing the data for the train set
 * \param *h_test Vector containing the data for the test set
 * \param window_size Length for the time series to be stored into `*data`
 * \param n_feat Number of variables for the time series stored into both train and test set
 * \param *distance_type Type of similarity measure to adopt for performing the classification task
 * \param verbose_mode Flag used to increase/reduce the verbosity of the output results
 * \return the number of misclassification
 */
__host__ float MDI_SIM_MES_CPU(int trainSize, int testSize, int *trainLabels, int *testLabels, float *h_train, float *h_test, int window_size, int n_feat, char *distance_type, int verbose_mode){

  int *minI = (int *)malloc(sizeof(int));
  float *h_Out = (float *)malloc(trainSize * window_size * sizeof(float));
  int err = 0;
  float min = 0, dtw_curr = 0, cum_sum = 0;

  for (int k = 0; k < testSize; k++) {
    for (int j = 0; j < trainSize; j++) {
      cum_sum = 0.0;
      for (int d = 0; d < n_feat; d++) {
        if (strcmp(distance_type, "DTW") == 0) // DTW distance
          dtw_curr = short_dtw_c(
              &h_train[(d * window_size) + (j * n_feat * window_size)],
              &h_test[(k * n_feat * window_size) + (d * window_size)],
              window_size, window_size);
        else
          dtw_curr = short_ed_c(
              &h_train[(d * window_size) + (j * n_feat * window_size)],
              &h_test[(k * n_feat * window_size) + (d * window_size)],
              window_size);
        cum_sum += dtw_curr;
      }
      h_Out[j] = cum_sum;
    }
    min = min_arr(h_Out, trainSize, minI);

    if (trainLabels[*minI] != testLabels[k])
      err++;

    if (verbose_mode > 0 && verbose_mode < testSize) {
      if (k % verbose_mode == 0)
        printf("\t%d\t gt: %d\t\tRI: %d\t%3.6f\n", k, testLabels[k],
               trainLabels[*minI], min);
      else if (k == testSize-1)
        printf("\t%d\t gt: %d\t\tRI: %d\t%3.6f\n", k, testLabels[k],
               trainLabels[*minI], min);
    }
  }

  free(minI);

  return err;
}

/**
 * \brief The function `MDI_SIM_MES_CPU` is a wrapper function used for computing the CPU multidimensional similary measure for the sub-sequence similarity search task.

 * \param nss Number of sub-sequences to search
 * \param *t_series Vector containing the first time series
 * \param *q_series Vector containing the time series to compare against `*instance`
 * \param t_size Length of the time series `*t_series`
 * \param q_size Length of the time series `*q_series`
 * \param n_feat Number of variables for the two MTS
 * \param *distance_type Type of similarity measure to adopt for performing the classification task
 * \param verbose_mode Flag used to increase/reduce the verbosity of the output results
 * \param *owp Support vector containing all the comparing
 * \param *ind_min_val Index containing the minimum value obtained by comparing `*q_series` over `*t_series`
 * \return minimum value obtained by comparing `*q_series` over `*t_series`
 */
__host__ float MDI_SIM_MES_CPU(int nss, float *t_series, float *q_series, int t_size, int q_size, int n_feat, char *distance_type, int verbose_mode, float *owp, int *ind_min_val){

  float min = 9999.99, dist, val_curr;

  for (int i = 0; i < nss; i++) {
    dist = 0.0;
    for (int k = 0; k < n_feat; k++) {
      if (strcmp(distance_type, "DTW") == 0) // DTW distance
        val_curr = short_dtw_c(&t_series[(k * t_size) + i],
                               &q_series[(k * q_size)], q_size,
                               q_size);
      else
        val_curr = short_ed_c(&t_series[(k * t_size) + i],
                              &q_series[(k * q_size)], q_size);

      dist += val_curr;
    }

    owp[i] = dist;

    if (verbose_mode > 0 && verbose_mode < nss) {
      if (i % verbose_mode == 0)
        printf("\tCurr val diff. [%d]: %f\n", i, owp[i]);
      else if (i == nss)
        printf("\tCurr val diff. [%d]: %f\n", i, owp[i]);
    }
  }
  min = min_arr(owp, nss, ind_min_val);

  return min;

}

/**
 * \brief The function `MDR_SIM_MES_CPU` is a wrapper function used for computing the CPU multidimensional rotation similary measure for the classification task.

 * \param trainSize Number of MTS contained into the train set
 * \param testSize Number of MTS contained into the test set
 * \param *trainLabels Vector containing the labels for the train set
 * \param *testLabels Vector containing the labels for the test set
 * \param *h_train Vector containing the data for the train set
 * \param *h_test Vector containing the data for the test set
 * \param window_size Length for the time series to be stored into `*data`
 * \param n_feat Number of variables for the time series stored into both train and test set
 * \param *distance_type Type of similarity measure to adopt for performing the classification task
 * \param verbose_mode Flag used to increase/reduce the verbosity of the output results
 * \param *err The number of misclassification using the basic similarity measure
 * \param *errNR The number of misclassification using the rotation similary measure
 */
__host__ void MDR_SIM_MES_CPU(int trainSize, int testSize, int *trainLabels, int *testLabels, float *h_train, float *h_test, int window_size, int n_feat, char *distance_type, int verbose_mode, int *err, int *errNR){

  float *h_Out = (float *)malloc(trainSize * window_size * sizeof(float));
  float minNR = 0.0, min = 0.0;
  int minINR = 0, minI = 0;

  for (int i = 0; i < testSize; i++) {
    for (int j = 0; j < trainSize; j++) {
      for (int k = 0; k < window_size; k++) {
        if (strcmp(distance_type, "DTW") == 0) // DTW distance
          h_Out[(j * window_size) + k] = short_md_dtw_c(
              &h_train[(2 * j * n_feat * window_size) + k],
              &h_test[i * n_feat * window_size], window_size,
              window_size, n_feat, 2 * window_size);
        else
          h_Out[(j * window_size) + k] = short_md_ed_c(
              &h_train[(2 * j * n_feat * window_size) + k],
              &h_test[i * n_feat * window_size], window_size, n_feat,
              2 * window_size);
      }
    }
    min = 9999999999.99;

    minI = -1;
    minINR = -1;
    minNR = 99999999999.99;
    for (int m = 0; m < trainSize; m++) {
      if (h_Out[m * window_size] < minNR) {
        minNR = h_Out[m * window_size];
        minINR = m;
      }
      for (int p = 0; p < window_size; p++) {
        int t = m * window_size + p;

        if (h_Out[t] < min) {
          min = h_Out[t];
          minI = m;
        }
      }
    }

    if (trainLabels[minI] != testLabels[i])
      (*err)++;

    if (trainLabels[minINR] != testLabels[i])
      (*errNR)++;

    if (verbose_mode > 0 && verbose_mode < testSize) {
      if (i % verbose_mode == 0)
        printf("\t%d\t gt: %d\t\tRI: %d\t%3.6f \t\t NRI: %d\t%3.6f\n", i,
               testLabels[i], trainLabels[minI], min,
               trainLabels[minINR], minNR);
      else if (i == testSize-1)
        printf("\t%d\t gt: %d\t\tRI: %d\t%3.6f \t\t NRI: %d\t%3.6f\n", i,
               testLabels[i], trainLabels[minI], min,
               trainLabels[minINR], minNR);
    }
  }
}

/**
 * \brief The function `MDD_SIM_MES_GPU` is a wrapper function used for computing the GPU dependent multidimensional similary measure for the classification task.

 * \param trainSize Number of MTS contained into the train set
 * \param testSize Number of MTS contained into the test set
 * \param *trainLabels Vector containing the labels for the train set
 * \param *testLabels Vector containing the labels for the test set
 * \param *h_train Vector containing the data for the train set
 * \param *h_test Vector containing the data for the test set
 * \param *d_train Vector containing the data for the train set stored in the GPU device
 * \param *d_test Vector containing the data for the test set stored in the GPU device
 * \param *d_Out Vector containing temporary result for the host
 * \param *h_Out Vector containing temporary result for the device
 * \param window_size Length for the time series to be stored into `*data`
 * \param n_feat Number of variables for the time series stored into both train and test set
 * \param blockSize Number of threads to use for comparing the MTS
 * \param deviceProp CUDA object containing several information about its own device 
 * \param *distance_type Type of similarity measure to adopt for performing the classification task
 * \param verbose_mode Flag used to increase/reduce the verbosity of the output results
 * \return the number of misclassification  
 */
__host__ float MDD_SIM_MES_GPU(int trainSize, int testSize, int *trainLabels, int *testLabels, float *h_train, float *h_test, float *d_train, float *d_test, float *d_Out, float *h_Out, int window_size, int n_feat, int blockSize, cudaDeviceProp deviceProp, char *distance_type, int verbose_mode){

  float grid_size, min = 9999.99;
  dim3 grid;
  dim3 threads;

  int *minI = (int *)malloc(sizeof(int));
  int err = 0;

  float T2 = (n_feat * window_size) * sizeof(float);
  int gm = 0;

  if (T2 > deviceProp.sharedMemPerBlock) {

    printf("\tWarning: The T2 test timeserie: %f doesn't fit into the shared "
           "memory: %lu, so it will be allocated into the global "
           "memory\n",
           T2, deviceProp.sharedMemPerBlock);
    gm = 1;
    T2 = 0;
  } else
    gm = 0;

  grid_size = ceil((float)trainSize / blockSize);

  // number of blocks (x,y) for a grid
  grid.x = grid_size;
  grid.y = 1;
  // number of threads (x,y) for each block
  threads.x = blockSize;
  threads.y = 1;

  if(verbose_mode > 0){
    printf("\tGrid_size_x: %d, number_of_threads_x: %d \n", grid.x,
           threads.x);
    printf("\tGrid_size_y: %d, number_of_threads_y: %d \n\n", grid.y,
           threads.y);
  }

  for (int k = 0; k < testSize; k++) {

    cudaMemset(d_test, 0, n_feat * window_size * sizeof(float));
    cudaMemcpy(d_test, h_test + k * (n_feat * window_size),
               n_feat * window_size * sizeof(float),
               cudaMemcpyHostToDevice);

    if (strcmp(distance_type, "DTW") == 0){ // DTW distance

      switch (foldit(window_size)) {
        case 0: MD_DTW_D<64><<<grid, threads, T2>>>(d_train, d_test, trainSize, window_size, 
                                                    n_feat, d_Out, 0, gm);
        break;
        case 1: MD_DTW_D<128><<<grid, threads, T2>>>(d_train, d_test, trainSize, window_size, 
                                                    n_feat, d_Out, 0, gm);
        break;
        case 2: MD_DTW_D<256><<<grid, threads, T2>>>(d_train, d_test, trainSize, window_size, 
                                                    n_feat, d_Out, 0, gm);
        break;
        case 3: MD_DTW_D<512><<<grid, threads, T2>>>(d_train, d_test, trainSize, window_size, 
                                                    n_feat, d_Out, 0, gm);
        break;
        case 4: MD_DTW_D<1024><<<grid, threads, T2>>>(d_train, d_test, trainSize, window_size, 
                                                    n_feat, d_Out, 0, gm);
        break;
        case 5: MD_DTW_D<2048><<<grid, threads, T2>>>(d_train, d_test, trainSize, window_size, 
                                                    n_feat, d_Out, 0, gm);
        break;
        case 6: MD_DTW_D<4096><<<grid, threads, T2>>>(d_train, d_test, trainSize, window_size, 
                                                    n_feat, d_Out, 0, gm);
        break;
        case 7: MD_DTW_D<8192><<<grid, threads, T2>>>(d_train, d_test, trainSize, window_size, 
                                                    n_feat, d_Out, 0, gm);
        break;
        case 8: MD_DTW_D<16384><<<grid, threads, T2>>>(d_train, d_test, trainSize, window_size, 
                                                    n_feat, d_Out, 0, gm);
        break;

        default: printf("No kernel exists for %d window_size\n", window_size); break;
      }
    }
    else
      MD_ED_D <<<grid, threads, T2>>> (d_train, d_test, trainSize, window_size,
                                        n_feat, d_Out, 0, gm);

    // cudaDeviceSynchronize(); // it may be avoided if there's not printf
                             // in the kernel function
    
    cudaMemcpy(h_Out, d_Out, trainSize * sizeof(float),
               cudaMemcpyDeviceToHost);

    min = min_arr(h_Out, trainSize, minI);

    if (trainLabels[*minI] != testLabels[k])
      err++;

    if (verbose_mode > 0 && verbose_mode < testSize) {
      if (k % verbose_mode == 0)
        printf("\t%d\t gt: %d\t\tRI: %d\t%3.6f\n", k, testLabels[k],
               trainLabels[*minI], min);
      else if (k == testSize-1)
        printf("\t%d\t gt: %d\t\tRI: %d\t%3.6f\n", k, testLabels[k],
               trainLabels[*minI], min);
    }
  }
  free(minI);

  return err;
}

/**
 * \brief The function `MDD_SIM_MES_GPU` is a wrapper function used for computing the GPU dependent multidimensional similary measure for the sub-sequence similarity search task.

 * \param nss Number of sub-sequences to search
 * \param *d_t_series Device vector containing the first time series
 * \param *d_q_series Device vector containing the time series to compare against `*instance`
 * \param t_size Length of the time series `*t_series`
 * \param q_size Length of the time series `*q_series`
 * \param n_feat Number of variables for the two MTS
 * \param blockSize Number of threads to use for comparing the MTS
 * \param deviceProp CUDA object containing several information about its own device 
 * \param *distance_type Type of similarity measure to adopt for performing the classification task
 * \param verbose_mode Flag used to increase/reduce the verbosity of the output results
 * \param *owp Support vector containing all the comparing
 * \param *d_owp Device support vector containing all the comparing
 * \param *ind_min_val Index containing the minimum value obtained by comparing `*q_series` over `*t_series`
 * \return minimum value obtained by comparing `*q_series` over `*t_series`
 */
__host__ float MDD_SIM_MES_GPU(int nss, float *d_t_series, float *d_q_series, int t_size, int q_size, int n_feat, int blockSize, cudaDeviceProp deviceProp, char *distance_type, int verbose_mode, float *owp, float *d_owp, int *ind_min_val){

  float grid_size, min = 9999.99;
  dim3 grid;
  dim3 threads;

  // Setting CUDA variables and structure
  grid_size = ceil((double)nss / blockSize);

  // number of blocks (x,y) for a grid
  grid.x = grid_size;
  grid.y = 1;

  // number of threads (x,y) for each block
  threads.x = blockSize;
  threads.y = 1;

  float T2 = (n_feat * q_size) * sizeof(float);
  int gm = 0;

  if (T2 > deviceProp.sharedMemPerBlock) {

    printf("\tWarning: The T2 test timeserie: %f doesn't fit into the shared "
           "memory: %lu, so it will be allocated into the global "
           "memory\n",
           T2, deviceProp.sharedMemPerBlock);
    gm = 1;
    T2 = 0;
  } else
    gm = 0;

  if(verbose_mode > 0){
    printf("\tGrid_size_x: %d, number_of_threads_x: %d \n", grid.x,
           threads.x);
    printf("\tGrid_size_y: %d, number_of_threads_y: %d \n\n", grid.y,
           threads.y);
  }

  if (strcmp(distance_type, "DTW") == 0){ // DTW distance

    switch (foldit(q_size)) {

      case 0: MD_DTW_D<64><<<grid, threads, T2>>>(d_t_series, d_q_series, t_size,
                                                  q_size, n_feat, d_owp, 1, gm);
      break;
      case 1: MD_DTW_D<128><<<grid, threads, T2>>>(d_t_series, d_q_series, t_size,
                                                  q_size, n_feat, d_owp, 1, gm);
      break;
      case 2: MD_DTW_D<256><<<grid, threads, T2>>>(d_t_series, d_q_series, t_size,
                                                  q_size, n_feat, d_owp, 1, gm);
      break;
      case 3: MD_DTW_D<512><<<grid, threads, T2>>>(d_t_series, d_q_series, t_size,
                                                  q_size, n_feat, d_owp, 1, gm);
      break;
      case 4: MD_DTW_D<1024><<<grid, threads, T2>>>(d_t_series, d_q_series, t_size,
                                                  q_size, n_feat, d_owp, 1, gm);
      break;
      case 5: MD_DTW_D<2048><<<grid, threads, T2>>>(d_t_series, d_q_series, t_size,
                                                  q_size, n_feat, d_owp, 1, gm);
      break;
      case 6: MD_DTW_D<4096><<<grid, threads, T2>>>(d_t_series, d_q_series, t_size,
                                                  q_size, n_feat, d_owp, 1, gm);
      break;
      case 7: MD_DTW_D<8192><<<grid, threads, T2>>>(d_t_series, d_q_series, t_size,
                                                  q_size, n_feat, d_owp, 1, gm);
      break;
      case 8: MD_DTW_D<16384><<<grid, threads, T2>>>(d_t_series, d_q_series, t_size,
                                                  q_size, n_feat, d_owp, 1, gm);
      break;
    }
  }
  else
    MD_ED_D << <grid, threads, T2>>> (d_t_series, d_q_series, t_size, q_size,
                                      n_feat, d_owp, 1, gm);

  cudaMemcpy(owp, d_owp, nss * sizeof(float), cudaMemcpyDeviceToHost);

  for (int i = 0; i < nss; ++i) {
    if (verbose_mode > 0 && verbose_mode < nss) {
      if (i % verbose_mode == 0)
        printf("\tCurr val diff. [%d]: %f\n", i, owp[i]);
      else if (i == nss)
        printf("\tCurr val diff. [%d]: %f\n", i, owp[i]);
    }
  }

  min = min_arr(owp, nss, ind_min_val);

  return min;
}

/**
 * \brief The function `MDI_SIM_MES_GPU` is a wrapper function used for computing the GPU independent multidimensional similary measure for the classification task.

 * \param trainSize Number of MTS contained into the train set
 * \param testSize Number of MTS contained into the test set
 * \param *trainLabels Vector containing the labels for the train set
 * \param *testLabels Vector containing the labels for the test set
 * \param *h_train Vector containing the data for the train set
 * \param *h_test Vector containing the data for the test set
 * \param *d_train Vector containing the data for the train set stored in the GPU device
 * \param *d_test Vector containing the data for the test set stored in the GPU device
 * \param *d_Out Vector containing temporary result for the host
 * \param *h_Out Vector containing temporary result for the device
 * \param window_size Length for the time series to be stored into `*data`
 * \param n_feat Number of variables for the time series stored into both train and test set
 * \param blockSize Number of threads to use for comparing the MTS
 * \param deviceProp CUDA object containing several information about its own device 
 * \param *distance_type Type of similarity measure to adopt for performing the classification task
 * \param verbose_mode Flag used to increase/reduce the verbosity of the output results
 * \return the number of misclassification  
 */
__host__ float MDI_SIM_MES_GPU(int trainSize, int testSize, int *trainLabels, int *testLabels, float *h_train, float *h_test, float *d_train, float *d_test, float *d_Out, float *h_Out, int window_size, int n_feat, int blockSize, cudaDeviceProp deviceProp, char *distance_type, int verbose_mode){


  float grid_size, min = 9999.99;
  dim3 grid;
  dim3 threads;

  int *minI = (int *)malloc(sizeof(int));
  int err = 0;

  grid_size = ceil((float)(trainSize * n_feat) / blockSize);
  float dim_row = floor((float)blockSize / n_feat);
  float dim_col = n_feat;

  // number of blocks (x,y) for a grid
  grid.x = grid_size;
  grid.y = 1;
  // number of threads (x,y) for each block
  threads.x = dim_row;
  threads.y = dim_col;

  if(verbose_mode > 0){
    printf("\tGrid_size_x: %d, number_of_threads_x: %d \n", grid.x,
           threads.x);
    printf("\tGrid_size_y: %d, number_of_threads_y: %d \n\n", grid.y,
           threads.y);
  }

  float sh_mem = ((threads.x * threads.y) + (n_feat * window_size)) *
                 sizeof(float);

  for (int k = 0; k < testSize; k++) {
    cudaMemcpy(d_test, h_test + k * (n_feat * window_size),
               n_feat * window_size * sizeof(float),
               cudaMemcpyHostToDevice);

    if (strcmp(distance_type, "DTW") == 0){ // DTW distance

      switch (foldit(window_size)) {

        case 0: MD_DTW_I<64><<<grid, threads, sh_mem>>>(d_train, d_test, trainSize, 
                                                        window_size, n_feat, d_Out, 0);
        break;
        case 1: MD_DTW_I<128><<<grid, threads, sh_mem>>>(d_train, d_test, trainSize, 
                                                        window_size, n_feat, d_Out, 0);
        break;
        case 2: MD_DTW_I<256><<<grid, threads, sh_mem>>>(d_train, d_test, trainSize, 
                                                        window_size, n_feat, d_Out, 0);
        break;
        case 3: MD_DTW_I<512><<<grid, threads, sh_mem>>>(d_train, d_test, trainSize, 
                                                        window_size, n_feat, d_Out, 0);
        break;
        case 4: MD_DTW_I<1024><<<grid, threads, sh_mem>>>(d_train, d_test, trainSize, 
                                                        window_size, n_feat, d_Out, 0);
        break;
        case 5: MD_DTW_I<2048><<<grid, threads, sh_mem>>>(d_train, d_test, trainSize, 
                                                        window_size, n_feat, d_Out, 0);
        break;
        case 6: MD_DTW_I<4096><<<grid, threads, sh_mem>>>(d_train, d_test, trainSize, 
                                                        window_size, n_feat, d_Out, 0);
        break;
        case 7: MD_DTW_I<8192><<<grid, threads, sh_mem>>>(d_train, d_test, trainSize, 
                                                        window_size, n_feat, d_Out, 0);
        break;
        case 8: MD_DTW_I<16384><<<grid, threads, sh_mem>>>(d_train, d_test, trainSize, 
                                                        window_size, n_feat, d_Out, 0);
        break;
      }
    }
    else
      MD_ED_I << <grid, threads, sh_mem>>>
          (d_train, d_test, trainSize, window_size, n_feat, d_Out, 0);

    cudaThreadSynchronize();
    cudaMemcpy(h_Out, d_Out, trainSize * sizeof(float),
               cudaMemcpyDeviceToHost);

    min = min_arr(h_Out, trainSize, minI);

    if (trainLabels[*minI] != testLabels[k])
      err++;

    if (verbose_mode > 0 && verbose_mode < testSize) {
      if (k % verbose_mode == 0)
        printf("\t%d\t gt: %d\t\tRI: %d\t%3.6f\n", k, testLabels[k],
               trainLabels[*minI], min);
      else if (k == testSize-1)
        printf("\t%d\t gt: %d\t\tRI: %d\t%3.6f\n", k, testLabels[k],
               trainLabels[*minI], min);
    }
  }
  free(minI);

  return err;
}

/**
 * \brief The function `MDD_SIM_MES_GPU` is a wrapper function used for computing the GPU independent multidimensional similary measure for the sub-sequence similarity search task.

 * \param nss Number of sub-sequences to search
 * \param *d_t_series Device vector containing the first time series
 * \param *d_q_series Device vector containing the time series to compare against `*instance`
 * \param t_size Length of the time series `*t_series`
 * \param q_size Length of the time series `*q_series`
 * \param n_feat Number of variables for the two MTS
 * \param blockSize Number of threads to use for comparing the MTS
 * \param deviceProp CUDA object containing several information about its own device 
 * \param *distance_type Type of similarity measure to adopt for performing the classification task
 * \param verbose_mode Flag used to increase/reduce the verbosity of the output results
 * \param *owp Support vector containing all the comparing
 * \param *d_owp Device support vector containing all the comparing
 * \param *ind_min_val Index containing the minimum value obtained by comparing `*q_series` over `*t_series`
 * \return minimum value obtained by comparing `*q_series` over `*t_series`
 */
__host__ float MDI_SIM_MES_GPU(int nss, float *d_t_series, float *d_q_series, int t_size, int q_size, int n_feat, int blockSize, cudaDeviceProp deviceProp, char *distance_type, int verbose_mode, float *owp, float *d_owp, int *ind_min_val){

  float grid_size, min = 9999.99;
  dim3 grid;
  dim3 threads;

  // Setting CUDA variables and structure
  grid_size = ceil((float)(nss * n_feat) / blockSize);
  float dim_row = floor((float)blockSize / n_feat);
  float dim_col = n_feat;

  // number of blocks (x,y) for a grid
  grid.x = grid_size;
  grid.y = 1;

  // number of threads (x,y) for each block
  threads.x = dim_row;
  threads.y = dim_col;

  printf("\tGrid_size_x: %d, number_of_threads_x: %d \n", grid.x,
         threads.x);
  printf("\tGrid_size_y: %d, number_of_threads_y: %d \n\n", grid.y,
         threads.y);

  float sh_mem = ((threads.x * threads.y) + (n_feat * t_size)) * sizeof(float);

  if (strcmp(distance_type, "DTW") == 0){ // DTW distance

    switch (foldit(q_size)) {

      case 0: MD_DTW_I<64><<<grid, threads, sh_mem>>> (d_t_series, d_q_series,
                                           t_size, q_size, n_feat, d_owp, 1);
      break;
      case 1: MD_DTW_I<128><<<grid, threads, sh_mem>>> (d_t_series, d_q_series,
                                           t_size, q_size, n_feat, d_owp, 1);
      break;
      case 2: MD_DTW_I<256><<<grid, threads, sh_mem>>> (d_t_series, d_q_series,
                                           t_size, q_size, n_feat, d_owp, 1);
      break;
      case 3: MD_DTW_I<512><<<grid, threads, sh_mem>>> (d_t_series, d_q_series,
                                           t_size, q_size, n_feat, d_owp, 1);
      break;
      case 4: MD_DTW_I<1024><<<grid, threads, sh_mem>>> (d_t_series, d_q_series,
                                           t_size, q_size, n_feat, d_owp, 1);
      break;
      case 5: MD_DTW_I<2048><<<grid, threads, sh_mem>>> (d_t_series, d_q_series,
                                           t_size, q_size, n_feat, d_owp, 1);
      break;
      case 6: MD_DTW_I<4096><<<grid, threads, sh_mem>>> (d_t_series, d_q_series,
                                           t_size, q_size, n_feat, d_owp, 1);
      break;
      case 7: MD_DTW_I<8192><<<grid, threads, sh_mem>>> (d_t_series, d_q_series,
                                           t_size, q_size, n_feat, d_owp, 1);
      break;
      case 8: MD_DTW_I<16384><<<grid, threads, sh_mem>>> (d_t_series, d_q_series,
                                           t_size, q_size, n_feat, d_owp, 1);
      break;
    }
  }
  else
    MD_ED_I << <grid, threads, sh_mem>>>
        (d_t_series, d_q_series, t_size, q_size, n_feat, d_owp, 1);

  cudaMemcpy(owp, d_owp, nss * sizeof(float), cudaMemcpyDeviceToHost);

  for (int i = 0; i < nss; ++i) {
    if (verbose_mode > 0 && verbose_mode < nss) {
      if (i % verbose_mode == 0)
        printf("\tCurr val diff. [%d]: %f\n", i, owp[i]);
      else if (i == nss)
        printf("\tCurr val diff. [%d]: %f\n", i, owp[i]);
    }
  }

  min = min_arr(owp, nss, ind_min_val);

  return min;
}

/**
 * \brief The function `MDR_SIM_MES_GPU` is a wrapper function used for computing the CPU multidimensional rotation similary measure for the classification task.

 * \param trainSize Number of MTS contained into the train set
 * \param testSize Number of MTS contained into the test set
 * \param *trainLabels Vector containing the labels for the train set
 * \param *testLabels Vector containing the labels for the test set
 * \param *h_train Vector containing the data for the train set
 * \param *h_test Vector containing the data for the test set
 * \param *d_train Vector containing the data for the train set stored in the GPU device
 * \param *d_test Vector containing the data for the test set stored in the GPU device
 * \param *d_Out Vector containing temporary result for the host
 * \param *h_Out Vector containing temporary result for the device
 * \param window_size Length for the time series to be stored into `*data`
 * \param n_feat Number of variables for the time series stored into both train and test set
 * \param blockSize Number of threads to use for comparing the MTS
 * \param deviceProp CUDA object containing several information about its own device 
 * \param *distance_type Type of similarity measure to adopt for performing the classification task
 * \param verbose_mode Flag used to increase/reduce the verbosity of the output results
 * \param *err The number of misclassification using the basic similarity measure
 * \param *errNR The number of misclassification using the rotation similary measure
 * \return the number of misclassification  
 */
__host__ void MDR_SIM_MES_GPU(int trainSize, int testSize, int *trainLabels, int *testLabels, float *h_train, float *h_test, float *d_train, float *d_test, float *d_Out, float *h_Out, int window_size, int n_feat, int blockSize, cudaDeviceProp deviceProp, char *distance_type, int verbose_mode, int *err, int *errNR){

  float grid_size, min = 9999.99,minNR = 99999.99;
  dim3 grid;
  dim3 threads;

  int minINR = 0, minI = 0;

  float T2 = (n_feat * window_size) * sizeof(float);
  int gm = 0;


  if (T2 > deviceProp.sharedMemPerBlock) {

    printf("\tWarning: The T2 test timeserie: %f doesn't fit into the shared "
           "memory: %lu, so it will be allocated into the global "
           "memory\n",
           T2, deviceProp.sharedMemPerBlock);
    gm = 1;
    T2 = 0;
  } else
    gm = 0;

  grid_size = ceil((float)trainSize * window_size / blockSize);

  // number of blocks (x,y) for a grid
  grid.x = grid_size;
  grid.y = 1;
  // number of threads (x,y) for each block
  threads.x = blockSize;
  threads.y = 1;

  if(verbose_mode > 0){
    printf("\tGrid_size_x: %d, number_of_threads_x: %d \n", grid.x,
           threads.x);
    printf("\tGrid_size_y: %d, number_of_threads_y: %d \n\n", grid.y,
           threads.y);
  }

  for (int k = 0; k < testSize; k++) {
    cudaMemcpy(d_test, h_test + (k * n_feat * window_size),
               n_feat * window_size * sizeof(float),
               cudaMemcpyHostToDevice);

    if (strcmp(distance_type, "DTW") == 0){ // DTW distance

      switch (foldit(window_size)) {

        case 0: rMD_DTW_D<64><<<grid, threads, T2>>>(d_train, d_test, trainSize,
                                                     window_size, n_feat, d_Out, gm);
        break;
        case 1: rMD_DTW_D<128><<<grid, threads, T2>>>(d_train, d_test, trainSize,
                                                     window_size, n_feat, d_Out, gm);
        break;
        case 2: rMD_DTW_D<256><<<grid, threads, T2>>>(d_train, d_test, trainSize,
                                                     window_size, n_feat, d_Out, gm);
        break;
        case 3: rMD_DTW_D<512><<<grid, threads, T2>>>(d_train, d_test, trainSize,
                                                     window_size, n_feat, d_Out, gm);
        break;
        case 4: rMD_DTW_D<1024><<<grid, threads, T2>>>(d_train, d_test, trainSize,
                                                       window_size, n_feat, d_Out, gm);
        break;
        case 5: rMD_DTW_D<2048><<<grid, threads, T2>>>(d_train, d_test, trainSize,
                                                       window_size, n_feat, d_Out, gm);
        break;
        case 6: rMD_DTW_D<4096><<<grid, threads, T2>>>(d_train, d_test, trainSize,
                                                       window_size, n_feat, d_Out, gm);
        break;
        case 7: rMD_DTW_D<8192><<<grid, threads, T2>>>(d_train, d_test, trainSize,
                                                       window_size, n_feat, d_Out, gm);
        break;
        case 8: rMD_DTW_D<16384><<<grid, threads, T2>>>(d_train, d_test, trainSize,
                                                        window_size, n_feat, d_Out, gm);
        break;
      }
    }
    else
      rMD_ED_D << <grid, threads, T2>>>
          (d_train, d_test, window_size, n_feat, d_Out, trainSize, gm);

    cudaThreadSynchronize();

    cudaMemcpy(h_Out, d_Out, trainSize * window_size * sizeof(float),
               cudaMemcpyDeviceToHost);

    min = 9999999999.99;

    minI = -1;
    minINR = -1;
    minNR = 99999999999.99;
    int i = 0;
    for (int j = 0; j < trainSize; j++) {
      if (h_Out[j * window_size] < minNR) {
        minNR = h_Out[j * window_size];
        minINR = j;
      }
      for (i = 0; i < window_size; i++) {
        int t = j * window_size + i;
        if (h_Out[t] < min) {
          min = h_Out[t];
          minI = j;
        }
      }
    }
    if (trainLabels[minI] != testLabels[k])
      (*err)++;

    if (trainLabels[minINR] != testLabels[k])
      (*errNR)++;

    if (verbose_mode > 0 && verbose_mode < testSize) {
      if (i % verbose_mode == 0)
        printf("\t%d\t gt: %d\t\tRI: %d\t%3.6f \t\t NRI: %d\t%3.6f\n", k,
               testLabels[i], trainLabels[minI], min,
               trainLabels[minINR], minNR);
      else if (i == testSize-1)
        printf("\t%d\t gt: %d\t\tRI: %d\t%3.6f \t\t NRI: %d\t%3.6f\n", k,
               testLabels[i], trainLabels[minI], min,
               trainLabels[minINR], minNR);
    }
  }
}