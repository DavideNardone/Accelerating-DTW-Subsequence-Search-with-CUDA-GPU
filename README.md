# Multivariate Time Series Software (MTSS)

A GP-GPU/CPU Dynamic Time Warping (DTW) implementation for the analysis of Multivariate Time Series (MTS).

## What MTSS is?

MTSS is a GPU/CPU software designed for the **_Classification_** and the **_Subsequence Similarity Search_** of MTS. Originally inspired by [1], MTSS aims to improve the *Time Performance* and *Accuracy* for classifying and sub-searching any kind of MTS by using the well known similarity measure: **_Dynamic Time Warping (DTW)_**. 

In order to improve the *Time Performance* of these two tasks (which may be considered highly time consuming), MTSS presents a **GP-GPU** implementation which allows it to achieve almost three order of magnitude speedup, whilst getting better *Accuracy* results. It uses different types of DTW, namely:

1. **D-MDTW:** Dependent-Multivariate Dynamic Time Warping
2. **I-MDTW:** Independent-Multivariate Dynamic Time Warping
3. **R-MDTW:** Rotation-Multivariate Dynamic Time Warping

For more information, please refer to [1-2].

## Dependecies

The software is purely written in CUDA, using C language as support. In order to use it you must have installed:

1. A working gcc compiler.

2. A CUDA version 5.0 or greater. For installing, please refer to the [official CUDA documention](http://docs.nvidia.com/cuda/#axzz4al7PKeAs).


## Usage
Runn the following command to clone the `master` repository into a *target* directory:

`git clone https://github.com/DavideNardone/MTSS-Multivariate-Time-Series-Sofwtare.git <target>`

### Compiling

Once you are in the main folder, you must compile the **MD_DTW.cu** and **module.cu** files as illustrated below:

`nvcc [options] [source files] -o output file>` (i.e., `nvcc -arch=sm_30 -D WS=<time_length> MD_DTW.cu module.cu -o mdtwObj`)

where `-D option` is necessary to define a `static variable` (representing the time's length of the MTS) which is used to store the MTS's query into the *local memory* of each CUDA block.

**NOTE:** The implementation presented here assumes that each compared MTS has the same time length.

### Running

The software can be used as a **Standard Command Line Options style** with a list of options that are explained as follows. 

As we said before, **MTSS** allows you to perform two tasks:

1. **Classification**

2. **Similarity subseq-search**

Each of the two tasks can be perfomed on **GPU** as well as on **CPU**.

The program can run with the following flag options:

- **-t**: It's used to decide which task you want to perform (i.e. CLASSIFICATION, SUBSEQ_SEARCH)
- **-i**: It represents the input parameters, where:
  * The version you want to use: *CPU* or *GPU*;
  * The number of dimension for the MTS;
  * The third/fourth parameter (depending on the first parameter) represents the *#thread* for each block (GPU) and/or the *read mode*.
  
  **NOTE:** For more information about the *read mode*, please refer to the section **_Data Format_**.
- **-f**: It's used to specify the file path of the data (refer to the section **_Data Format_**).
- **-k (optional)**: In the **CLASSIFICATION** task, it's possible to perform *k-fold cross validation*, specifying then the number of folders and a flag for performing the shuffling among the folders generated. <br>
**NB:** Setting the flag to `1` does not allow the reproducibility of the results on the same dataset among the GPU and CPU versions.
- **-o**: Depending on the *task* and *read mode*, the followig parameters represents
  1. **CLASSIFICATION (read-mode=0 oppure 1):**
  * The number of MTS samples;
  * The length of each MTS sample (same size for each dimension);
    
    **NOTE:** For this combination it's necessary the *-k flag*.
  2. **CLASSIFICATION (read-mode=2):**
  * The number of MTS sample in the TRAINING SET;
  * The number of MTS sample in the TESTING SET;
  * The MTS length (same size for each dimension).
  3. **SUBSEQ_SEARCH (read-mode=0 oppure 1):**
  * The MTS length (same size for each dimension);
  * The MTS *query*'s length to search for.
- **-m**: It's used to specify the type of **_MDTW_** to use:
  * **0**: Dependent similarity measure;
  * **1**: Independent similarity measure;
  * **2**: Rotation similarity measure (It suites only for the CLASSIFICATION task)
  * **\<similarity distance\>**: ED or DTW.
- **-d**: It specify the GPU's ID you want to use (e.g. 0: GeForce GTX TITAN X).
- **--help**: It quickly explain how to use MTSS software.
- **--version**: It's show the info version about the sofware.

### Data format

MTSS works only with `txt` file format. Depending on the type of task to perform, the data file must be adequayely formatted.

**_CLASSIFICATION_**

For this task, MTSS provides three different types of reading mode:

1. read-mode=0: It's possible to feed MTSS with two files, (DATA, LABEL). The DATA file must be formatted as a *T*D* data matrix, where each row must represents the t-th features values of the MTS at the time instant *d-th* (in this case, the MTS are appended in the file), while the LABEL file just contains the integer class label. (A template file is placed in [data/classification/rm_0](data/classification/rm_0)

2. read-mode=1: It's possible to feed MTSS with *N files*, where each of them is formatted as a *N*T* data matrix, where each of rows must contain in the first position the integer class label and then the *T-1* values of the MTS. (A template file is placed in [data/classification/rm_1](data/classification/rm_1)).

3. read-mode=2: It's possible to feed MTSS with a TRAINING SET and a TESTING SET file. Both the file must be formatted as *D\*T* data matrix, where each d-th row must contain the MTS values. (Also in this case, the MTS are appended in the file). (A template file is placed in [data/classification/rm_2](data/classification/rm_2)).

**SUBSEQ_SEARCH**

For this task, MTSS takes in input two files (T_SERIES, Q_SERIES). The T_SERIES represents the Time Series on which the Q_SERIES has to be searched. Both the files must be formatted as *D*T* data matrix where each column must contain the dimensional values of the T_SERIES at each time instant *t* and viceversa (depending on the read-mode (0|1)). (A template file is placed in [data/subseq_search/](data/subseq_search/).

**NOTE:** The MTSS presented here assume that all the MTS have the same time length.


## Dataset 

The example data set [3] (differently formatted) and all the information about it can be retrieved from the following source: https://sites.google.com/site/dtwadaptive/hom

# Examples

Some examples follow:

**CLASSIFICATION**

**Compiling**

`nvcc -arch=sm_30 -D WS=152 MD_DTW.cu module.cu -o mdtwObj`

**Running**

CPU: 

`./mdtwObj -t CLASSIFICATION -i CPU 3 1 -f data/classification/rm_1/X_MAT data/classification/rm_1/Y_MAT data/classification/rm_1/Z_MAT -k 10 0 -o 1000 152 -m 0 DTW`

`./mdtwObj -t CLASSIFICATION -i CPU 3 2 -f data/classification/rm_2/TRAIN data/classification/rm_2/TEST -o 500 1000 152 -m 0 DTW`

GPU:

`./mdtwObj -t CLASSIFICATION -i GPU 3 512 0 -f data/classification/rm_0/DATA data/classification/rm_0/LABEL -k 10 0 -o 1000 152 -m 0 DTW -d 0`

`./mdtwObj -t CLASSIFICATION -i GPU 3 512 1 -f data/classification/rm_1/X_MAT data/classification/rm_1/Y_MAT data/classification/rm_1/Z_MAT -k 10 0 -o 1000 152 -m 0 DTW -d 0`

`./mdtwObj -t CLASSIFICATION -i GPU 3 512 2 -f data/classification/rm_2/TRAIN data/classification/rm_2/TEST -o 150 850 152 -k 10 0 -m 0 DTW -d 0`

**SUBSEQ_SEARCH**

**Compiling**

`nvcc -arch=sm_30 -D WS=421 MD_DTW.cu module.cu -o mdtwObj`

**Running**

CPU: 

`./mdtwObj -t SUBSEQ_SEARCH -i CPU 3 0 -f data/subseq_search/T_series data/subseq_search/Q_series -o 3907 421 -m 1 DTW`

GPU:

`./mdtwObj -t SUBSEQ_SEARCH -i GPU 3 512 0 -f data/subseq_search/T_series data/subseq_search/Q_series -o 3907 421 -m 1 DTW -d 0`

# AUTHORS

  Davide Nardone, University of Naples Parthenope, Science and Techonlogies Departement, Msc Applied Computer Science
  https://www.linkedin.com/in/davide-nardone-127428102/
  
# CONTACTS

For any kind of problem, questions, ideas or suggestions, please don't esitate to contact me at: 
- **davide.nardone@studenti.uniparthenope.it**

## References

[1] Sart, Doruk, et al. "Accelerating dynamic time warping subsequence search with GPUs and FPGAs." Data Mining (ICDM), 2010 IEEE 10th International Conference on. IEEE, 2010.
APA

[2] Shokoohi-Yekta, Mohammad, Jun Wang, and Eamonn Keogh. "On the non-trivial generalization of dynamic time warping to the multi-dimensional case." Proceedings of the 2015 SIAM International Conference on Data Mining. Society for Industrial and Applied Mathematics, 2015.

[3] Shokoohi-Yekta, M., Hu, B., Jin, H. et al. Data Min Knowl Disc (2017) 31: 1. https://doi.org/10.1007/s10618-016-0455-0

## License

[MIT LICENSE](LICENSE)

# Paper

The following software is under riew for the [The Journal of Open Source Software]https://joss.theoj.org/papers/0a9f9006cebb80198e0ad5448cc1fc10 so i wouldn't recommend you to use it for any plagiarism.
