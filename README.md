# Multivariate Time Series Software (MTSS)

A GP-GPU/CPU Dynamic Time Warping (DTW) implementation for the analysis of Multivariate Time Series Object (MTSO).

## What MTSS is?

MTSS is a GPU/CPU software designed for the **_Classification_** and the **_Subsequence Similarity Search_** of MTSO. Originally inspired by [1], MTSS aims to improve the *Time Perfomance* and *Accuracy* for classyfing and sub-searching any kind of MTS by using the well known similarity measure: **_Dynamic Time Warping (DTW)_**. 

In order to improve the *Time Performace* of these two tasks (which may be considered highly time consuming), MTSS present a **GPGPU** implementation which allows to achieve almost three order of magnitude speedup, while to get better *Accuracy* results, it uses different type of DTW, namely:

1. **D-MDTW:** Dependent-Multivariate Dynamic Time Warping
2. **I-MDTW:** Independent-Multivariate Dynamic Time Warping
3. **R-MDTW:** Rotation-Multivariate Dynamic Time Warping

For more information, please refer to [1-2].

## Installation

The software is purely written in CUDA using C language as support. In order to use it you must have installed:

1. A working gcc compiler. 

2. A CUDA version 5.0 or greater. For installing, please refer to the [official CUDA documention](http://docs.nvidia.com/cuda/#axzz4al7PKeAs).


## Usage

The software can be used as a **Standard Command Line Options style** with a list of options that are explained as following. 

### Compiling

Once you are in the main folder, you must compile the **MD_DTW.cu** and **module.cu** files as illustred below:

`nvcc [options] [source files] -o output file>` (i.e. `nvcc -D WS=421 MD_DTW.cu module.cu -o mdtwObj`)

where `-D option` is necessary to define a `static variable` which is used to store the MTS to test against into the *local memory* of each CUDA block.

**NOTE:** The implementation presented here assumes that each compared MTS has the same time length.

### Running

As we said before, **MTSS** allow you to perform two task:

1. **Classification**

2. **Similarity subseq-search**

Each of the two tasks can be perfomed on **GPU** as well as on **CPU**.

The program can run with the following flag options:

- **-t**: It's used to decide what tasks you want to perform (i.e. CLASSIFICATION, SUBSEQ_SEARCH)
- **-i**: It represents the input parameters, where:
  * The first parameter represents which version you want to use: *CPU* or *GPU*;
  * The second parameter represents the number of dimension for the MTS;
  * The third/fourth parameter (depending on the first parameter) represents the *#thread* for each block (GPU) and/or the *read mode*.
  
  **NOTE:** For more information about the read mode, please refer to the section **_Data Format_**.
- **-f**: It's used to specify the file path of the data (refer to the section **_Data Format_**).
- **-k**:(optional) In the **CLASSIFICATION** task, it's possible to perform *k-fold cross validation*, specifying then the number of folders. 
- **-o**: Depending on the *task* and *read mode*, it represents the MTS option parameters:
  1. **CLASSIFICATION (read-mode=0 oppure 1):**
    * The first parameter represents the number of MTS samples;
    * The second parameter represents the length of each MTS sample (same size for each dimension);
    
    **NOTE:** For this combination it's necessary the *-k flag*,
  2. **CLASSIFICATION (read-mode=2):**
    * The first parameter represents the number of MTS sample in the TRAINING SET;
    * The second parameter represents the number of MTS sample in the TESTING SET;
    * The third parameter represents the MTS length (same size for each dimension).
  3. **SUBSEQ_SEARCH (read-mode=0 oppure 1):**
    * The first parameter represents the MTS length (same size for each dimension);
    * The second parameter represents the MTS *query* length to search for.
- **-m**: It's used to specify the type of **_MDTW_** to use:
  * **0**: Dependent similarity measure;
  * **1**: Independent similarity measure;
  * **2**: Rotation similarity measure (It suites only for the CLASSIFICATION task)
  * **\<similarity distance\>**: ED or DTW.
- **-d**: It specify the GPU's ID you want to use (e.g. 0: GeForce GTX TITAN X).
- **--help**: It quickly explain how to use MTSS software.
- **--version**: It's show the info version about the sofware.

Some examples follow:

**CLASSIFICATION**

`nvcc MD_DTW_Classification.cu fun.cu -D WS=152 -o mdtwObj`

`./mdtwObj -t CLASSIFICATION -i CPU 3 1 -f X_MAT Y_MAT Z_MAT -k 10 -o 1000 152 -m 2 -d 0`

`./mdtwObj -t CLASSIFICATION -i GPU 3 512 0 -f DATA LABEL -k 10 -o 1000 152 -m 0 -d 0`

`./mdtwObj -t CLASSIFICATION -i GPU 3 512 1 -f X_MAT Y_MAT Z_MAT -k 10 -o 1000 152 -m 0 -d 0`

`./mdtwObj -t CLASSIFICATION -i GPU 3 512 2 -f TRAINING_SET TESTING_SET -o 500 1500 152 -m 0 -d 0`

**SUBSEQ_SEARCH**

`nvcc -D WS=421 MD_DTW.cu module.cu -o mdtwObj`

`./mdtwObj -t SUBSEQ_SEARCH -i CPU 3 0 -f ECGseries ECGquery -o 3907 421 -m 1`

`./mdtwObj -t SUBSEQ_SEARCH -i GPU 3 512 0 -f ECGseries ECGquery -o 3907 421 -m 0 -d 1`

`./mdtwObj -t SUBSEQ_SEARCH -i GPU 3 512 1 -f ECGseries ECGquery -o 3907 421 -m 1 -d 1`


### Data format

MTSS works only with `txt` file format. Depending on the type of task to perform, the data file must be adequayely formatted.

**_CLASSIFICATION_**

For this task, MTSS provides three different type of reading mode:

1. read-mode=0: It's possible to feed MTSS with two files, (DATA, LABEL). The DATA file must be formatted as a *T*D* data matrix, where each row must contain the dimensional values of the MTS at time instant *t* (in this case, the MTS are appended in the file), while the LABEL file just contains the integer class label. (A template file is placed in [data/classification/rm_0](data/classification/rm_0)

2. read-mode=1: It's possible to feed MTSS with *N files*, where each of them is formatted as a *N*T* data matrix, where each of rows must contain in the first position the integer class label and then the *T-1* values of the MTS. (A template file is placed in [data/classification/rm_1](data/classification/rm_1)).

3. read-mode=2: It's possible to feed MTSS with a TRAINING SET and a TESTING SET file. Both the file must be formatted as *D\*T* data matrix, where each d-th row must contain the MTS values. (Also in this case, the MTS are appended in the file). (A template file is placed in [data/classification/rm_2](data/classification/rm_2)).

**SUBSEQ_SEARCH**

For this task, MTSS takes in input two files (T_SERIES, Q_SERIES). The T_SERIES represents the Time Series on which the Q_SERIES has to be searched. Both the files must be formatted as *D*T* data matrix where each column must contain the dimensional values of the T_SERIES at each time instant *t* and viceversa (depending on the read-mode (0|1)). (A template file is placed in [data/subseq_search/](data/subseq_search/).

**NOTE:** The MTSS presented here assume that all the MTS have the same time length.


## Tests

TODO: Add some test examples

## Contributing

1. Fork it!
2. Create your feature branch: `git checkout -b my-new-feature`
3. Commit your changes: `git commit -am 'Add some feature'`
4. Push to the branch: `git push origin my-new-feature`
5. Submit a pull request

## History

TODO: Write history

## References

[1] Sart, Doruk, et al. "Accelerating dynamic time warping subsequence search with GPUs and FPGAs." Data Mining (ICDM), 2010 IEEE 10th International Conference on. IEEE, 2010.
APA

[2] Shokoohi-Yekta, Mohammad, Jun Wang, and Eamonn Keogh. "On the non-trivial generalization of dynamic time warping to the multi-dimensional case." Proceedings of the 2015 SIAM International Conference on Data Mining. Society for Industrial and Applied Mathematics, 2015.


## License

APGL-3.0
