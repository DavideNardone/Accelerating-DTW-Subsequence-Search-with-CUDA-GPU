# Multivariate-Time Series Object Package (M-TSOP)

A GP-GPU/CPU Dynamic Time Warping (DTW) implementation for the analysis of Multivariate Time Series Object (MTSO).

## What M-TSOP is?

M-TSOP is a GPU/CPU library for the `classification` and the `subsequence similarity search` of MTSO. Originally inspired by [1], M-TSOP aims to improve the `time perfomance` and `accuracy` for classyfing and sub-searching any kind of MTS by using the well known similarity measure: `Dynamic Time Warping`. 

In order to improve the `time performace` of these two tasks (which may be considered highly time consuming), M-TSOP present a `GPGPU` implementation which allows to achieve almost three order of magnitude speedup, while to get better `accuracy` results, it uses different type of DTW, namely:

1. D-MDTW: Dependent-Multivariate Dynamic Time Warping
2. I-MDTW: Independent-Multivariate Dynamic Time Warping
3. R-MDTW: Rotation-Multivariate Dynamic Time Warping

For more information, please refer to [1-2].

## Installation

The package is purely written in CUDA using C language as support. In order to use this package you must have installed:

1. A working gcc compiler. 

2. A CUDA version 5.0 or greater. For installing, please refer to the official CUDA documention at [CUDA documention](http://docs.nvidia.com/cuda/#axzz4al7PKeAs).


## Usage

###### Compiling and Linking



###### Running


###### Data format

M-TSOP library works only with `.txt` file.

Depending on the chosen task, the input files must be adequately formatted. 


## Tests

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
