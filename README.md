# Multivariate-Time Series Object Package (M-TSOP)

A GP-GPU/CPU Dynamic Time Warping (DTW) implementation for the analysis of Multivariate Time Series Object (MTSO).

## What M-TSOP is?

M-TSOP is a GPU/CPU library for the `classification` and the `subsequence similarity search` of MTSO. Originally inspired by [ref], M-TSOP aims to improve the `time perfomance` and `accuracy` for classyfing and sub-searching any kind of MTS by using the well known similarity measure: `Dynamic Time Warping`. 

In order to `speed-up` these two tasks (which may be considered highly time consuming), M-TSOP present a `GPGPU` version which allow to achieve almost three order of magnitude speedup, while to get better `accuracy` results it uses two types of DTW, namely:

1. D-DTW: Dependent-Dynamic Time Warping
2. I-DTW: Independent-Dynamic Time Warping



## Installation

The package is purely written in CUDA using C language as support. In order to use this package you must have:

1. A working gcc compiler. 

2. A CUDA version 5.0 or greater. For installing, please refer to the official CUDA documention at http://docs.nvidia.com/cuda/#axzz4al7PKeAs.


## Usage


## Contributing

1. Fork it!
2. Create your feature branch: `git checkout -b my-new-feature`
3. Commit your changes: `git commit -am 'Add some feature'`
4. Push to the branch: `git push origin my-new-feature`
5. Submit a pull request :D

## History

TODO: Write history

## Credits

TODO: Write credits

## License

APGL-3.0
