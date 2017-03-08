# Multivariate-Time Series Object Package (M-TOP)

A GP-GPU/CPU Dynamic Time Warping (DTW) implementation for the analysis of Multivariate Time Series Object. M-TOP is a GPU/CPU library for the `classification` and `subsequence search` of MTS  an inspired by [ref].

## What does it do?

The aim of this package is to perform 2 kind of task:

1. MTS Classification

2. MTS Subsequence search

`MTS Classification` task is about to classify a set 

Time series classification is to build a classification model based on labelled time series and then use the model to predict the label of unlabelled time series. The way for time series classification with R is to extract and build features from time series data first, and then apply existing classification techniques, such as SVM, k-NN, neural networks, regression and decision trees, to the feature set.

Discrete Wavelet Transform (DWT) provides a multi-resolution representation using wavelets and is used in the example below. Another popular feature extraction technique is Discrete Fourier Transform (DFT).

## Installation

The package is purely written in CUDA using C language as support. In order to use this package you will need:

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
