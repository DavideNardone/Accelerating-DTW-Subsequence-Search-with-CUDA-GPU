---
title: 'MTSS: A CUDA software for the analysis of Multivariate Time Series'
tags:
- machine learning
- time series
- multivariate analysis
- classification
- subsequence similarity search
- GPU
- CUDA
authors:
 - name: Davide Nardone
   orcid: 0000-0003-0486-1791
   affiliation: "1"
affiliations:
 - name: Dept. of Science and Technology, University of Naples Parthenope
   index: 1
date: 14 September 2018
bibliography: paper.bib
---

# Summary
In these days, the alignment of multidimensional sequences is required in many fields such as bio-informatics, speech recognition, time-series analysis, etc. The comparison of different time series is involved in discovering similarities and patterns in data. The most important involved tasks are: 1) *indexing* is a process to identify the most similar time series in a dataset given a query time series; 2) *classification* is used to categorize data into predefined groups; 3) *clustering* is an unsupervised categorization of data; 4) *anomaly detection* is the identification of abnormal or unique data items. For such tasks it is necessary to compare time series using an appropriate similarity measure which could not be handled very efficiently by using a traditional DTW, since it makes each pairwise alignment independently and cause each dimension to be compared separately. Although these problems seem to be already solved in the state-of-art, nobody ention the problem to process these tasks on large quantities of data, in a reasonable amount of time. In this work, we propose a GP-GPU Multivariate Time Series Software (MTSS) for addressing this problem.

The Dynamic Time Warping (DTW) is known as a distance measure used for comparing two time signals. In particular, the characteristic of these signals can vary both in dimension and time (time warping). Let’s think, for example, two equal sentences spoken at different rates. The DTW is used to align pairs of signals but with the drawback, that it can be only used for mono-dimensional signals. In many cases, signals are represented by more than on single attribute at time (e.g., Mel-Frequency Cepstra Coef-ficients (MFCC), gesture signals, etc.). This problem has been addressed in [sanguansat2012multiple] with the Multidimensional DynamicTime Warping (MD-DTW), which allows the alignment of Multidimensional Time Signals (MTS) by simply extending the concept behind the DTW but showing *low-speed performance* when applied on high dimensional problems, which so far has not been investigated yet. The reason why such task is so time consuming is due to the alignment process of the K dimensions of two time series.

MTSS has been developed with the purpose of tackling down this problem and speeding-up the above mentioned tasks. Its usage is referred/recommended to those people who need to analyze large quantities of MTS in an affordable amount of time or to those people which aim is to analyze the execution speed of these tasks among GPU and CPU. The platform used for developing the software is CUDA toolkit, which is a parallel computing platform and programming model, developed by NVIDIA for general computing on graphical processing units (GPUs). CUDA has its own device memory on the card and can execute many threads, in parallel [@NVIDIA] which in turn allow each k-th DTW distance (on different segments of the time series) to be computed in parallel. Each of these threads is assigned an ID, that's used to determine the memory addresses (i.e., the segment of the time series) on which it should operate on. The hardware is free to determine the mapping and scheduling of these threads on the available processing cores. A thread block (which dimension can be specified at runtime) is defined as a batch of threads that are guaranteed to run simultaneously and cooperate with each other through shared resources.

# Software Details
The software is purely written in CUDA, using C language as support and can be used as a Standard Command Line Options style. It's mainly designed for MTS *Classification* and *Subsequence Similarity Search*. Originally inspired by [sart2010accelerating], MTSS aims at improving the *Time Performance* for classifying and sub-searching any kind of MTS by using the DTW measure.
While testing the software, we verified that MTSS drastically improves the time performance of these tasks, achieving almost three order of magnitude speedup.

Although the classic DTW approach for measuring the K-dimensional DTW distance of two signals may represent the best approach, in some cases different approaches have shown better results [shokoohi2015non]. MTSS includes two variants for computing the DTW similarity measure, namely:

1. **D-MDTW:** Dependent-Multivariate Dynamic Time Warping
2. **I-MDTW:** Independent-Multivariate Dynamic Time Warping

These two approaches differ from one to another for the way in which the *k's* DTW are computed. In addition to these two approaches, our software provides a *Multivariate Rotation Dynamic Time Warping* (R-MDTW) approach, for getting better classification results on certain scenarios.

# Examples
Compiling:

nvcc -D WS=421 MD_DTW.cu module.cu -o mdtwObj

Running:

./mdtwObj -t CLASSIFICATION -i GPU 3 512 1 -f X_MAT Y_MAT Z_MAT -k 10 -o 1000 152 DTW -m 0 -d 0

# Source Code
The source code for MTSS is avaiable on Github at the following [link](https://github.com/DavideNardone/MTSS-Multivariate-Time-Series-Sofwtare).

# Acknowledgements
The author thanks the department of Science and Technology, University of Naples Parthenope, for the support in developing the software and acknowledges support from the Computer Vision and Pattern Recognition-Lab and the High Performance Scientific Computing Smart-Lab.

# References
