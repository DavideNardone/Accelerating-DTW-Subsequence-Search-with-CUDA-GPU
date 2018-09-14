---
title: 'MTSS: A CUDA software for the analysis of Multivariate Time Series'
tags:
- machine learning
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
Multivariate Time Series Software (MTSS) is a GP-GPU/CPU Dynamic Time Warping (DTW) implementation for the analysis of Multivariate Time Series Object (MTSO). The warping of multidimensional time series either for the purpose of classification or sub-sequence comparing is a really time consuming task. We developed a GP-GPU software for tackle down this problem. The platform used for this purpose is CUDA which is a parallel computing platform and programming model developed by NVIDIA for general computing on graphical processing units (GPUs).

MTSS is mainly designed for the *Classification* and the *Subsequence Similarity Search* of MTSO. Originally inspired by [@sart2010accelerating], MTSS aims to improve the Time Performance and Accuracy for classifying and sub-searching any kind of MTSO by using the well known similarity measure: Dynamic Time Warping (DTW).

In order to improve the Time Performance of these two tasks (which may be considered highly time consuming), MTSS presents a GP-GPU implementation which allows it to achieve almost three order of magnitude speedup, whilst getting better Accuracy results. It uses different types of DTW, namely:

1. **D-MDTW:** Dependent-Multivariate Dynamic Time Warping
2. **I-MDTW:** Independent-Multivariate Dynamic Time Warping
3. **R-MDTW:** Rotation-Multivariate Dynamic Time Warping

The software is purely written in CUDA, using C language as support and can be used as a Standard Command Line Options style.

For more information, please refer to [@sart2010accelerating-@shokoohi2015non].

# References

# Acknowledgements
