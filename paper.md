---
title: 'A CUDA software for the analysis of Multivariate Time Series'
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
 - name: Parthenope University of Naples
   index: 1
date: 14 September 2018
bibliography: paper.bib
---

# Summary

MTSS is a GPU/CPU software designed for the Classification and the Subsequence Similarity Search of MTSO.
Originally inspired by [@sart2010accelerating], MTSS aims to improve the Time Performance and Accuracy for classifying and sub-searching any kind of MTS by using the well known similarity measure: Dynamic Time Warping (DTW).

In order to improve the Time Performance of these two tasks (which may be considered highly time consuming), MTSS presents a GP-GPU implementation which allows it to achieve almost three order of magnitude speedup, whilst getting better Accuracy results. It uses different types of DTW, namely:

D-MDTW: Dependent-Multivariate Dynamic Time Warping
I-MDTW: Independent-Multivariate Dynamic Time Warping
R-MDTW: Rotation-Multivariate Dynamic Time Warping
For more information, please refer to [sart2010accelerating-shokoohi2015non].

# References
