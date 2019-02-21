---
title: 'MTSS: A CUDA software for the analysis of Multivariate Time Series'
tags:
- machine learning
- time series
- multivariate analysis
- big data
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
Nowadays the alignment of multidimensional sequences is required in many fields such as bio-informatics, image and audio analysis, business, etc. The comparison of time series is involved in several tasks for discovering similarities and patterns in data. To name a few, we cite: 1) *indexing* is the process of identifying the most similar time series in a dataset given a query time series; 2) *classification* is the supervised categorization process of data into predefined groups [@wei2006semi]; 3) *clustering* is the process of partitioning a set of data into a set of meaningful sub-classes [@alon2003discovering]-[@liao2005clustering]; 4) *anomaly detection* is the identification process of abnormal or unique data items in a dataset given a reference pattern [@chin2005symbolic].
Unfortunately, the above tasks cannot be handled very efficiently by using the traditional Euclidean distance, since it can only aligns a pair of 1D sequence at once, therefore causing each dimension to be compared separately. By definition, the Dynamic Time Warping (DTW) is known as a distance measure used for aligning pairs of monodimensional signals which may vary both in dimension and time, the latter known in the literature as the *time warping problem* (e.g., two equal sentences spoken at different rates). That said, the use of this measure of similarity is not suitable when dealing with signals represented by more than just one single attribute at a time (e.g., set of audio/video features, gesture signals components, etc.).

This problem has been addressed in [@sanguansat2012multiple] which allows the alignment of multidimensional time signals by simply extending the concept behind the DTW but showing *low-speed performance* when applied on high dimensional problems. The reason why such task is so time consuming is due to the alignment process of the several dimensions of two time series. Other approaches [@sart2010accelerating]-[@shokoohi2015non]-[@tapinos2013method]-[@sanguansat2012multiple] try to face the problem of handling large amounts of data on just one dimension or propose methodologies for addressing either the problem of multivariate or the multiple multidimensional sequence alignment. To our knowledge, nobody mention the problem to efficiently process high dimensional problems (multivariate variables) on large amounts of data. In this work, we propose a GP-GPU Multivariate Time Series Software (MTSS) for tackling down this problem and speeding-up the above mentioned tasks.

The use of our software is primarily referred/recommended to those people who need to analyze large quantities of multidimensional data in an affordable amount of time or to those who aim to analyze the execution speed of these tasks among GPU and CPU. The platform used for developing the software is CUDA toolkit, which is a parallel computing platform and programming model, developed by NVIDIA for general computing on graphical processing units. CUDA has its own device memory on the card and can execute many threads in parallel [@NVIDIA] which in turn allow each k-th DTW distance (on different segments of the time series) to be computed faster. For each of these threads is assigned an ID, that is used to determine the memory addresses (i.e., the segment of the time series) on which it should operate on. The hardware is free to determine the mapping and scheduling of these threads on the available processing cores. A thread block (which dimension can be specified at runtime) is defined as a batch of threads that are guaranteed to run simultaneously and cooperate with each other through shared resources.

![Representation for the classification task](../img/classification.png)
![Representation for the subsequencece similarity search task](../img/sub-seq.PNG) 

# Software Details
The software is purely written in CUDA, using C/C++ language as support and can be used as a *standard command line options style*. It's mainly designed for multidimensioal time series *classification* and *subsequence similarity search*. Originally inspired by [@sart2010accelerating], MTSS aims at improving the speed-up [@sun1991toward] for the earlier mentioned tasks, especially when facing high dimensional problems on large amounts of data. The software works on any kind of multidimensional signals and implements two variations of the classic Multidimensional Dynamic Time Warping (MDTW), which in some scenarios have shown either better accuracy or speed-up performance[@shokoohi2015non]. Here are listed the variations of MDTW implemented by the software:

1. **D-MDTW:** Dependent-Multivariate Dynamic Time Warping
2. **I-MDTW:** Independent-Multivariate Dynamic Time Warping
3. **R-MDTW:** Rotation-Multivariate Dynamic Time Warping

The first two approaches differ from one to another in the way in which the *k*'s DTW measure are computed [@shokoohi2015non]. Differently, the *R-MDTW* version allows to compute either the dependent or independent MDTW for all the possible alignments/shift of a given multivaries time series.

However, these methods produce different distance values, therefore the choice of one of these methods can make a significant difference in the classification accuracy, while their different mathematical representations, can be exploited by implementing differents GPU strategies aiming at improving the speed-up performance. Technically speaking, these approaches may outperform one with the other based on the specific scenario you are facing.

# Source Code
The source code for MTSS is avaiable on Github at the following [link](https://github.com/DavideNardone/MTSS-Multivariate-Time-Series-Sofwtare).

# References
