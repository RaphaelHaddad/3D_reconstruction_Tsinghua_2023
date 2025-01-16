# IMC 2023: Dealing with Multi-View for SfM

## Overview
This repository contains the code and resources for our participation in the Image Matching Challenge (IMC) 2023. The challenge involves generating 3D scenes from multiple images using state-of-the-art methods for pair-making and keypoint extraction/matching, as well as the Structure from Motion (SfM) software COLMAP.

[📄 View IMC 2023 Report](https://drive.google.com/file/d/11-2vdP-LnVHnwQp68271HwfVUVBpLlom/view?usp=share_link)

## Table of Contents
- [Introduction](#introduction)
- [Data](#data)
- [Image Processing](#image-processing)
- [Pairs Making](#pairs-making)
- [Feature Extraction and Matches](#feature-extraction-and-matches)
- [Post Processing](#post-processing)
- [Results Analysis](#results-analysis)
- [Point Cloud Reconstruction](#point-cloud-reconstruction)
- [Improvements and Conclusion](#improvements-and-conclusion)
- [Acknowledgements](#acknowledgements)
- [References](#references)

## Introduction
The IMC 2023 aims to generate 3D scenes using images. This report describes our solution, which uses state-of-the-art methods for pair-making with Cosplace or Eigenplaces, combined with powerful detector-based and detector-free methods for keypoint extraction and matching. We utilized the Structure from Motion software COLMAP and achieved a high rank in the IMC 2023 competition on Kaggle.

<img width="820" alt="image" src="https://github.com/user-attachments/assets/fd15358a-53b1-434c-9be5-32ec4880cadc" />


## Data
The dataset provided by the challenge consists of various categories of scenes, each containing between 30 and 700 images. The dataset also includes a CSV file with rotation matrices and translation vectors for each image in the training dataset. Additionally, each folder contains COLMAP files for 3D scene reconstruction.

## Image Processing
### Rotation
We employed a pre-trained vision transformer to rotate images in multiples of 90 degrees when required. We either reverted the key points to their original orientation before initiating SfM or adjusted the 3D camera poses post-SfM through rotation.

### Cropping
We implemented a cropping technique to concentrate on crucial regions to enhance matching between image pairs. However, this approach proved impractical due to computational costs and issues with missing information.

## Pairs Making
We implemented and tested three solutions from well-known papers: NetVlad, CosPlace, and EigenPlaces. NetVlad ended up being the final solution due to its superior performance on smaller datasets.

## Feature Extraction and Matches
We used both detector-based and detector-free methods for feature extraction and matching.

### Detector-Based
- **SuperPoint**: Neural network for feature detection and description.
- **LightGlue**: Deep neural network for matching local features across images.

### Detector-Free
- **LoFTR**: Transformer model for dense matching.
- **DKM**: Another deep learning model for dense keypoint matching.

## Post Processing
### Database Making
We created a database containing all image names, camera parameters, keypoints, and matches, following COLMAP's specific structure.

### Geometry Verification
We used RANSAC to clean the database by verifying the consistency of matches.

### Mapping
We obtained the final output (rotation matrices and translation vectors) using COLMAP's incremental and iterative method.

## Results Analysis
### Output
The pipeline outputs a CSV file with the file path, rotation matrix, and translation vector for each image. We assessed our pipeline locally using the training dataset.

### Comparison of Methods
We compared different methods (LoFTR, DKM, LightGlue) and found that combining LightGlue and DKM with splitting technique yielded the best results.

## Point Cloud Reconstruction
We compared the final 3D reconstruction from our pipeline with the accurate 3D point cloud provided in the original dataset using COLMAP's dense reconstruction mode.

## Improvements and Conclusion
Future improvements could include exploring multi-view transformers, additional geometric correction methods, and newer techniques like Neural Radiance Fields and Gaussian Splatting. Our current methods show promise but require further development for enhanced performance and accuracy.

## Acknowledgements
We would like to thank our mentors and the organizers of the IMC 2023 for their support and guidance.

## References
- [COSPLACE](https://arxiv.org/abs/2103.01603)
- [EIGENPLACES](https://arxiv.org/abs/2202.01891)
- [COLMAP](https://colmap.github.io/)
- [DKMV3](https://arxiv.org/abs/2103.01900)
- [LightGlue](https://arxiv.org/abs/2103.02000)
- [SuperPoint](https://arxiv.org/abs/1712.07629)
- [LOFTR](https://arxiv.org/abs/2103.01900)
- [NeRF](https://arxiv.org/abs/2003.08934)
- [GaussianSplatting](https://arxiv.org/abs/2106.02023)
