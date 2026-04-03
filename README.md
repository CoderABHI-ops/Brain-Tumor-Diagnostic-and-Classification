# Brain Tumor Diagnostic AI Pipeline

## Overview

This repository contains a dual-pipeline Machine Learning architecture designed to classify brain tumors from MRI scans. The primary aim is to systematically identify the best CNN configuration for high-accuracy medical image classification.

The project is divided into two architectural approaches:

1. **Custom CNN (Scratch Model):** A progressively tuned convolutional neural network built from the ground up.
2. **Transfer Learning (VGG16):** An optimized VGG-16 architecture (16 layers: 13 Conv + 3 Fully Connected) utilizing layer unfreezing (fine-tuning) for highly accurate medical feature extraction.

## Dataset Details

The models were trained on a 4-class Brain Tumor MRI dataset, strictly split into Training (70%), Validation (20%), and Testing (10%) sets to ensure robust evaluation:

* **Glioma Tumor:** 648 Train | 185 Val | 93 Test
* **Meningioma Tumor:** 655 Train | 187 Val | 95 Test
* **Pituitary Tumor:** 630 Train | 180 Val | 91 Test
* **No Tumor:** 350 Train | 100 Val | 50 Test

*(Preprocessing included mean subtraction and rescaling, standardizing all inputs to 224x224x3 RGB dimensions).*

## Hyperparameter Optimization Tournament

To find the optimal architecture, a multi-stage hyperparameter tuning "tournament" was programmed to evaluate:

* **Network Width & Kernels:** Tested starting filter sizes (32 vs 64) and receptive fields (3x3 vs 5x5 kernels).
* **Network Depth:** Compared 2, 3, and 4 convolutional block depths.
* **Regularization:** Tested dropout rates (0.0, 0.3, 0.5) to prevent overfitting on complex MRI textures.
* **Training Dynamics:** Grid searched Batch Sizes (32 vs 64), Learning Rates (0.001 vs 0.0001), and Optimizers (Adam vs SGD).
* **Transfer Learning Tuning:** Compared a completely frozen VGG16 base (~14.7M frozen parameters) vs. fine-tuning by unfreezing the last 4 layers.

## Repository Structure

* `Medical_CNN_from_Scratch.ipynb`: Contains the end-to-end pipeline for the custom CNN and the 6-stage hyperparameter tuning engine.
* `Medical_CNN_Transfer_Learning.ipynb`: Contains the VGG16 transfer learning pipeline, demonstrating advanced layer freezing/unfreezing techniques.

## Results & Metrics

Both the optimized custom CNN and the fine-tuned VGG16 model achieved excellent diagnostic performance:

* **Accuracy:** Reached up to **~94%** overall testing accuracy.
* **Precision/Recall:** Achieved high F1-Scores across tumor classes (e.g., >0.90 for Glioma, Meningioma, and Pituitary).
* **Evaluation Methods:** Evaluated using standard diagnostic metrics including Precision, Recall, F1-Score, and Seaborn Confusion Matrices to minimize false negatives in a clinical context.
  
  <div align="center">
  <img width="315" height="283" alt="image" src="https://github.com/user-attachments/assets/7c1e2aa3-4cc1-40fd-b23d-5be75953cb1e" />
  </div>
  
**Tech Stack:** `Python`, `TensorFlow/Keras`, `NumPy`, `Pandas`, `Seaborn/Matplotlib`
