
# Image Classification with CIFAR-10 Dataset using CNN

## Overview

This repository contains code for an image classification project using the CIFAR-10 dataset. It's a side project where I explored deep learning concepts and built a Convolutional Neural Network (CNN) model to classify images into 10 different categories.

## Table of Contents

1. [Project Overview](#overview)
2. [Dataset](#dataset)
3. [Model Architecture](#model-architecture)
4. [Training](#training)
5. [Results](#results)
6. [Conclusion](#conclusion)

## Concepts Learned

Throughout this project, I gained hands-on experience and knowledge in the following areas:

- Data preprocessing, including resizing, normalization, and splitting.
- Building a deep learning model using TensorFlow and Keras.
- Data augmentation to improve model robustness.
- Model training, validation, and evaluation.
- Confusion matrix and classification report analysis.
- Visualization of training and validation metrics.

## Dataset

I used the [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) dataset, which consists of 60,000 32x32 color images in 10 different classes, with 6,000 images per class. The dataset is divided into a training set (50,000 images) and a test set (10,000 images).

## Model Architecture

The model used for this project is a Convolutional Neural Network (CNN) with the following architecture:

- Convolutional Layer 1: 32 filters, kernel size (3, 3), ReLU activation.
- Max-Pooling Layer 1: (2, 2) pool size.
- Dropout Layer 1: 25% dropout rate.
- Convolutional Layer 2: 64 filters, kernel size (3, 3), ReLU activation.
- Max-Pooling Layer 2: (2, 2) pool size.
- Dropout Layer 2: 25% dropout rate.
- Flatten Layer: To flatten the feature maps.
- Fully Connected Layer 1: 128 units, ReLU activation.
- Dropout Layer 3: 50% dropout rate.
- Output Layer: 10 units (for 10 classes), softmax activation.

The model was compiled with the Adam optimizer and trained using sparse categorical cross-entropy loss.

## Training

The model was trained over 20 epochs with data augmentation techniques applied during training. Training and validation accuracy and loss were monitored and visualized.

## Results

- Test Accuracy: 94.28%
- Confusion Matrix:
  <img width="489" alt="image" src="https://github.com/rakshita003/Image-classification-using-CNN/assets/43514952/7649dc20-6814-45ca-8489-a816d8df2449">
- Classification Report:
  <img width="337" alt="image" src="https://github.com/rakshita003/Image-classification-using-CNN/assets/43514952/5603a6ae-3058-4b0c-a51f-897858dce8c5">


## Conclusion

In this side project, I successfully implemented a CNN-based image classification model using the CIFAR-10 dataset. I learned various deep learning concepts and techniques, including data preprocessing, data augmentation, model design, and model evaluation. The model achieved a test accuracy of 94.28%, and the results were analyzed using a confusion matrix and classification report.

This project serves as a learning experience and can be further extended with additional experimentation and improvements to achieve even better results.

Feel free to explore the code in the [Colab Notebook](https://colab.research.google.com/drive/1E6IW1gr734x1dG_KNK07uMHbFwxXXoEd?usp=sharing).

If you have any questions or suggestions, please don't hesitate to [email me](mailto:rmath040@uottawa.ca).


