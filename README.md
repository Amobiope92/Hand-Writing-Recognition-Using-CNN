# Hand-Writing-Recognition-Using-CNN

Handwritten Digit Recognition Using CNN
This project implements a Convolutional Neural Network (CNN) to recognize handwritten digits using the MNIST dataset. The model is trained to classify digits from 0 to 9 by processing grayscale images of size 28x28. This repository provides the code to preprocess the dataset, define and train the CNN, and evaluate the model's performance.

Table of Contents
Introduction
Dataset
Model Architecture
Dependencies
Installation
Training the Model
Evaluation
Results
Acknowledgements
Introduction
This project uses a CNN model to identify handwritten digits from the MNIST dataset. CNNs are widely used in computer vision tasks due to their ability to capture spatial hierarchies in images. The model architecture includes convolutional layers, pooling layers, dropout for regularization, and fully connected layers for classification.

Dataset
The MNIST dataset consists of 60,000 training images and 10,000 test images of handwritten digits. Each image is represented as a 28x28 pixel grayscale image with a label corresponding to the digit (0–9).

Input shape: (28, 28, 1) – Each image is 28 pixels high and 28 pixels wide with 1 grayscale channel.
Number of classes: 10 – Representing digits from 0 to 9.
Model Architecture
The model is built using the Keras Sequential API, consisting of several layers:

Conv2D Layer:
32 filters of size (3x3) with ReLU activation.
Input shape: (28, 28, 1) for grayscale images.
Conv2D Layer:
64 filters of size (3x3) with ReLU activation.
MaxPooling2D Layer:
Pool size of (2x2).
Dropout:
Dropout rate of 0.25 to prevent overfitting.
Flatten Layer:
Flatten the 2D output into 1D for fully connected layers.
Dense Layer:
256 neurons with ReLU activation.
Dropout:
Dropout rate of 0.5.
Dense Output Layer:
10 neurons with softmax activation for multiclass classification.
Model Summary
Layer (type)	Output Shape	Param #
conv2d (Conv2D)	(None, 26, 26, 32)	320
conv2d_1 (Conv2D)	(None, 24, 24, 64)	18,496
max_pooling2d (MaxPooling2D)	(None, 12, 12, 64)	0
dropout (Dropout)	(None, 12, 12, 64)	0
flatten (Flatten)	(None, 9216)	0
dense (Dense)	(None, 256)	2,359,552
dropout_1 (Dropout)	(None, 256)	0
dense_1 (Dense)	(None, 10)	2,570
