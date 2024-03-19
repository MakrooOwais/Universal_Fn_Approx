# Universal Function Estimation with Neural Networks

This project demonstrates the concept of using neural networks as universal function estimators. The code provided trains a neural network to learn an image as a function, using the image as input and generating an output prediction. The neural network architecture used in this project consists of several fully connected layers, along with activation functions and batch normalization.

## Neural Networks as Universal Function Estimators

Neural networks are powerful mathematical models that can learn complex relationships between input and output data. The universal approximation theorem states that a feedforward neural network with a single hidden layer containing a finite number of neurons can approximate any continuous function on a compact input space to arbitrary accuracy, given a sufficiently large number of neurons. This property makes neural networks versatile tools for function approximation tasks in various domains, including image processing, natural language processing, and reinforcement learning.

In this project, the neural network is trained to learn an image as a function. The input to the neural network is the pixel values of the image, and the output is a prediction of the image. By optimizing the parameters of the neural network using gradient descent-based optimization algorithms, such as Adam, the neural network learns to approximate the underlying function that maps input images to output predictions.

## Project Overview

The provided code performs the following steps:

1. Load an image using the Python Imaging Library (PIL) and resize it to a specified size.
2. Define a neural network architecture using PyTorch, consisting of fully connected layers, activation functions (Leaky ReLU), and batch normalization layers.
3. Prepare the input data by converting the image to grayscale and converting it into a PyTorch tensor.
4. Define the loss function (Cross Entropy Loss) and the optimizer (Adam) for training the neural network.
5. Train the neural network by iterating over a specified number of epochs, where each iteration involves forward and backward propagation to update the network parameters.
6. Save the output predictions of the neural network as images at regular intervals during training.
7. Generate a transition video using the saved plots.

## Dependencies

- PyTorch: Deep learning library for building and training neural networks.
- Matplotlib: Plotting library for visualizing data and results.
- PIL (Python Imaging Library): Library for image processing tasks.
- Torchvision: PyTorch library for computer vision tasks.

## Usage

1. Install the required dependencies using `pip install torch matplotlib Pillow`.
2. Run the provided code snippet in a Python environment with GPU support if available for faster training.
3. Replace the input image file path with the desired image to train the neural network on a different image dataset.
4. Adjust the neural network architecture, hyperparameters, and training settings as needed for specific tasks and datasets.
5. After training, use the saved output images to generate a transition video.

## Outputs

<img src="Drawings.png" width="512" height="512" />

<video>
  <source src="video.mp4" type="video/mp4">
Your browser does not support the video tag.
</video>

## Acknowledgments

This project is inspired by the universal approximation theorem and demonstrates the application of neural networks for function approximation tasks in image processing.
