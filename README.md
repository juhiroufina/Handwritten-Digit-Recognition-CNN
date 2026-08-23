# Handwritten Digit Recognition using CNN

A deep learning project that recognizes handwritten digits (0–9) using a Convolutional Neural Network (CNN) trained on the MNIST dataset.

## Project Overview

This project uses TensorFlow and Keras to build and train a Convolutional Neural Network for handwritten digit classification.

The system can:

- Train a CNN model using the MNIST dataset
- Evaluate the model on unseen test data
- Predict handwritten digits
- Display prediction results
- Show prediction confidence
- Allow users to draw a digit using an interactive interface
- Recognize the drawn digit using the trained CNN model

## Model Performance

| Metric | Result |
|---|---:|
| Test Accuracy | **99.69%** |
| Test Loss | **0.0092** |
| Dataset | MNIST |
| Training Samples | 60,000 |
| Testing Samples | 10,000 |

The model achieved **99.69% test accuracy** on the MNIST test dataset.

> Accuracy may vary slightly when the model is trained again because of training randomness.

## Technologies Used

- Python
- TensorFlow
- Keras
- NumPy
- Matplotlib
- SciPy
- OpenCV / GUI tools used by the project
- MNIST Dataset
- Convolutional Neural Networks (CNN)

## CNN Architecture

The model uses multiple CNN layers to learn important features from handwritten digit images.

The architecture includes:

- Convolutional layers
- Batch Normalization
- ReLU activation
- Max Pooling
- Dropout
- Flatten layer
- Fully Connected Dense layer
- Softmax output layer

The final output layer contains **10 neurons**, representing digits from **0 to 9**.

## Dataset

The project uses the **MNIST handwritten digit dataset**.

The dataset contains:

- **60,000 training images**
- **10,000 testing images**
- Image size: **28 × 28 pixels**
- **10 classes:** digits 0–9

The image pixel values are normalized before being given to the CNN model.

## Project Structure

```text
Handwritten-Digit-Recognition-CNN/
│
├── .gitignore
├── README.md
├── requirements.txt
│
├── digit_recognition.py
├── draw_digit.py
├── predict_digit.py
│
└── screenshots/
    ├── drawing-interface.png
    ├── prediction-output-1.png
    ├── prediction-output-2.png
    └── prediction-output-3.png