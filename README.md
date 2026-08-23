# Handwritten Digit Recognition using CNN

A deep learning project that recognizes handwritten digits (0–9) using a Convolutional Neural Network (CNN) trained on the MNIST dataset.

## Project Overview

This project uses TensorFlow and Keras to build and train a CNN model for handwritten digit classification.

The system can:

- Train a CNN model using the MNIST dataset
- Evaluate the model on unseen test data
- Predict handwritten digits
- Display the predicted digit
- Show prediction confidence
- Allow users to draw a digit for prediction

## Model Performance

| Metric | Result |
|---|---:|
| Test Accuracy | **99.69%** |
| Test Loss | **0.0092** |
| Dataset | MNIST |
| Training Samples | 60,000 |
| Testing Samples | 10,000 |

> The accuracy may vary slightly when the model is trained again because of training randomness.

## Technologies Used

- Python
- TensorFlow
- Keras
- NumPy
- Matplotlib
- MNIST Dataset
- Convolutional Neural Networks (CNN)

## Project Structure

```text
Handwritten-Digit-Recognition-CNN/
│
├── digit_recognition.py
├── predict_digit.py
├── draw_digit.py
├── requirements.txt
├── README.md
├── .gitignore
└── screenshots/
    └── prediction-output.png