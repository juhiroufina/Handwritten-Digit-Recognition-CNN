import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Load the trained CNN model
model = tf.keras.models.load_model("handwritten_digit_cnn.keras")

# Load MNIST test dataset
(_, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Select one test image
index = 0
image = x_test[index]
actual_digit = y_test[index]

# Prepare image for the CNN
image_for_prediction = image / 255.0
image_for_prediction = image_for_prediction.reshape(1, 28, 28, 1)

# Make prediction
prediction = model.predict(image_for_prediction, verbose=0)

# Get predicted digit
predicted_digit = np.argmax(prediction)

# Get confidence
confidence = np.max(prediction) * 100

# Display result
print("==============================")
print("HANDWRITTEN DIGIT PREDICTION")
print("==============================")
print(f"Actual Digit: {actual_digit}")
print(f"Predicted Digit: {predicted_digit}")
print(f"Confidence: {confidence:.2f}%")

# Display the image
plt.imshow(image, cmap="gray")
plt.title(
    f"Predicted: {predicted_digit} | "
    f"Actual: {actual_digit} | "
    f"Confidence: {confidence:.2f}%"
)
plt.axis("off")
plt.show()