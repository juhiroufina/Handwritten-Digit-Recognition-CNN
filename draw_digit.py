import tkinter as tk
import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw

# Load trained CNN model
model = tf.keras.models.load_model("handwritten_digit_cnn.keras")

# Window settings
WINDOW_SIZE = 400
CANVAS_SIZE = 280

# Create main window
root = tk.Tk()
root.title("Handwritten Digit Recognition - CNN")
root.geometry("500x600")
root.resizable(False, False)

# Title
title = tk.Label(
    root,
    text="Draw a Digit (0-9)",
    font=("Arial", 20, "bold")
)
title.pack(pady=10)

# Canvas
canvas = tk.Canvas(
    root,
    width=CANVAS_SIZE,
    height=CANVAS_SIZE,
    bg="black"
)
canvas.pack()

# Create image for drawing
image = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), 0)
draw = ImageDraw.Draw(image)


# Draw with mouse
def draw_digit(event):
    x = event.x
    y = event.y

    # Draw thick white line
    radius = 10

    canvas.create_oval(
        x - radius,
        y - radius,
        x + radius,
        y + radius,
        fill="white",
        outline="white"
    )

    draw.ellipse(
        [
            x - radius,
            y - radius,
            x + radius,
            y + radius
        ],
        fill=255
    )


# Predict digit
def predict_digit():
    # Resize image to MNIST size
    img = image.resize((28, 28))

    # Convert to numpy array
    img_array = np.array(img)

    # Normalize pixel values
    img_array = img_array / 255.0

    # Reshape for CNN
    img_array = img_array.reshape(1, 28, 28, 1)

    # Prediction
    prediction = model.predict(img_array, verbose=0)

    predicted_digit = np.argmax(prediction)
    confidence = np.max(prediction) * 100

    # Display result
    result_label.config(
        text=f"Predicted Digit: {predicted_digit}\n"
             f"Confidence: {confidence:.2f}%"
    )


# Clear canvas
def clear_canvas():
    canvas.delete("all")
    draw.rectangle(
        [0, 0, CANVAS_SIZE, CANVAS_SIZE],
        fill=0
    )

    result_label.config(
        text="Draw a digit and click Predict"
    )


# Mouse drawing
canvas.bind("<B1-Motion>", draw_digit)

# Predict button
predict_button = tk.Button(
    root,
    text="Predict",
    font=("Arial", 14, "bold"),
    command=predict_digit,
    width=15
)
predict_button.pack(pady=10)

# Clear button
clear_button = tk.Button(
    root,
    text="Clear",
    font=("Arial", 14),
    command=clear_canvas,
    width=15
)
clear_button.pack(pady=5)

# Result
result_label = tk.Label(
    root,
    text="Draw a digit and click Predict",
    font=("Arial", 16, "bold")
)
result_label.pack(pady=20)

# Start application
root.mainloop()