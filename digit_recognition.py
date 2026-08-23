import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ============================================================
# HANDWRITTEN DIGIT RECOGNITION USING CNN
# ============================================================

print("=" * 50)
print("HANDWRITTEN DIGIT RECOGNITION - CNN")
print("=" * 50)

# ------------------------------------------------------------
# 1. Load MNIST dataset
# ------------------------------------------------------------

print("\nLoading MNIST dataset...")

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

print("Training samples:", x_train.shape[0])
print("Testing samples :", x_test.shape[0])

# ------------------------------------------------------------
# 2. Normalize the images
# ------------------------------------------------------------

x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Add channel dimension
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

# ------------------------------------------------------------
# 3. Data augmentation
# ------------------------------------------------------------

datagen = ImageDataGenerator(
    rotation_range=8,
    width_shift_range=0.08,
    height_shift_range=0.08,
    zoom_range=0.08
)

datagen.fit(x_train)

# ------------------------------------------------------------
# 4. Build improved CNN model
# ------------------------------------------------------------

model = models.Sequential([

    # First CNN block
    layers.Conv2D(
        32,
        (3, 3),
        padding="same",
        activation="relu",
        input_shape=(28, 28, 1)
    ),
    layers.BatchNormalization(),

    layers.Conv2D(
        32,
        (3, 3),
        padding="same",
        activation="relu"
    ),
    layers.BatchNormalization(),

    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.20),

    # Second CNN block
    layers.Conv2D(
        64,
        (3, 3),
        padding="same",
        activation="relu"
    ),
    layers.BatchNormalization(),

    layers.Conv2D(
        64,
        (3, 3),
        padding="same",
        activation="relu"
    ),
    layers.BatchNormalization(),

    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),

    # Third CNN block
    layers.Conv2D(
        128,
        (3, 3),
        padding="same",
        activation="relu"
    ),
    layers.BatchNormalization(),

    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),

    # Fully connected layers
    layers.Flatten(),

    layers.Dense(128, activation="relu"),
    layers.BatchNormalization(),
    layers.Dropout(0.35),

    # Output layer
    layers.Dense(10, activation="softmax")
])

# ------------------------------------------------------------
# 5. Display model architecture
# ------------------------------------------------------------

print("\nCNN MODEL ARCHITECTURE")
print("=" * 50)

model.summary()

# ------------------------------------------------------------
# 6. Compile model
# ------------------------------------------------------------

optimizer = tf.keras.optimizers.Adam(
    learning_rate=0.001
)

model.compile(
    optimizer=optimizer,
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# ------------------------------------------------------------
# 7. Callbacks
# ------------------------------------------------------------

early_stopping = EarlyStopping(
    monitor="val_accuracy",
    patience=5,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=2,
    min_lr=1e-6,
    verbose=1
)

# ------------------------------------------------------------
# 8. Train the CNN
# ------------------------------------------------------------

print("\nStarting CNN training...")
print("=" * 50)

history = model.fit(
    datagen.flow(
        x_train,
        y_train,
        batch_size=128
    ),
    epochs=30,
    validation_data=(x_test, y_test),
    callbacks=[
        early_stopping,
        reduce_lr
    ],
    verbose=1
)

# ------------------------------------------------------------
# 9. Evaluate model
# ------------------------------------------------------------

print("\n")
print("=" * 50)
print("CNN MODEL RESULTS")
print("=" * 50)

test_loss, test_accuracy = model.evaluate(
    x_test,
    y_test,
    verbose=0
)

print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
print(f"Test Loss: {test_loss:.4f}")

# ------------------------------------------------------------
# 10. Save model
# ------------------------------------------------------------

model.save("handwritten_digit_cnn.keras")

print("\nModel saved as: handwritten_digit_cnn.keras")
print("Project completed successfully!")