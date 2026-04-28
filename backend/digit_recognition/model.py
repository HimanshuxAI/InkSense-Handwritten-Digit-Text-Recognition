"""
Handwritten Digit Recognition Model
Based on: https://github.com/aakashjhawar/handwritten-digit-recognition

Uses a CNN trained on MNIST dataset to recognize handwritten digits (0-9).
The model architecture: Conv2D -> Conv2D -> MaxPool -> Conv2D -> MaxPool -> Conv2D -> MaxPool -> Dense -> Dense
"""

import os
import numpy as np
import cv2
from tensorflow.keras import models, layers
from tensorflow.keras.models import Sequential

# Path to the pre-trained model
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
MODEL_PATH = os.path.join(MODEL_DIR, 'digit_model.h5')


def create_model():
    """Create the CNN model architecture for digit recognition."""
    model = Sequential([
        layers.Conv2D(64, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def train_and_save_model():
    """Train the digit recognition model on MNIST and save it."""
    from tensorflow.keras import datasets

    print("[INFO] Loading MNIST dataset...")
    (train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()

    # Reshape and normalize
    train_images = train_images.reshape((60000, 28, 28, 1)).astype('float32') / 255.0
    test_images = test_images.reshape((10000, 28, 28, 1)).astype('float32') / 255.0

    print(f"[INFO] Train images shape: {train_images.shape}")
    print(f"[INFO] Test images shape: {test_images.shape}")

    # Create and train model
    model = create_model()
    model.summary()

    print("[INFO] Training model (3 epochs)...")
    history = model.fit(
        train_images, train_labels,
        epochs=3,
        validation_data=(test_images, test_labels),
        batch_size=256
    )

    # Evaluate
    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=0)
    print(f"[INFO] Test accuracy: {test_acc * 100:.2f}%")

    # Save model
    os.makedirs(MODEL_DIR, exist_ok=True)
    model.save(MODEL_PATH)
    print(f"[INFO] Model saved to {MODEL_PATH}")

    return history


class DigitRecognizer:
    """Handles digit recognition inference."""

    def __init__(self):
        self.model = None
        self._load_model()

    def _load_model(self):
        """Load the pre-trained model, or train a new one if not found."""
        if os.path.exists(MODEL_PATH):
            print("[INFO] Loading digit recognition model...")
            self.model = models.load_model(MODEL_PATH)
            print("[INFO] Digit recognition model loaded successfully.")
        else:
            print("[INFO] No pre-trained digit model found. Training a new one...")
            train_and_save_model()
            self.model = models.load_model(MODEL_PATH)
            print("[INFO] Digit recognition model trained and loaded.")

    def preprocess_image(self, img):
        """Preprocess image for digit recognition."""
        # Convert to grayscale if needed
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Resize to 28x28
        img = cv2.resize(img, (28, 28))

        # Invert if background is white (MNIST has white digits on black background)
        if np.mean(img) > 127:
            img = 255 - img

        # Normalize
        img = img.astype('float32') / 255.0

        # Reshape for model input
        img = img.reshape(1, 28, 28, 1)

        return img

    def predict(self, image):
        """
        Predict the digit in the image.

        Args:
            image: numpy array of the image (can be any size, will be preprocessed)

        Returns:
            dict with 'digit' (predicted digit), 'confidence' (probability),
            and 'probabilities' (all class probabilities)
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")

        processed = self.preprocess_image(image)
        predictions = self.model.predict(processed, verbose=0)
        predicted_digit = int(np.argmax(predictions[0]))
        confidence = float(predictions[0][predicted_digit])
        probabilities = {str(i): float(predictions[0][i]) for i in range(10)}

        return {
            'digit': predicted_digit,
            'confidence': confidence,
            'probabilities': probabilities
        }

    def predict_from_canvas(self, image_data):
        """
        Predict digit from canvas drawing data (already preprocessed as numpy array).
        """
        return self.predict(image_data)
