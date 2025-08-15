import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.utils import to_categorical, load_img, img_to_array
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import save_model

# Paths to the cleaned dataset
cleaned_auth_path = "/content/drive/MyDrive/CASIA2/Au_cleaned"
cleaned_tamp_path = "/content/drive/MyDrive/CASIA2/Tp_cleaned"

# Load dataset and preprocess images
def load_casia_dataset(auth_path, tamp_path, img_size=(128, 128)):
    images, labels = [], []
    for label, path in enumerate([auth_path, tamp_path]):
        for fname in os.listdir(path):
            fpath = os.path.join(path, fname)
            try:
                img = load_img(fpath, target_size=img_size)  # Load image and resize
                img = img_to_array(img)  # Convert image to numpy array
                images.append(img)
                labels.append(label)  # 0 for authentic, 1 for tampered
            except Exception as e:
                print(f"Skipping {fpath}: {e}")
    return np.array(images, dtype="float32"), np.array(labels)

# Load and preprocess data
X, y = load_casia_dataset(cleaned_auth_path, cleaned_tamp_path)
X = preprocess_input(X)  # Apply MobileNetV2 preprocessing
y = to_categorical(y, 2)  # One-hot encode the labels

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the model using MobileNetV2 as base
def build_mobilenet_model(input_shape=(128, 128, 3)):
    base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=input_shape)
    base_model.trainable = False  # Freeze the base model layers

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation="relu"),
        layers.Dense(2, activation="softmax")  # Output layer for binary classification
    ])

    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model

# Initialize and train the model
model = build_mobilenet_model()
early_stop = EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True)

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.1, callbacks=[early_stop])

# Evaluate the model on the test set
loss, acc = model.evaluate(X_test, y_test)
print(f"✅ Test Accuracy: %2f{acc}")

# Save the trained model
model.save("/content/drive/MyDrive/tampernet_model_trained_mobilenet.h5")
print("✅ Model saved to Drive as tampernet_model_trained_mobilenet.h5")
