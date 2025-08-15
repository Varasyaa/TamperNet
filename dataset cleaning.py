import os
import numpy as np
from tensorflow.keras.preprocessing import image
from PIL import Image
import tensorflow as tf

# Paths to the dataset (change these paths accordingly)
auth_path = "/content/drive/MyDrive/CASIA2/Au"
tamp_path = "/content/drive/MyDrive/CASIA2/Tp"
cleaned_auth_path = "/content/drive/MyDrive/CASIA2/Au_cleaned"
cleaned_tamp_path = "/content/drive/MyDrive/CASIA2/Tp_cleaned"

# Create directories for cleaned data if they don't exist
os.makedirs(cleaned_auth_path, exist_ok=True)
os.makedirs(cleaned_tamp_path, exist_ok=True)

# Function to clean and process dataset
def clean_and_process_images(input_path, output_path, target_size=(128, 128)):
    for fname in os.listdir(input_path):
        file_path = os.path.join(input_path, fname)
        try:
            # Open image to check if it's valid
            img = Image.open(file_path)
            img = img.convert("RGB")  # Ensure it's RGB

            # Resize image
            img = img.resize(target_size)

            # Save cleaned image to new directory
            cleaned_file_path = os.path.join(output_path, fname)
            img.save(cleaned_file_path)

        except Exception as e:
            print(f"Skipping {file_path}: {e}")

# Clean both authentic and tampered datasets
clean_and_process_images(auth_path, cleaned_auth_path)
clean_and_process_images(tamp_path, cleaned_tamp_path)

print("✅ Dataset cleaning completed!")
