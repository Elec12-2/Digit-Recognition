import os
import cv2
import logging
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from skimage.feature import hog  # For HOG feature extraction
from skimage import exposure  # For better visualization of HOG features

# Project Introduction: Digit Recognition System with CNN and Feature Extraction
# This project aims to build a system capable of accurately identifying handwritten digits.
# We will use the MNIST dataset and apply various feature extraction techniques like pixel intensities, 
# HOG (Histogram of Oriented Gradients), edge detection, and more to enhance the recognition system.

# Step 1: Load MNIST Dataset
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Step 2: Preprocess the Data (Normalization)
# Preprocessing techniques like normalization are applied to standardize the images.
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

# Step 3: Build a Basic CNN Model
model = tf.keras.models.Sequential()

# Flatten layer to convert the 2D image into 1D
model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))

# Fully connected layers
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(128, activation='relu'))

# Output layer for 10 classes (digits 0-9)
model.add(tf.keras.layers.Dense(10, activation='softmax'))

# Step 4: Compile the CNN Model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Step 5: Train the Model
model.fit(x_train, y_train, epochs=3)

# Step 6: Save the Model
model.save('handwritten.h5')

# Step 7: Load the Saved Model
model = tf.keras.models.load_model('handwritten.h5')

# Optional: Evaluate the model performance on the test dataset
# loss, accuracy = model.evaluate(x_test, y_test)
# print(f'Loss: {loss}, Accuracy: {accuracy}')

# Step 8: Feature Extraction Techniques
# ----------------------------------------

# Example of using Pixel Intensities as Features:
# Each pixel's intensity in the grayscale image is already used as a feature by CNN.

# Example of using HOG (Histogram of Oriented Gradients) for feature extraction:
# HOG captures shape information by computing the distribution of gradient orientations in localized portions of the image.

def apply_hog(image):
    hog_features, hog_image = hog(image, pixels_per_cell=(14, 14), cells_per_block=(2, 2), visualize=True)
    return hog_features, hog_image

# Load and process images for prediction with HOG features and CNN
image_number = 1
while os.path.isfile(f"digits/digit{image_number}.png"):
    try:
        img = cv2.imread(f"digits/digit{image_number}.png", cv2.IMREAD_GRAYSCALE)
        img = np.invert(np.array([img]))
        
        # Resize the image to (28, 28) if needed
        img_resized = cv2.resize(img[0], (28, 28))

        # Apply HOG feature extraction
        hog_features, hog_image = apply_hog(img_resized)
        
        # Show HOG visualization
        plt.figure(figsize=(8, 4))
        plt.subplot(1, 2, 1)
        plt.title("Original Image")
        plt.imshow(img_resized, cmap=plt.cm.gray)
        
        plt.subplot(1, 2, 2)
        plt.title("HOG Features")
        plt.imshow(hog_image, cmap=plt.cm.gray)
        plt.show()

        # Use the CNN model to predict based on original image (CNN doesn't require manual feature extraction)
        prediction = model.predict(img)
        print(f"This digit is probably a {np.argmax(prediction)}")

    except Exception as e:
        print(f"Error processing image: {e}")
    finally:
        image_number += 1

# Additional Feature Extraction Techniques:
# ----------------------------------------

# Edge Detection: Features can be derived from detected edges using Sobel or Canny edge detectors.
# Below is an example of applying Sobel edge detection:
def sobel_edge_detection(image):
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)  # Horizontal edges
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)  # Vertical edges
    edges = np.hypot(sobelx, sobely)
    return edges

# Correct Harris Corner Detection Function
def harris_corner_detection(image):
    # Convert image to float32 as required by cornerHarris
    gray = np.float32(image)
    
    # Apply Harris corner detector
    dst = cv2.cornerHarris(gray, blockSize=2, ksize=3, k=0.04)

    # Dilate the result for marking the corners
    dst = cv2.dilate(dst, None)

    # Threshold for optimal value, marking the corners in the original image
    image[dst > 0.01 * dst.max()] = 255
    
    return dst