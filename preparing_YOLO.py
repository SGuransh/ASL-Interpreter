import cv2
import numpy as np

# I know Guransh is working on his version; I made this using ChatGPT suggestions

# Example of extracting hand region using YOLO
def preprocess_image(image_path):
    # Load image using OpenCV
    img = cv2.imread(image_path)
    # Assuming 'hand_roi' is obtained using YOLO
    hand_roi = cv2.resize(img, (224, 224))
    # Normalize the image
    hand_roi = hand_roi / 255.0
    return hand_roi
