
import cv2
import mediapipe as mp
import numpy as np

def isolate_hand(image_path, output_path="isolated_hand.png"):
    # Initialize MediaPipe Hands
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)
    mp_drawing = mp.solutions.drawing_utils

    # Read the input image
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Could not load image.")
        return

    # Convert the image to RGB (MediaPipe requires RGB images)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Process the image and detect hands
    results = hands.process(image_rgb)

    if not results.multi_hand_landmarks:
        print("No hand detected in the image.")
        return

    # Get the bounding box of the detected hand
    h, w, _ = image.shape
    for hand_landmarks in results.multi_hand_landmarks:
        # Extract landmark coordinates
        x_min, y_min = w, h
        x_max, y_max = 0, 0
        for lm in hand_landmarks.landmark:
            x, y = int(lm.x * w), int(lm.y * h)
            x_min, y_min = min(x_min, x), min(y_min, y)
            x_max, y_max = max(x_max, x), max(y_max, y)

        # Add some padding to the bounding box
        padding = 20
        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)
        x_max = min(w, x_max + padding)
        y_max = min(h, y_max + padding)

        # Crop the hand region
        cropped_hand = image[y_min:y_max, x_min:x_max]

        # Save and display the cropped hand
        cv2.imwrite(output_path, cropped_hand)
        print(f"Hand isolated and saved to {output_path}")

        cv2.imshow("Isolated Hand", cropped_hand)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # Release MediaPipe resources
    hands.close()

if __name__ == "__main__":
    input_image = "webcam_frames/frame_0000.png"  # Replace with your input image path
    isolate_hand(input_image)
