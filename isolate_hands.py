import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5)

def isolate_hands(image_path, output_path):
    # Load the image
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Process the image to detect hands
    results = hands.process(image_rgb)

    if results.multi_hand_landmarks:
        mask = np.zeros_like(image, dtype=np.uint8)  # Create a blank mask for the hands

        for hand_landmarks in results.multi_hand_landmarks:
            # Convert landmarks to pixel coordinates
            h, w, _ = image.shape
            hand_points = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks.landmark]

            # Create a polygon mask around the hand
            hand_contour = np.array(hand_points, dtype=np.int32)
            cv2.fillPoly(mask, [hand_contour], (255, 255, 255))

        # Apply the mask to the original image
        hand_isolated = cv2.bitwise_and(image, mask)

        # Save the isolated hand image
        cv2.imwrite(output_path, hand_isolated)
        print(f"Hand isolated and saved to {output_path}")
    else:
        print("No hands detected in the image.")

# Example usage
input_image = "input.jpg"  # Path to your input image
output_image = "hand_isolated.jpg"  # Path to save the isolated hand image
isolate_hands(input_image, output_image)
