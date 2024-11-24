import cv2
import mediapipe as mp
import numpy as np

def isolate_hand(image_path, output_path="isolated_hand.png"):
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)
    image = cv2.imread(image_path)
    if image is None:
        print("Not able to load the image")
        return
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    if not results.multi_hand_landmarks:
        print("Null coords for hands")
        return

    # Bound for the hands
    height, width, _ = image.shape
    for hand_landmarks in results.multi_hand_landmarks:
        # Extract landmark coordinates
        x_min, y_min = width, height
        x_max, y_max = 0, 0
        for lm in hand_landmarks.landmark:
            x, y = int(lm.x * width), int(lm.y * height)
            x_min, y_min = min(x_min, x), min(y_min, y)
            x_max, y_max = max(x_max, x), max(y_max, y)

        padding = 20
        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)
        x_max = min(width, x_max + padding)
        y_max = min(height, y_max + padding)

        cropped_hand = image[y_min:y_max, x_min:x_max]

        cv2.imwrite(output_path, cropped_hand)
        print(f"Hand isolated and saved to {output_path}")

        cv2.destroyAllWindows()
    hands.close()

if __name__ == "__main__":
    input_image = "webcam_frames/frame_0000.png"  
    isolate_hand(input_image)