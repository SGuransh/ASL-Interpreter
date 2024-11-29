import cv2
import mediapipe as mp
import numpy as np
def capture_image_from_webcam(output_path="webcam_frame.png"):
    # Capture an image from the webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return None

    # Warm-up frames to adjust the camera settings
    for _ in range(30):
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame from webcam.")
            cap.release()
            return None

    # Capture the final frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame from webcam.")
        cap.release()
        return None

    # Save the captured frame
    cv2.imwrite(output_path, frame)
    cap.release()
    return output_path

def isolate_hand(image_path, output_path="isolated_hand.png"):
    # Initialize MediaPipe Hands
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5
    )
    mp_drawing = mp.solutions.drawing_utils

    # Read the input image
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Could not load image.")
        return False

    # Convert the image to RGB (MediaPipe requires RGB images)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Process the image and detect hands
    results = hands.process(image_rgb)

    if not results.multi_hand_landmarks:
        print("No hand detected in the image.")
        return False

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

        # Resize the cropped hand to 256x256 and convert to grayscale
        resized_hand = cv2.resize(cropped_hand, (256, 256))
        grayscale_hand = cv2.cvtColor(resized_hand, cv2.COLOR_BGR2GRAY)

        # Save and display the grayscale hand
        cv2.imwrite(output_path, grayscale_hand)
        print(f"Hand isolated, resized, and saved to {output_path}")

        #cv2.imshow("Isolated Hand (Grayscale)", grayscale_hand)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()

    # Release MediaPipe resources
    hands.close()
    return True

if __name__ == "__main__":
    # Capture an image from the webcam
    input_image = capture_image_from_webcam()
    isolate_hand(input_image)
