import cv2
import os
import shutil

def save_webcam_footage(output_dir="webcam_frames", frame_prefix="frame", camera_index=0):
    # Clear the output directory if it exists, otherwise create it
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    # Open the webcam
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Press 'q' to stop capturing frames.")
    frame_count = 0

    while True:
        # Capture a single frame
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break

        # Display the frame
        cv2.imshow("Webcam Footage", frame)

        # Save the frame as an image
        frame_filename = os.path.join(output_dir, f"{frame_prefix}_{frame_count:04d}.png")
        cv2.imwrite(frame_filename, frame)
        print(f"Saved {frame_filename}")
        frame_count += 1

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close the display window
    cap.release()
    cv2.destroyAllWindows()
    print(f"Webcam footage saved as {frame_count} frames in '{output_dir}'.")

if __name__ == "__main__":
    save_webcam_footage()
