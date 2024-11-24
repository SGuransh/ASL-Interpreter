from ultralytics import YOLO
import cv2

# Load the pre-trained YOLOv5 model
model = YOLO('yolov5su.pt')  # You can replace 'yolov5s.pt' with the specific model you want

# Load an image to detect hands in
img_path = 'O.png'  # Replace with the path to your image
img = cv2.imread(img_path)

# Perform inference (hand detection)
results = model(img)

# Access the first result from the list of results
result = results[0]  # Get the first result

# Show the results (bounding boxes around detected objects)
result.show()  # Display the image with bounding boxes

# Extract the bounding boxes and crop the image around hands
for pred in result.boxes:
    # Assuming class 0 represents "person" in a general YOLOv5 model
    # For a specific hand detection model, adjust the class index as needed
    if pred.cls == 0:  # Adjust class index based on your model (hands may not be class 0)
        x_center, y_center, width, height = pred.xywh[0]
        
        # Calculate pixel coordinates of the bounding box
        x1 = int((x_center - width / 2) * img.shape[1])
        y1 = int((y_center - height / 2) * img.shape[0])
        x2 = int((x_center + width / 2) * img.shape[1])
        y2 = int((y_center + height / 2) * img.shape[0])

        # Ensure the coordinates are within valid image bounds
        if x1 >= 0 and y1 >= 0 and x2 <= img.shape[1] and y2 <= img.shape[0]:
            # Crop the hand region from the image
            cropped_hand = img[y1:y2, x1:x2]

            # Display the cropped hand
            cv2.imshow("Cropped Hand", cropped_hand)
            cv2.waitKey(0)  # Wait for a key press before closing
            cv2.destroyAllWindows()
        else:
            print("Invalid bounding box coordinates, skipping cropping.")