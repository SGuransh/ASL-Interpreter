import cv2
import os

def resize_image(input_path, output_path, size=(513, 512)):
    img = cv2.imread(input_path)
    if img is None:
        print(f"Error: Image not found")
        return
    resized_img = cv2.resize(img, size)
    os.makedirs(os.path.dirname(output_path), exist_ok=True) 
    cv2.imwrite(output_path, resized_img)
    print(f"Image resized: {output_path}")

def process_images(input_directory, output_directory, size=(513, 512)):
    for root, dirs, files in os.walk(input_directory): 
        for file in files:
            input_path = os.path.join(root, file)  
            relative_path = os.path.relpath(input_path, input_directory)
            output_path = os.path.join(output_directory, relative_path)
            resize_image(input_path, output_path, size)

if __name__ == "__main__":
    input_directory = "to_resize"
    output_directory = "resized_hands"      
    process_images(input_directory, output_directory)
