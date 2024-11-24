import os
from isolate_hands import isolate_hand

def process_all_images(root_folder, output_root_folder):
    for subdir, _, files in os.walk(root_folder):
        relative_path = os.path.relpath(subdir, root_folder)
        output_folder = os.path.join(output_root_folder, relative_path)

        # Skipping the folders for already traversed becuase the laptop crashed multiple times
        if os.path.exists(output_folder) and any(
            file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))
            for file in os.listdir(output_folder)
        ):
            print(f"Skipping already processed folder: {subdir}")
            continue

        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                input_path = os.path.join(subdir, file)
                output_path = os.path.join(output_folder, file)
                os.makedirs(output_folder, exist_ok=True)

                isolate_hand(input_path, output_path)
                print(f"Processed: {input_path} -> {output_path}")

if __name__ == "__main__":
    input_folder = "Data"  
    output_folder = "Data_Cropped" 
    process_all_images(input_folder, output_folder)
