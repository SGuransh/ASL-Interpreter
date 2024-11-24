import os
import random
import shutil

def split_data(train_dir, val_dir, val_split=0.2, seed=42):
    """
    Making a validation folder for the data to train the model
    """
    random.seed(seed)
    
    if not os.path.exists(val_dir):
        os.makedirs(val_dir)
    
    for class_name in os.listdir(train_dir):
        class_train_path = os.path.join(train_dir, class_name)
        class_val_path = os.path.join(val_dir, class_name)
        
        if not os.path.isdir(class_train_path):
            continue

        if not os.path.exists(class_val_path):
            os.makedirs(class_val_path)
        
        # Making a list of files to shuffle
        images = [f for f in os.listdir(class_train_path) if os.path.isfile(os.path.join(class_train_path, f))]
        
        # Using random to shuffle
        random.shuffle(images)
        split_index = int(len(images) * val_split)
        
        for img in images[:split_index]:
            src = os.path.join(class_train_path, img)
            dst = os.path.join(class_val_path, img)
            shutil.move(src, dst)
        print(f"Moved {split_index} images from {class_train_path} to {class_val_path}")

train_dir = "Data/Train_Alphabet"
val_dir = "Data/Validation_Alphabet"

split_data(train_dir, val_dir, val_split=0.2)
