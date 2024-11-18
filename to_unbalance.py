import os
import numpy as np
import random

# Define the path to the dataset
dataset_path = "E:/Internship/tiny-imagenet-200/tiny-imagenet-200/train"

# Get the list of all class folders in the dataset
class_folders = [os.path.join(dataset_path, folder) for folder in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, folder))]

# Randomly select 50 folders
selected_folders = np.random.choice(class_folders, size=50, replace=False)

deleted_images_info = {}

# Iterate through each selected folder
for folder in selected_folders:
    class_name = os.path.basename(folder)
    images = [os.path.join(folder, "images", img) for img in os.listdir(os.path.join(folder, "images")) if os.path.isfile(os.path.join(folder, "images", img))]
    
    num_images_to_delete = np.random.randint(150, 250)
    
    images_to_delete = random.sample(images, min(num_images_to_delete, len(images)))
    
    # Delete each selected image
    for image_path in images_to_delete:
        os.remove(image_path)
        print(f"Deleted: {image_path}")
    
    # Store the class name and number of deleted images in the dictionary
    deleted_images_info[class_name] = len(images_to_delete)

print("Completed deletion process.")
print(deleted_images_info)
