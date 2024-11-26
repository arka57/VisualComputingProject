import os
import random
import shutil

# Paths to the current and source directories
current_dataset_path = "E:/Augumented/Balanced/train"
source_dataset_path = "E:/ALL DATA/Synthetic-Imagenet1_4/train"  # Directory containing replacement images

# Number of images missing per class
deleted_counts = {
    'n02279972': 214, 'n02281406': 241, 'n02917067': 185, 'n03837869': 158, 'n02074367': 165,
    'n04149813': 181, 'n01910747': 222, 'n04146614': 232, 'n03026506': 243, 'n06596364': 175,
    'n02094433': 182, 'n04562935': 228, 'n04486054': 190, 'n04487081': 229, 'n01917289': 243,
    'n03393912': 196, 'n01443537': 222, 'n02002724': 195, 'n03733131': 227, 'n01984695': 154,
    'n02056570': 235, 'n02814860': 162, 'n03355925': 234, 'n02669723': 186, 'n02165456': 157,
    'n04560804': 243, 'n04596742': 151, 'n04070727': 199, 'n02423022': 167, 'n04311004': 249,
    'n02481823': 226, 'n01774384': 227, 'n03447447': 239, 'n04376876': 189, 'n02509815': 150,
    'n02113799': 159, 'n01768244': 155
}

# Iterate over each class in deleted_counts
for class_name, num_images_to_import in deleted_counts.items():
    # Paths to the class folders in the current and source directories
    current_class_folder = os.path.join(current_dataset_path, class_name, "images")
    source_class_folder = os.path.join(source_dataset_path, class_name, "images")

    # List all .png files in the source class folder
    png_images = [img for img in os.listdir(source_class_folder) if img.endswith(".png")]

    # Randomly select the required number of images
    images_to_import = random.sample(png_images, num_images_to_import)

    # Copy each selected image to the current directory
    for img_name in images_to_import:
        src_path = os.path.join(source_class_folder, img_name)
        dst_path = os.path.join(current_class_folder, img_name)
        
        shutil.copy(src_path, dst_path)
        print(f"Copied {img_name} to {current_class_folder}")

print("Image import process completed.")
