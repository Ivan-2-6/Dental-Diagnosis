# -*- coding: utf-8 -*-
"""
Created on Wed Apr  2 10:44:43 2025

@author: admin
"""

import json
import random

# Load the consolidated COCO JSON file
with open('consolidated_coco.json', 'r') as f:
    coco_data = json.load(f)

# Get all image IDs
image_ids = [img["id"] for img in coco_data["images"]]

# Shuffle the image IDs to ensure a random split
random.seed(42)  # For reproducibility; you can change or remove this
random.shuffle(image_ids)

# Calculate split sizes: 80% training, 20% validation
total_images = len(image_ids)
train_size = int(0.8 * total_images)  # 478 images
val_size = total_images - train_size  # 120 images

# Split image IDs into training and validation sets
train_image_ids = image_ids[:train_size]
val_image_ids = image_ids[train_size:]

# Create training and validation datasets
train_data = {
    "images": [],
    "annotations": [],
    "categories": coco_data["categories"]
}

val_data = {
    "images": [],
    "annotations": [],
    "categories": coco_data["categories"]
}

# Assign images to training and validation sets
for img in coco_data["images"]:
    if img["id"] in train_image_ids:
        train_data["images"].append(img)
    else:
        val_data["images"].append(img)

# Assign annotations to training and validation sets based on image_id
for ann in coco_data["annotations"]:
    if ann["image_id"] in train_image_ids:
        train_data["annotations"].append(ann)
    else:
        val_data["annotations"].append(ann)

# Save the training and validation JSON files
with open('train.json', 'w') as f:
    json.dump(train_data, f, indent=4)

with open('val.json', 'w') as f:
    json.dump(val_data, f, indent=4)

# Print the sizes of the splits for verification
print(f"Training set: {len(train_data['images'])} images, {len(train_data['annotations'])} annotations")
print(f"Validation set: {len(val_data['images'])} images, {len(val_data['annotations'])} annotations")