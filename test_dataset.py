# -*- coding: utf-8 -*-
"""
Created on Wed Apr  2 11:19:34 2025

@author: admin
"""

from dataset import TeethDataset

# Load and test the training dataset
dataset_train = TeethDataset()
dataset_train.load_teeth('train.json')
dataset_train.prepare()

# Load and test the validation dataset
dataset_val = TeethDataset()
dataset_val.load_teeth('val.json')
dataset_val.prepare()

# Print some information to verify
print(f"Training images: {len(dataset_train.image_ids)}")
print(f"Validation images: {len(dataset_val.image_ids)}")
print(f"Classes: {dataset_train.class_names}")

# Test loading a mask for the first training image
image_id = dataset_train.image_ids[0]
masks, class_ids = dataset_train.load_mask(image_id)
print(f"Masks shape for image {image_id}: {masks.shape}")
print(f"Class IDs: {class_ids}")