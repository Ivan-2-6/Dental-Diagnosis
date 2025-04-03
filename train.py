# -*- coding: utf-8 -*-
"""
Created on Wed Apr  2 11:29:38 2025

@author: admin
"""

import os
from mrcnn.model import MaskRCNN
from config import TeethConfig
from dataset import TeethDataset

# Define paths
model_dir = r'C:\Users\admin\Downloads\video-4\logs'  # Directory to save training logs and weights
coco_weights_path = r'C:\Users\admin\Downloads\video-4\mask_rcnn_coco.h5'  # Path to pretrained COCO weights
final_weights_path = r'C:\Users\admin\Downloads\video-4\mask_rcnn_teeth_final.h5'  # Path to save final trained weights

# Create the logs directory if it doesn’t exist
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

# Load training dataset
dataset_train = TeethDataset()
dataset_train.load_teeth('train.json')
dataset_train.prepare()

# Load validation dataset
dataset_val = TeethDataset()
dataset_val.load_teeth('val.json')
dataset_val.prepare()

# Initialize the model in training mode
config = TeethConfig()
model = MaskRCNN(mode="training", config=config, model_dir=model_dir)

# Load pretrained COCO weights
print("Loading pretrained COCO weights...")
model.load_weights(
    coco_weights_path,
    by_name=True,
    exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"]
)

# Train the model
print("Starting training...")
model.train(
    dataset_train,
    dataset_val,
    learning_rate=config.LEARNING_RATE,
    epochs=5,  # Number of epochs; adjust as needed
    layers='heads'  # Start by training only the heads (faster and more stable)
)

# Save the final trained weights
print("Saving final trained weights...")
model.keras_model.save_weights(final_weights_path)

print(f"Training complete! Final weights saved to: {final_weights_path}")