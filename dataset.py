# -*- coding: utf-8 -*-
"""
Created on Wed Apr  2 11:19:00 2025

@author: admin
"""

import os
import json
import numpy as np
from mrcnn.utils import Dataset
from pycocotools import mask as mask_utils

class TeethDataset(Dataset):
    def __init__(self):
        super().__init__()
        self.image_dir = r'C:\Users\admin\Downloads\video-4\xrays'  # Path to images

    def load_teeth(self, annotation_file):
        """Load the teeth dataset from a COCO JSON file."""
        # Load the COCO JSON file
        with open(annotation_file, 'r') as f:
            coco_data = json.load(f)

        # Add classes (excluding background, which is implicit as class 0)
        for category in coco_data['categories']:
            self.add_class("teeth", category['id'], category['name'])

        # Add images
        for image in coco_data['images']:
            self.add_image(
                source="teeth",
                image_id=image['id'],
                path=os.path.join(self.image_dir, image['file_name']),
                width=image['width'],
                height=image['height'],
                annotations=[ann for ann in coco_data['annotations'] if ann['image_id'] == image['id']]
            )

    def load_mask(self, image_id):
        """Generate instance masks for an image."""
        image_info = self.image_info[image_id]
        annotations = image_info['annotations']

        # Initialize masks and class IDs
        masks = []
        class_ids = []

        for ann in annotations:
            # Decode RLE mask
            rle = ann['segmentation']
            # Ensure RLE counts are in bytes (required by pycocotools)
            if isinstance(rle['counts'], str):
                rle['counts'] = rle['counts'].encode('utf-8')
            mask = mask_utils.decode(rle)  # Decode RLE to binary mask
            masks.append(mask)
            class_ids.append(self.map_source_class_id(f"teeth.{ann['category_id']}"))

        # Stack masks into a single array: [height, width, num_instances]
        if masks:
            masks = np.stack(masks, axis=-1)
        else:
            # If no masks, return an empty array with shape [height, width, 0]
            masks = np.zeros((image_info['height'], image_info['width'], 0), dtype=np.uint8)

        return masks, np.array(class_ids, dtype=np.int32)

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        return info['path']