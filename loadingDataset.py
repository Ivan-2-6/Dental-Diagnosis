"""
COCO Annotation Converter for Dental X-ray Masks
Handles base64-encoded masks with error resilience and Windows path support
"""

import json
import os
import base64
import logging
import numpy as np
from pathlib import Path
from tqdm import tqdm
from pycocotools import mask as mask_utils
from typing import Dict, List, Optional, Tuple

# Configure logging
logging.basicConfig(
    filename='conversion_errors.log',
    level=logging.WARNING,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class CocoConverter:
    def __init__(self, image_dir: str, annotation_dir: str):
        self.image_dir = Path(image_dir)
        self.annotation_dir = Path(annotation_dir)
        self._validate_directories()
        
        self.coco_data = {
            "images": [],
            "annotations": [],
            "categories": [],
            "info": {
                "description": "Dental X-ray Dataset",
                "version": "1.0",
                "year": 2023
            }
        }
        self.category_map: Dict[int, str] = {}
        self.image_id = 1
        self.annotation_id = 1

    def _validate_directories(self):
        """Ensure input directories exist and are accessible"""
        if not self.image_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {self.image_dir}")
        if not self.annotation_dir.exists():
            raise FileNotFoundError(f"Annotation directory not found: {self.annotation_dir}")

    @staticmethod
    def decode_base64_mask(
        base64_str: str,
        origin: Tuple[int, int],
        image_size: Tuple[int, int]
    ) -> Optional[np.ndarray]:
        """
        Decode base64 mask data and position it in the full image context
        with comprehensive error handling
        """
        try:
            # Base64 decoding with padding check
            missing_padding = len(base64_str) % 4
            if missing_padding:
                base64_str += '=' * (4 - missing_padding)
                
            decoded_bytes = base64.b64decode(base64_str, validate=True)
            if len(decoded_bytes) < 4:
                logging.warning(f"Invalid mask data: {base64_str[:50]}...")
                return None

            # Convert to numpy array
            mask_flat = np.frombuffer(decoded_bytes, dtype=np.uint8)
            
            # Determine optimal mask dimensions
            data_length = len(mask_flat)
            factors = [
                (i, data_length // i)
                for i in range(1, int(np.sqrt(data_length)) + 1)
                if data_length % i == 0
            ]
            
            if not factors:
                logging.warning(f"Unfactorable mask length: {data_length}")
                return None
                
            # Select dimensions closest to square
            height, width = max(factors, key=lambda x: min(x))
            mask = mask_flat.reshape((height, width))

            # Create full-size mask
            img_height, img_width = image_size
            full_mask = np.zeros((img_height, img_width), dtype=np.uint8)
            
            # Calculate safe placement coordinates
            x, y = origin
            end_y = min(y + height, img_height)
            end_x = min(x + width, img_width)
            
            valid_height = end_y - y
            valid_width = end_x - x
            
            if valid_height <= 0 or valid_width <= 0:
                logging.warning(f"Invalid mask placement: {origin} in {image_size}")
                return None

            full_mask[y:end_y, x:end_x] = mask[:valid_height, :valid_width]
            return full_mask

        except (base64.binascii.Error, ValueError) as e:
            logging.error(f"Base64 decoding failed: {str(e)}")
            return None
        except Exception as e:
            logging.error(f"Unexpected mask error: {str(e)}")
            return None

    def _process_single_image(self, image_idx: int):
        """Process a single image and its annotations"""
        img_file = f"{image_idx}.jpg"
        ann_file = self.annotation_dir / f"{img_file}.json"

        if not ann_file.exists():
            logging.warning(f"Missing annotation: {ann_file.name}")
            return

        try:
            with open(ann_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            logging.error(f"Invalid JSON in {ann_file.name}: {str(e)}")
            return

        # Validate required fields
        if 'size' not in data or 'width' not in data['size'] or 'height' not in data['size']:
            logging.warning(f"Invalid size data in {ann_file.name}")
            return

        # Register image
        self.coco_data['images'].append({
            "id": self.image_id,
            "file_name": img_file,
            "width": data['size']['width'],
            "height": data['size']['height'],
            "license": 1  # Add if you have license information
        })

        # Process annotations
        for obj in data.get('objects', []):
            if not all(k in obj for k in ('classId', 'classTitle', 'bitmap')):
                logging.warning("Invalid object structure, skipping")
                continue

            class_id = obj['classId']
            class_title = obj['classTitle']

            # Register category
            if class_id not in self.category_map:
                self.category_map[class_id] = class_title
                self.coco_data['categories'].append({
                    "id": class_id,
                    "name": class_title,
                    "supercategory": "tooth"
                })

            # Decode and validate mask
            mask = self.decode_base64_mask(
                obj['bitmap']['data'],
                tuple(obj['bitmap']['origin']),
                (data['size']['height'], data['size']['width'])
            )
            
            if mask is None or mask.sum() == 0:
                continue

            # Create COCO annotation
            try:
                rows, cols = np.where(mask)
                x_min, x_max = cols.min(), cols.max()
                y_min, y_max = rows.min(), rows.max()
                bbox = [int(x_min), int(y_min), 
                        int(x_max - x_min), int(y_max - y_min)]

                rle = mask_utils.encode(np.asfortranarray(mask))
                rle['counts'] = rle['counts'].decode('utf-8')

                self.coco_data['annotations'].append({
                    "id": self.annotation_id,
                    "image_id": self.image_id,
                    "category_id": class_id,
                    "bbox": bbox,
                    "segmentation": rle,
                    "area": int(mask.sum()),
                    "iscrowd": 0,
                    "attributes": {}
                })
                self.annotation_id += 1
            except Exception as e:
                logging.error(f"Annotation creation failed: {str(e)}")
                continue

        self.image_id += 1

    def convert(self, output_path: str):
        """Main conversion method with progress tracking"""
        print(f"Processing {len(list(self.annotation_dir.glob('*.json')))} annotations...")
        
        for idx in tqdm(range(1, 599), desc="Converting annotations"):
            try:
                self._process_single_image(idx)
            except Exception as e:
                logging.critical(f"Critical error processing image {idx}: {str(e)}")
                continue

        self._save_output(output_path)
        print(f"\nConversion complete! Results saved to {output_path}")
        print(f"Processed: {len(self.coco_data['images'])} images, "
              f"{len(self.coco_data['annotations'])} annotations, "
              f"{len(self.coco_data['categories'])} categories")

    def _save_output(self, output_path: str):
        """Save COCO data with proper formatting"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.coco_data, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    # Windows path handling with raw strings
    converter = CocoConverter(
        image_dir=r"C:\Users\admin\Downloads\video-4\xrays",
        annotation_dir=r"C:\Users\admin\Downloads\video-4\annots"
    )
    converter.convert(output_path="consolidated_coco.json")