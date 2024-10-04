import logging
import os
from typing import List

import cv2
import numpy as np
from PIL import Image
from transformers import pipeline 
from controlnet_aux import OpenposeDetector 

log = logging.getLogger(__name__)

segmenter = pipeline(model="mattmdjaga/segformer_b2_clothes")
openpose = OpenposeDetector.from_pretrained('lllyasviel/ControlNet')

class ControlnetUtil:
    @staticmethod
    def preprocess_canny(img: Image.Image) -> Image.Image:
        image_to_canny = np.array(img)
        image_to_canny = cv2.Canny(image_to_canny, 100, 200)
        image_to_canny = np.array(image_to_canny[:, :, None])
        image_to_canny = np.concatenate([image_to_canny, image_to_canny, image_to_canny], axis=2)
        return Image.fromarray(image_to_canny)
    
    @staticmethod
    def preprocess_mask(img: Image.Image) -> Image.Image:
        image_to_mask = np.array(img)
        segments = segmenter(image_to_mask)
        segment_include = ["Upper-clothes", "Skirt", "Pants", "Dress", "Belt", "Bag", "Scarf", "Left-shoe", "Right-shoe"]
        mask_list = [np.array(s['mask']) for s in segments if s['label'] not in segment_include]
        final_mask = np.array(mask_list[0]) if mask_list else np.zeros_like(image_to_mask[:,:,0])
        for mask in mask_list[1:]:
            current_mask = np.array(mask)
            final_mask = final_mask + current_mask
        image_to_mask = Image.fromarray(final_mask)
        image_to_mask = np.array(image_to_mask)[:, :, None]
        image_to_mask = np.concatenate([image_to_mask, image_to_mask, image_to_mask], axis=2)
        return Image.fromarray(image_to_mask)
    
    @staticmethod
    def preprocess_pose(img: Image.Image) -> Image.Image:
        image_to_pose = np.array(img)
        image_to_pose = openpose(image_to_pose)
        # Ensure the output is in the correct shape and type
        image_to_pose = np.squeeze(image_to_pose)  # Remove single-dimensional entries
        image_to_pose = np.clip(image_to_pose, 0, 255).astype(np.uint8)  # Ensure values are in the correct range and type
        image_to_pose = image_to_pose[:, :, None]  # Add channel dimension if necessary
        image_to_pose = np.concatenate([image_to_pose, image_to_pose, image_to_pose], axis=2)
        return Image.fromarray(image_to_pose)
    
    @staticmethod
    def scale_image(height: int, width: int, img: Image.Image) -> Image.Image:
        if height != img.height or width != img.width:
            log.warning(f"Control image has different dimensions than the model. Resizing to {width}x{height}")
            img = img.resize((width, height), Image.LANCZOS)
        return img

    @staticmethod
    def save_canny_image(control_image: Image.Image, path: str):
        from mflux import ImageUtil

        base, ext = os.path.splitext(path)
        new_filename = f"{base}_controlnet_canny{ext}"
        ImageUtil.save_image(control_image, new_filename)
