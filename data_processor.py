# File: data_processor.py
"""Data preprocessing and augmentation pipeline"""

import cv2
import numpy as np
from typing import Generator, List, Tuple
import albumentations as A
from config import DataConfig

class VideoProcessor:
    """Handles video data preprocessing and augmentation"""
    
    def __init__(self, config: DataConfig):
        self.config = config
        self.augmentor = self._create_augmentation_pipeline()
    
    def _create_augmentation_pipeline(self) -> A.Compose:
        """Creates the augmentation pipeline using albumentations"""
        return A.Compose([
            A.RandomBrightnessContrast(
                brightness_limit=self.config.augmentation_params["brightness_range"][1],
                contrast_limit=self.config.augmentation_params["contrast_range"][1]
            ),
            A.MotionBlur(
                blur_limit=self.config.augmentation_params["motion_blur_kernel"][1]
            ),
            A.Rotate(
                limit=self.config.augmentation_params["rotation_range"][1]
            ),
            A.RandomResizedCrop(
                height=self.config.resolution[1],
                width=self.config.resolution[0],
                scale=self.config.augmentation_params["crop_range"]
            )
        ])
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Processes a single frame with augmentation"""
        frame = cv2.resize(frame, self.config.resolution)
        if self.config.color_space == "HSV":
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
        augmented = self.augmentor(image=frame)
        return augmented["image"]
