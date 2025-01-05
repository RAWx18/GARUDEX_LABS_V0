# File: config.py
"""Configuration settings for the Indian Driving Behavior RL System"""

from dataclasses import dataclass
from typing import Dict, List, Tuple

@dataclass
class DataConfig:
    """Data collection and preprocessing configuration"""
    frame_rate: int = 30
    resolution: Tuple[int, int] = (1920, 1080)
    color_space: str = "RGB"
    augmentation_params: Dict = {
        "brightness_range": (-0.3, 0.3),
        "contrast_range": (-0.2, 0.2),
        "motion_blur_kernel": (3, 7),
        "rotation_range": (-5, 5),
        "crop_range": (0.8, 1.0)
    }

@dataclass
class ModelConfig:
    """Neural network architecture configuration"""
    cnn_backbone: str = "resnet152"
    feature_channels: List[int] = (256, 512, 1024, 2048)
    lstm_units: int = 512
    lstm_layers: int = 3
    dropout_rate: float = 0.3
    batch_size: int = 256
    learning_rate: float = 3e-4

@dataclass
class SimulationConfig:
    """SUMO simulation environment configuration"""
    lane_width: float = 3.2
    min_lanes: int = 1
    max_lanes: int = 6
    vehicle_distribution: Dict[str, float] = {
        "two_wheeler": 0.4,
        "car": 0.35,
        "bus_truck": 0.15,
        "auto_rickshaw": 0.1
    }
    peak_flow_rate: Tuple[int, int] = (2000, 3000)
    off_peak_flow_rate: Tuple[int, int] = (800, 1200)
    night_flow_rate: Tuple[int, int] = (300, 500)