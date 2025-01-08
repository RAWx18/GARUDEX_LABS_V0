# File: config.py
"""Configuration settings for the Indian Driving Behavior RL System"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

def get_default_augmentation_params() -> Dict:
    return {
        "brightness_range": (-0.3, 0.3),
        "contrast_range": (-0.2, 0.2),
        "motion_blur_kernel": (3, 7),
        "rotation_range": (-5, 5),
        "crop_range": (0.8, 1.0)
    }

def get_default_feature_channels() -> List[int]:
    return [256, 512, 1024, 2048]

def get_default_vehicle_distribution() -> Dict[str, float]:
    return {
        "two_wheeler": 0.4,
        "car": 0.35,
        "bus_truck": 0.15,
        "auto_rickshaw": 0.1
    }

def get_default_resolution() -> Tuple[int, int]:
    return (1920, 1080)

def get_default_peak_flow_rate() -> Tuple[int, int]:
    return (2000, 3000)

def get_default_off_peak_flow_rate() -> Tuple[int, int]:
    return (800, 1200)

def get_default_night_flow_rate() -> Tuple[int, int]:
    return (300, 500)

@dataclass
class DataConfig:
    """Data collection and preprocessing configuration"""
    frame_rate: int = 30
    resolution: Tuple[int, int] = field(default_factory=get_default_resolution)
    color_space: str = "RGB"
    augmentation_params: Dict = field(default_factory=get_default_augmentation_params)

@dataclass
class ModelConfig:
    """Neural network architecture configuration"""
    cnn_backbone: str = "resnet152"
    feature_channels: List[int] = field(default_factory=get_default_feature_channels)
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
    vehicle_distribution: Dict[str, float] = field(default_factory=get_default_vehicle_distribution)
    peak_flow_rate: Tuple[int, int] = field(default_factory=get_default_peak_flow_rate)
    off_peak_flow_rate: Tuple[int, int] = field(default_factory=get_default_off_peak_flow_rate)
    night_flow_rate: Tuple[int, int] = field(default_factory=get_default_night_flow_rate)