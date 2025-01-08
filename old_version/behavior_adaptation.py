# File: behavior_adaptation.py
"""Behavior adaptation system for handling different conditions"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

@dataclass
class AdaptationConfig:
    """Configuration for behavior adaptation"""
    update_frequency: float = 300  # 5 minutes in seconds
    historical_window: float = 3600  # 1 hour in seconds
    prediction_horizon: float = 900  # 15 minutes in seconds
    confidence_threshold: float = 0.85
    weather_response_time: float = 30  # seconds
    safety_margin: float = 1.2  # 20% increase
    visibility_threshold: float = 100  # meters
    density_levels: Dict[str, Tuple[float, float]] = {
        "low": (0.0, 0.3),
        "medium": (0.3, 0.6),
        "high": (0.6, 0.8),
        "extreme": (0.8, 1.0)
    }

class BehaviorAdapter:
    """Handles adaptation of driving behavior based on conditions"""
    
    def __init__(self, config: AdaptationConfig):
        self.config = config
        self.current_state = {}
        self.history = []
        
    def update_weather_response(self, visibility: float, precipitation: float) -> Dict[str, float]:
        """Adjust behavior parameters based on weather conditions"""
        safety_multiplier = 1.0
        
        if visibility < self.config.visibility_threshold:
            safety_multiplier *= self.config.safety_margin
            
        return {
            "following_distance": self.config.safety_margin * safety_multiplier,
            "speed_limit_factor": 1.0 / safety_multiplier,
            "reaction_time": max(0.15, 0.15 * safety_multiplier)
        }
    
    def adapt_to_traffic_density(self, density: float) -> Dict[str, float]:
        """Adjust behavior based on traffic density"""
        current_level = "low"
        for level, (low, high) in self.config.density_levels.items():
            if low <= density <= high:
                current_level = level
                break
                
        adaptations = {
            "low": {"aggression": 0.5, "cooperation": 0.8},
            "medium": {"aggression": 0.7, "cooperation": 0.7},
            "high": {"aggression": 0.9, "cooperation": 0.6},
            "extreme": {"aggression": 1.0, "cooperation": 0.5}
        }
        
        return adaptations[current_level]