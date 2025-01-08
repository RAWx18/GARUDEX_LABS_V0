# File: metrics.py
"""Performance metrics and evaluation"""

import numpy as np
from typing import Dict, List
from collections import deque

class PerformanceMetrics:
    """Tracks and calculates various performance metrics"""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.metrics = {
            "safety_violations": deque(maxlen=window_size),
            "avg_speed": deque(maxlen=window_size),
            "comfort_score": deque(maxlen=window_size),
            "cultural_compliance": deque(maxlen=window_size),
            "fuel_efficiency": deque(maxlen=window_size)
        }
        
    def update(self, 
               safety_score: float,
               speed: float,
               acceleration: float,
               cultural_score: float,
               fuel_consumption: float) -> None:
        """Update metrics with new values"""
        self.metrics["safety_violations"].append(1.0 - safety_score)
        self.metrics["avg_speed"].append(speed)
        self.metrics["comfort_score"].append(self._calculate_comfort(acceleration))
        self.metrics["cultural_compliance"].append(cultural_score)
        self.metrics["fuel_efficiency"].append(1.0 / max(fuel_consumption, 1e-6))
    
    def get_summary(self) -> Dict[str, float]:
        """Get summary statistics of all metrics"""
        return {
            "safety_score": 1.0 - np.mean(self.metrics["safety_violations"]),
            "avg_speed": np.mean(self.metrics["avg_speed"]),
            "comfort_score": np.mean(self.metrics["comfort_score"]),
            "cultural_compliance": np.mean(self.metrics["cultural_compliance"]),
            "efficiency_score": np.mean(self.metrics["fuel_efficiency"])
        }
    
    def _calculate_comfort(self, acceleration: float) -> float:
        """Calculate comfort score based on acceleration"""
        # Comfort decreases with higher acceleration
        return np.exp(-np.abs(acceleration) / 2.0)
