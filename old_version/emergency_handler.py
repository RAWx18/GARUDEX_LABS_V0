# File: emergency_handler.py
"""Emergency situation handling module"""

import numpy as np
from typing import Dict, List, Tuple

class EmergencyHandler:
    """Handles emergency situations and collision avoidance"""
    
    def __init__(self):
        self.reaction_time = 0.15  # seconds
        self.max_deceleration = 7.0  # m/s²
        self.steering_rate_limit = 500.0  # degrees/s
        self.min_safety_distance = 1.5  # meters
        self.ttc_threshold = 2.0  # seconds
        
    def calculate_ttc(self, relative_position: np.ndarray, relative_velocity: np.ndarray) -> float:
        """Calculate Time To Collision (TTC)"""
        distance = np.linalg.norm(relative_position)
        closing_speed = np.dot(relative_velocity, relative_position) / distance
        
        if closing_speed <= 0:
            return float('inf')
        return distance / closing_speed
    
    def generate_emergency_maneuver(self, 
                                  state: Dict[str, np.ndarray],
                                  obstacles: List[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
        """Generate emergency maneuver based on current situation"""
        min_ttc = float('inf')
        critical_obstacle = None
        
        # Find most critical obstacle
        for obstacle in obstacles:
            ttc = self.calculate_ttc(
                obstacle["position"] - state["position"],
                obstacle["velocity"] - state["velocity"]
            )
            if ttc < min_ttc:
                min_ttc = ttc
                critical_obstacle = obstacle
        
        # Generate avoidance maneuver if necessary
        if min_ttc < self.ttc_threshold:
            return self._compute_avoidance_maneuver(state, critical_obstacle, min_ttc)
        
        return {"acceleration": 0.0, "steering": 0.0}
    
    def _compute_avoidance_maneuver(self,
                                   state: Dict[str, np.ndarray],
                                   obstacle: Dict[str, np.ndarray],
                                   ttc: float) -> Dict[str, np.ndarray]:
        """Compute specific avoidance maneuver parameters"""
        relative_pos = obstacle["position"] - state["position"]
        distance = np.linalg.norm(relative_pos)
        
        # Determine if braking is sufficient
        required_deceleration = (state["speed"]**2) / (2 * distance)
        
        if required_deceleration <= self.max_deceleration:
            # Braking is sufficient
            return {
                "acceleration": -required_deceleration,
                "steering": 0.0
            }
        else:
            # Need combined braking and steering
            steering_angle = np.clip(
                np.arctan2(relative_pos[1], relative_pos[0]),
                -self.steering_rate_limit * self.reaction_time,
                self.steering_rate_limit * self.reaction_time
            )
            return {
                "acceleration": -self.max_deceleration,
                "steering": steering_angle
            }
