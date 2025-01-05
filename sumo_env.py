# File: sumo_env.py
"""SUMO simulation environment wrapper"""

import traci
import numpy as np
from typing import Dict, Tuple

class SUMOEnvironment:
    """Wrapper for SUMO traffic simulation"""
    
    def __init__(self, config: SimulationConfig):
        self.config = config
        self._setup_simulation()
    
    def _setup_simulation(self) -> None:
        """Initialize SUMO simulation with custom parameters"""
        traci.start(["sumo-gui", "-c", "indian_traffic.sumocfg"])
        self._setup_vehicle_types()
        self._setup_traffic_flow()
    
    def _setup_vehicle_types(self) -> None:
        """Configure vehicle types and their properties"""
        for vtype, prob in self.config.vehicle_distribution.items():
            traci.vehicletype.setParameter(vtype, "probability", str(prob))
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """Execute one simulation step"""
        traci.simulationStep()
        state = self._get_state()
        reward = self._calculate_reward()
        done = self._check_termination()
        info = self._get_info()
        return state, reward, done, info
    
    def _calculate_reward(self) -> float:
        """Calculate reward based on safety and efficiency"""
        safety_score = self._evaluate_safety()
        efficiency_score = self._evaluate_efficiency()
        return 0.7 * safety_score + 0.3 * efficiency_score
