# File: value_network.py
"""Value network implementation"""

import torch
import torch.nn as nn

class ValueNetwork(nn.Module):
    """Dual Q-network for value estimation"""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.q_network = nn.Sequential(
            nn.Linear(config.feature_channels[-1], 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass through value network"""
        return self.q_network(state)