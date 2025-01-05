# File: policy_network.py
"""Policy network implementation"""

import torch
import torch.nn as nn
from typing import Tuple
from config import ModelConfig

class PolicyNetwork(nn.Module):
    """Policy network combining LSTM and attention mechanisms"""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=config.feature_channels[-1],
            hidden_size=config.lstm_units,
            num_layers=config.lstm_layers,
            dropout=config.dropout_rate,
            batch_first=True
        )
        self.attention = MultiHeadAttention(config.lstm_units, num_heads=8)
        self.policy_head = nn.Sequential(
            nn.Linear(config.lstm_units, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )
    
    def forward(self, x: torch.Tensor, hidden: Tuple[torch.Tensor, torch.Tensor] = None) \
            -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass through policy network"""
        lstm_out, hidden = self.lstm(x, hidden)
        attended = self.attention(lstm_out)
        action_logits = self.policy_head(attended)
        return action_logits, hidden