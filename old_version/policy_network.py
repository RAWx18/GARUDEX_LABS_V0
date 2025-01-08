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

class MultiHeadAttention(nn.Module):
    """
    Multi-head attention mechanism for processing sequential data
    Allows the model to jointly attend to information from different representation subspaces
    """
    
    def __init__(self, embed_dim: int, num_heads: int = 8, dropout: float = 0.1):
        """
        Initialize multi-head attention module
        
        Args:
            embed_dim: Dimension of input embeddings
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()
        
        assert embed_dim % num_heads == 0, "Embedding dimension must be divisible by number of heads"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        # Linear layers for query, key, value projections
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        
        # Output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5  # Scaling factor for dot product
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for multi-head attention
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, embed_dim]
            
        Returns:
            Output tensor of shape [batch_size, seq_len, embed_dim]
        """
        batch_size, seq_len, _ = x.shape
        
        # Project inputs to queries, keys, and values
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute scaled dot-product attention
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn_weights = torch.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention weights to values
        out = torch.matmul(attn_weights, v)
        
        # Reshape and project output
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        out = self.out_proj(out)
        
        return out