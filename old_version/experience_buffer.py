# File: experience_buffer.py
"""Prioritized experience replay buffer implementation"""

import numpy as np
from collections import deque
from typing import Dict, List, Tuple

class PrioritizedReplayBuffer:
    """Implements prioritized experience replay"""
    
    def __init__(self, capacity: int, alpha: float = 0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)
    
    def add(self, experience: Tuple) -> None:
        """Add experience to buffer with maximum priority"""
        max_priority = max(self.priorities) if self.priorities else 1.0
        self.buffer.append(experience)
        self.priorities.append(max_priority)
    
    def sample(self, batch_size: int, beta: float) -> Tuple[List, np.ndarray]:
        """Sample batch of experiences based on priorities"""
        priorities = np.array(self.priorities)
        probs = priorities ** self.alpha
        probs /= probs.sum()
        
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]
        
        weights = (len(self.buffer) * probs[indices]) ** (-beta)
        weights /= weights.max()
        
        return samples, indices, weights
