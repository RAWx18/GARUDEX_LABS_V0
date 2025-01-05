# File: main.py
"""Main training loop and system coordination"""

import torch
from torch.optim import Adam
import logging
from typing import Dict, List
from config import DataConfig, SimulationConfig, ModelConfig
from data_processor import VideoProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class IndianDrivingSystem:
    """Main system coordinator"""
    
    def __init__(self, configs: Dict):
        self.configs = configs
        self._setup_components()
        
    def _setup_components(self) -> None:
        """Initialize all system components"""
        self.data_processor = VideoProcessor(self.configs["data"])
        self.perception = PerceptionSystem(self.configs["model"])
        self.policy_net = PolicyNetwork(self.configs["model"])
        self.value_net = ValueNetwork(self.configs["model"])
        self.buffer = PrioritizedReplayBuffer(1000000)
        self.env = SUMOEnvironment(self.configs["simulation"])
        self.quantum_optimizer = QuantumRouteOptimizer(20, 5)
        
    def train(self, num_episodes: int) -> List[float]:
        """Main training loop"""
        rewards = []
        optimizer = Adam(
            list(self.policy_net.parameters()) + 
            list(self.value_net.parameters()),
            lr=self.configs["model"].learning_rate
        )
        
        for episode in range(num_episodes):
            episode_reward = self._train_episode(optimizer)
            rewards.append(episode_reward)
            logger.info(f"Episode {episode}: Reward = {episode_reward}")
        
        return rewards
    
    def _train_episode(self, optimizer: Adam) -> float:
        """Train for one episode"""
        state = self.env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            # Process state through perception system
            features = self.perception(torch.tensor(state).unsqueeze(0))
            
            # Get action from policy network
            action, _ = self.policy_net(features)
            
            # Execute action in environment
            next_state, reward, done, _ = self.env.step(action.detach().numpy())
            
            # Store experience
            self.buffer.add((state, action, reward, next_state, done))
            
            # Update networks
            if len(self.buffer) >= self.configs["model"].batch_size:
                self._update_networks(optimizer)
            
            state = next_state
            episode_reward += reward
        
        return episode_reward

if __name__ == "__main__":
    configs = {
        "data": DataConfig(),
        "model": ModelConfig(),
        "simulation": SimulationConfig()
    }
    
    system = IndianDrivingSystem(configs)
    rewards = system.train(num_episodes=1000)