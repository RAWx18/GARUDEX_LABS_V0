# File: utils.py
"""Utility functions for the RL system"""

import numpy as np
import torch
from typing import Dict, List, Union
import logging

logger = logging.getLogger(__name__)

def setup_logging(log_file: str = "driving_rl.log") -> None:
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

def normalize_state(state: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Normalize state values using running statistics"""
    return (state - mean) / (std + 1e-8)

def create_visualization(
    state: Dict[str, np.ndarray],
    action: Dict[str, float],
    metrics: Dict[str, float]
) -> Dict[str, Union[np.ndarray, dict]]:
    """Create visualization data for debugging and monitoring"""
    return {
        "state_visualization": _create_state_plot(state),
        "action_visualization": _create_action_plot(action),
        "metrics_visualization": _create_metrics_plot(metrics)
    }

def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    path: str
) -> None:
    """Save model checkpoint"""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, path)
    logger.info(f"Checkpoint saved at epoch {epoch}")
