# traffic_congestion_pred.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import List, Tuple, Dict
import pandas as pd
from sklearn.preprocessing import StandardScaler
import h3
import warnings

def get_device() -> str:
    """Safely determine the available device."""
    if torch.cuda.is_available():
        try:
            # Test CUDA device
            torch.cuda.current_device()
            return 'cuda'
        except Exception as e:
            warnings.warn(f"CUDA initialization error: {str(e)}. Using CPU instead.")
            return 'cpu'
    return 'cpu'

class SpatialTemporalGraph(nn.Module):
    """Spatial-Temporal Graph Neural Network for traffic prediction."""
    def __init__(
        self,
        num_nodes: int,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        num_timesteps: int
    ):
        super(SpatialTemporalGraph, self).__init__()
        
        self.num_nodes = num_nodes
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.num_timesteps = num_timesteps
        
        # Graph Convolution layers
        self.graph_convs = nn.ModuleList()
        prev_dim = input_dim
        for dim in hidden_dims:
            self.graph_convs.append(nn.Linear(prev_dim, dim))
            prev_dim = dim
            
        # LSTM layer for temporal dependencies
        self.lstm = nn.LSTM(
            input_size=hidden_dims[-1],
            hidden_size=hidden_dims[-1],
            num_layers=2,
            batch_first=True
        )
        
        # Output layer
        self.output_layer = nn.Linear(hidden_dims[-1], output_dim)
        
    def forward(self, x: torch.Tensor, adj_matrix: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, num_timesteps, num_nodes, input_dim)
            adj_matrix: Adjacency matrix of shape (num_nodes, num_nodes)
            
        Returns:
            Predictions tensor of shape (batch_size, num_nodes, output_dim)
        """
        batch_size = x.size(0)
        
        # Reshape for graph convolution
        x = x.view(-1, self.num_nodes, self.input_dim)
        
        # Apply graph convolution layers
        for conv in self.graph_convs:
            x = torch.matmul(adj_matrix, x)  # Graph propagation
            x = conv(x)  # Feature transformation
            x = F.relu(x)
            
        # Reshape for LSTM
        x = x.view(batch_size, self.num_timesteps, self.num_nodes, -1)
        x = x.permute(0, 2, 1, 3)  # (batch_size, num_nodes, num_timesteps, hidden_dim)
        x = x.reshape(batch_size * self.num_nodes, self.num_timesteps, -1)
        
        # Apply LSTM
        x, _ = self.lstm(x)
        x = x[:, -1, :]  # Take last timestep
        
        # Output layer
        x = self.output_layer(x)
        x = x.view(batch_size, self.num_nodes, -1)
        
        return x

class TrafficDataset(Dataset):
    """Dataset for traffic prediction."""
    def __init__(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        num_timesteps: int
    ):
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets)
        self.num_timesteps = num_timesteps
        
    def __len__(self) -> int:
        return len(self.features) - self.num_timesteps
        
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return (
            self.features[idx:idx + self.num_timesteps],
            self.targets[idx + self.num_timesteps]
        )

class TrafficCongestionPredictor:
    def __init__(
        self,
        num_nodes: int,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        num_timesteps: int,
        learning_rate: float = 0.001,
        batch_size: int = 32,
        device: str = None
    ):
        """Initialize the traffic congestion predictor."""
        # Set device
        self.device = device if device is not None else get_device()
        print(f"Using device: {self.device}")
        
        self.batch_size = batch_size
        self.num_timesteps = num_timesteps
        self.scaler = StandardScaler()
        
        # Initialize model
        self.model = SpatialTemporalGraph(
            num_nodes=num_nodes,
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim,
            num_timesteps=num_timesteps
        ).to(self.device)
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()

    def prepare_data(
        self,
        traffic_data: pd.DataFrame,
        hex_ids: List[str]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Prepare data for training/prediction."""
        # Debug print
        print(f"Input traffic_data shape: {traffic_data.shape}")
        print(f"Number of hex_ids: {len(hex_ids)}")
        
        if traffic_data.empty:
            raise ValueError("Empty traffic data provided")
        
        # Create feature matrix
        features = []
        valid_hex_ids = []
        
        for hex_id in hex_ids:
            hex_data = traffic_data[traffic_data['hex_id'] == hex_id]
            if not hex_data.empty:
                feat_array = hex_data[['traffic_density', 'time_of_day', 'day_of_week']].values
                features.append(feat_array)
                valid_hex_ids.append(hex_id)
        
        if not features:
            raise ValueError("No valid data found for any hex_id")
        
        # Convert to numpy array and check shape
        features = np.array(features)
        print(f"Features shape after initial processing: {features.shape}")
        
        # Transpose to (timesteps, nodes, features)
        features = features.transpose(1, 0, 2)
        print(f"Features shape after transpose: {features.shape}")
        
        # Scale features
        original_shape = features.shape
        features_2d = features.reshape(-1, features.shape[-1])
        
        if features_2d.shape[0] == 0:
            raise ValueError("Empty feature array after reshaping")
            
        print(f"Features shape before scaling: {features_2d.shape}")
        
        try:
            features_scaled = self.scaler.fit_transform(features_2d)
            features = features_scaled.reshape(original_shape)
        except Exception as e:
            print(f"Scaling error: {str(e)}")
            raise
        
        # Create adjacency matrix based on H3 neighbors
        num_valid_hexes = len(valid_hex_ids)
        adj_matrix = np.zeros((num_valid_hexes, num_valid_hexes))
        
        for i, hex1 in enumerate(valid_hex_ids):
            for j, hex2 in enumerate(valid_hex_ids):
                if i != j and h3.grid_distance(hex1, hex2) == 1:
                    adj_matrix[i, j] = 1
                    
        # Normalize adjacency matrix
        row_sums = np.sum(adj_matrix, axis=1, keepdims=True)
        adj_matrix = adj_matrix / (row_sums + 1e-6)
        
        print(f"Final shapes - Features: {features.shape}, Adj Matrix: {adj_matrix.shape}")
        
        return features, features[:, :, 0], adj_matrix  # features, targets, adj_matrix        


    def train(
        self,
        train_features: np.ndarray,
        train_targets: np.ndarray,
        adj_matrix: np.ndarray,
        num_epochs: int,
        validation_split: float = 0.2
    ) -> Dict[str, List[float]]:
        """Train the model."""
        # Split data
        split_idx = int(len(train_features) * (1 - validation_split))
        train_dataset = TrafficDataset(
            train_features[:split_idx],
            train_targets[:split_idx],
            self.num_timesteps
        )
        val_dataset = TrafficDataset(
            train_features[split_idx:],
            train_targets[split_idx:],
            self.num_timesteps
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True if self.device == 'cuda' else False
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True if self.device == 'cuda' else False
        )
        
        # Convert adjacency matrix to tensor
        adj_matrix = torch.FloatTensor(adj_matrix).to(self.device)
        
        # Training history
        history = {
            'train_loss': [],
            'val_loss': [],
            'train_mae': [],
            'val_mae': []
        }
        
        # Training loop
        for epoch in range(num_epochs):
            # Training phase
            self.model.train()
            train_loss = 0
            train_mae = 0
            num_batches = 0
            
            for batch_features, batch_targets in train_loader:
                batch_features = batch_features.to(self.device)
                batch_targets = batch_targets.to(self.device)
                
                # Forward pass
                self.optimizer.zero_grad()
                predictions = self.model(batch_features, adj_matrix)
                loss = self.criterion(predictions, batch_targets)
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                # Metrics
                train_loss += loss.item()
                train_mae += torch.mean(torch.abs(predictions - batch_targets)).item()
                num_batches += 1
            
            train_loss /= num_batches
            train_mae /= num_batches
            
            # Validation phase
            self.model.eval()
            val_loss = 0
            val_mae = 0
            num_batches = 0
            
            with torch.no_grad():
                for batch_features, batch_targets in val_loader:
                    batch_features = batch_features.to(self.device)
                    batch_targets = batch_targets.to(self.device)
                    
                    predictions = self.model(batch_features, adj_matrix)
                    loss = self.criterion(predictions, batch_targets)
                    
                    val_loss += loss.item()
                    val_mae += torch.mean(torch.abs(predictions - batch_targets)).item()
                    num_batches += 1
                    
            val_loss /= num_batches
            val_mae /= num_batches
            
            # Update history
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['train_mae'].append(train_mae)
            history['val_mae'].append(val_mae)
            
            print(f"Epoch {epoch+1}/{num_epochs}")
            print(f"Train Loss: {train_loss:.4f}, Train MAE: {train_mae:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val MAE: {val_mae:.4f}")
            
        return history
    
    def predict(
        self,
        features: np.ndarray,
        adj_matrix: np.ndarray
    ) -> np.ndarray:
        """Generate predictions for the given features."""
        self.model.eval()
        
        # Prepare data
        features = torch.FloatTensor(features).to(self.device)
        adj_matrix = torch.FloatTensor(adj_matrix).to(self.device)
        
        with torch.no_grad():
            predictions = self.model(features, adj_matrix)
            
        return predictions.cpu().numpy()
    
    def save_model(self, path: str):
        """Save model state."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scaler': self.scaler
        }, path)
        
    def load_model(self, path: str):
        """Load model state."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scaler = checkpoint['scaler']