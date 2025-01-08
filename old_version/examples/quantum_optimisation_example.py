"""Example usage of quantum route optimizer in Indian driving scenario."""

import numpy as np
from quantum_optimizer import TrafficAwareQuantumOptimizer

def optimize_driving_route(
    locations: List[Tuple[float, float]],
    traffic_conditions: Dict[str, np.ndarray]
) -> List[int]:
    """
    Optimize driving route considering Indian traffic conditions.
    
    Args:
        locations: List of (latitude, longitude) coordinates
        traffic_conditions: Current traffic data
    """
    # Initialize optimizer
    optimizer = TrafficAwareQuantumOptimizer(
        num_qubits=20,  # Adjust based on problem size
        depth=5         # Increase for better solutions
    )
    
    # Create base cost matrix (using distances)
    n_locations = len(locations)
    base_costs = np.zeros((n_locations, n_locations))
    
    for i in range(n_locations):
        for j in range(i+1, n_locations):
            # Calculate Haversine distance between locations
            distance = calculate_distance(locations[i], locations[j])
            base_costs[i,j] = distance
            base_costs[j,i] = distance
    
    # Add traffic considerations
    traffic_costs = optimizer.calculate_traffic_cost_matrix(
        base_costs,
        traffic_conditions
    )
    
    # Get optimal route
    route, cost = optimizer.optimize_route(traffic_costs)
    
    return route, cost

# Example usage
if __name__ == "__main__":
    # Sample locations (lat, lon)
    locations = [
        (12.9716, 77.5946),  # Bangalore
        (13.0827, 80.2707),  # Chennai
        (17.3850, 78.4867),  # Hyderabad
        (19.0760, 72.8777)   # Mumbai
    ]
    
    # Sample traffic conditions
    traffic_conditions = {
        'congestion': np.array([
            [0.0, 0.5, 0.3, 0.8],
            [0.5, 0.0, 0.4, 0.6],
            [0.3, 0.4, 0.0, 0.7],
            [0.8, 0.6, 0.7, 0.0]
        ]),
        'time_factor': np.array([
            [0.0, 0.3, 0.2, 0.4],
            [0.3, 0.0, 0.3, 0.5],
            [0.2, 0.3, 0.0, 0.6],
            [0.4, 0.5, 0.6, 0.0]
        ]),
        'weather': np.array([
            [0.0, 0.2, 0.1, 0.3],
            [0.2, 0.0, 0.2, 0.4],
            [0.1, 0.2, 0.0, 0.2],
            [0.3, 0.4, 0.2, 0.0]
        ]),
        'festival': np.array([
            [0.0, 0.4, 0.2, 0.6],
            [0.4, 0.0, 0.3, 0.5],
            [0.2, 0.3, 0.0, 0.4],
            [0.6, 0.5, 0.4, 0.0]
        ])
    }
    
    # Calculate optimal route
    optimal_route, total_cost = optimize_driving_route(
        locations,
        traffic_conditions
    )
    
    print(f"Optimal Route: {optimal_route}")
    print(f"Total Cost: {total_cost:.2f}")