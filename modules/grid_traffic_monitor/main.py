# main.py
import geopandas as gpd
import pandas as pd
import numpy as np
from grid_traffic_monitor import GridTrafficMonitor
from traffic_congestion_pred import TrafficCongestionPredictor
import time
from datetime import datetime
import h3
from IPython.display import display, clear_output
import warnings
import sys
warnings.filterwarnings('ignore')

def main():
    # Initialize GridTrafficMonitor with more conservative parameters
    monitor = GridTrafficMonitor(
        base_resolution=8,  # Reduced from 9 to handle larger areas
        min_resolution=7,
        max_resolution=10,  # Reduced from 11 to prevent over-fragmentation
        min_traffic_density=100.0,
        max_merge_threshold=20.0,
        smoothing_factor=0.3
    )
    
    # Load city boundary
    try:
        print("Loading geographic data...")
        city_gdf = monitor.load_city_boundary('/home/raw/Desktop/Coding/Jhakaas_Rasta/geopkg/Ahmedabad.gpkg')
        boundary_gdf = gpd.read_file('/home/raw/Desktop/Coding/Jhakaas_Rasta/geopkg/clipping_boundary.geojson')
        
        # Ensure both files are in the same CRS
        if city_gdf.crs != boundary_gdf.crs:
            boundary_gdf = boundary_gdf.to_crs(city_gdf.crs)
            
        # Clip city to boundary
        city_gdf = gpd.clip(city_gdf, boundary_gdf)
        print("Geographic data loaded successfully")
    except Exception as e:
        print(f"Error loading geographic data: {str(e)}")
        return
    
    # Initialize grid system
    try:
        print("Initializing grid system...")
        hex_polygons = monitor.initialize_grid(city_gdf)
        print(f"Grid system initialized with {len(hex_polygons)} hexagons")
    except Exception as e:
        print(f"Error initializing grid system: {str(e)}")
        return
    
    # Initialize traffic predictor
    try:
        print("Initializing traffic predictor...")
        predictor = TrafficCongestionPredictor(
            num_nodes=len(hex_polygons),
            input_dim=3,  # traffic_density, time_of_day, day_of_week
            hidden_dims=[64, 32, 16],
            output_dim=1,
            num_timesteps=12,
            batch_size=32
        )
        print("Traffic predictor initialized successfully")
    except Exception as e:
        print(f"Error initializing traffic predictor: {str(e)}")
        return
    
    # Simulate real-time updates
    try:
        while True:
            current_time = datetime.now()
            
            # Simulate traffic updates (replace with real data in production)
            for hex_id in monitor.current_grids.keys():
                # Simulate traffic density based on time of day
                hour = current_time.hour
                base_density = 50 + 50 * np.sin(np.pi * hour / 12)  # Higher during day
                noise = np.random.normal(0, 10)
                new_density = max(0, base_density + noise)
                
                monitor.update_traffic_density(hex_id, new_density)
            
            # Adjust grid resolution
            resolution_changes = monitor.adjust_grid_resolution()
            
            # Prepare data for prediction
            traffic_data = []
            for hex_id, density in monitor.current_grids.items():
                traffic_data.append({
                    'hex_id': hex_id,
                    'traffic_density': density,
                    'time_of_day': current_time.hour + current_time.minute / 60,
                    'day_of_week': current_time.weekday()
                })
            
            traffic_df = pd.DataFrame(traffic_data)
            
            # Prepare features for prediction
            features, targets, adj_matrix = predictor.prepare_data(
                traffic_df,
                list(monitor.current_grids.keys())
            )
            
            # Generate predictions
            predictions = predictor.predict(
                features[-predictor.num_timesteps:].reshape(1, predictor.num_timesteps, -1, 3),
                adj_matrix
            )
            
            # Visualize current state
            center_lat = city_gdf.geometry.centroid.y.iloc[0]
            center_lon = city_gdf.geometry.centroid.x.iloc[0]
            m = monitor.visualize_grid(center_lat, center_lon)
            
            # Display statistics
            stats = monitor.get_grid_stats()
            print("\nGrid Statistics:")
            for key, value in stats.items():
                print(f"{key}: {value:.2f}")
            
            # Display predictions
            print("\nPredicted traffic densities (next timestep):")
            for hex_id, pred in zip(monitor.current_grids.keys(), predictions[0]):
                print(f"{hex_id}: {pred[0]:.2f}")
            
            # Update visualization
            display(m)
            
            # Wait before next update
            time.sleep(300)  # 5 minutes
            clear_output(wait=True)
            
    except KeyboardInterrupt:
        print("\nMonitoring stopped by user")
    except Exception as e:
        print(f"Error during monitoring: {str(e)}")

if __name__ == "__main__":
    main()