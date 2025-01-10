# grid_traffic_monitor.py
import geopandas as gpd
import h3
import numpy as np
import pandas as pd
from shapely.geometry import Polygon, mapping
import warnings
from typing import Dict, List, Tuple
from sklearn.cluster import KMeans
import folium
from branca.colormap import LinearColormap
import json

class GridTrafficMonitor:
    def __init__(
        self,
        base_resolution: int = 9,
        min_resolution: int = 7,
        max_resolution: int = 11,
        min_traffic_density: float = 100.0,
        max_merge_threshold: float = 20.0,
        smoothing_factor: float = 0.3
    ):
        """Initialize the grid traffic monitoring system."""
        self.base_resolution = base_resolution
        self.min_resolution = min_resolution
        self.max_resolution = max_resolution
        self.min_traffic_density = min_traffic_density
        self.max_merge_threshold = max_merge_threshold
        self.smoothing_factor = smoothing_factor
        self.current_grids = {}  # hex_id -> traffic_density
        self.grid_history = {}   # hex_id -> List[traffic_density]
        
    def load_city_boundary(self, boundary_path: str) -> gpd.GeoDataFrame:
        """Load and process city boundary."""
        try:
            if boundary_path.endswith('.gpkg'):
                gdf = gpd.read_file(boundary_path)
            elif boundary_path.endswith('.geojson'):
                gdf = gpd.read_file(boundary_path)
            else:
                raise ValueError("Unsupported file format")
            
            if gdf.crs is None:
                warnings.warn("CRS not found, assuming WGS84")
                gdf.set_crs(epsg=4326, inplace=True)
            elif gdf.crs != 'EPSG:4326':
                gdf = gdf.to_crs(epsg=4326)
                
            return gdf
        except Exception as e:
            raise RuntimeError(f"Error loading boundary file: {str(e)}")

    def initialize_grid(self, boundary_gdf: gpd.GeoDataFrame) -> Dict[str, Polygon]:
            """Initialize hexagonal grid over the city boundary."""
            try:
                # Get boundary coordinates
                boundary_geom = boundary_gdf.geometry.iloc[0]
                coords = list(boundary_geom.exterior.coords)
                
                # Create polygon array for h3 ([lat, lng] pairs)
                outer_ring = []
                for coord in coords:
                    outer_ring.append([coord[1], coord[0]])  # Convert to [lat, lng]
                
                # Get holes if they exist
                holes = []
                if hasattr(boundary_geom, 'interiors'):
                    for interior in boundary_geom.interiors:
                        hole = []
                        for coord in interior.coords:
                            hole.append([coord[1], coord[0]])  # Convert to [lat, lng]
                        holes.append(hole)
                
                # Create polygon structure
                # h3_polygon = {
                #     "outer_ring": outer_ring,
                #     "holes": holes
                # } # doesn't support dictionary type 
                h3_polygon = h3.LatLngPoly(outer_ring, holes)
                
                # Generate hexagons using polyfill is depreciated from h3.4x version so use polygon_to_cells
                hexagons = h3.polygon_to_cells(h3_polygon, self.base_resolution)
                
                # Convert to polygons
                hex_polygons = {}
                for hex_id in hexagons:
                    # Get hex boundary coordinates
                    hex_boundary = h3.cell_to_boundary(hex_id)
                    # Convert to [lon, lat] pairs for Shapely
                    hex_boundary = [[coord[1], coord[0]] for coord in hex_boundary]
                    hex_polygons[hex_id] = Polygon(hex_boundary)
                    self.current_grids[hex_id] = 0.0  # Initialize traffic density
                    
                if not hex_polygons:
                    raise ValueError("No hexagons generated for the given boundary")
                    
                return hex_polygons
            except Exception as e:
                raise RuntimeError(f"Error initializing grid: {str(e)}")

    def update_traffic_density(self, hex_id: str, new_density: float):
        """Update traffic density for a specific hexagon."""
        if hex_id in self.current_grids:
            # Apply exponential smoothing
            current_density = self.current_grids[hex_id]
            smoothed_density = (self.smoothing_factor * new_density + 
                              (1 - self.smoothing_factor) * current_density)
            self.current_grids[hex_id] = smoothed_density
            
            # Update history
            if hex_id not in self.grid_history:
                self.grid_history[hex_id] = []
            self.grid_history[hex_id].append(smoothed_density)
            
            # Keep only last 24 hours (assuming 5-minute updates)
            max_history = 288  # 24 hours * 12 updates per hour
            if len(self.grid_history[hex_id]) > max_history:
                self.grid_history[hex_id] = self.grid_history[hex_id][-max_history:]

    def adjust_grid_resolution(self) -> Dict[str, int]:
        """Dynamically adjust grid resolution based on traffic density."""
        resolution_changes = {}
        
        for hex_id, density in list(self.current_grids.items()):  # Create list to avoid runtime modification
            current_res = h3.get_resolution(hex_id)  # Updated function name
            
            if density > self.min_traffic_density and current_res < self.max_resolution:
                # Refine grid
                child_hexagons = h3.cell_to_children(hex_id)  # Updated function name
                for child in child_hexagons:
                    self.current_grids[child] = density / len(child_hexagons)
                del self.current_grids[hex_id]
                resolution_changes[hex_id] = current_res + 1
                
            elif density < self.max_merge_threshold and current_res > self.min_resolution:
                # Merge grid
                parent = h3.cell_to_parent(hex_id)  # Updated function name
                if parent not in self.current_grids:
                    siblings = h3.cell_to_children(parent)  # Updated function name
                    sibling_densities = [self.current_grids.get(s, 0) for s in siblings]
                    if all(d < self.max_merge_threshold for d in sibling_densities):
                        self.current_grids[parent] = sum(sibling_densities)
                        for sibling in siblings:
                            if sibling in self.current_grids:
                                del self.current_grids[sibling]
                        resolution_changes[hex_id] = current_res - 1
                        
        return resolution_changes

    def visualize_grid(self, center_lat: float, center_lon: float, zoom: int = 11) -> folium.Map:
        """Create an interactive visualization of the grid system."""
        m = folium.Map(location=[center_lat, center_lon], zoom_start=zoom)
        
        # Create color map based on traffic density
        densities = list(self.current_grids.values())
        if not densities:
            return m
            
        colormap = LinearColormap(
            colors=['green', 'yellow', 'red'],
            vmin=min(densities),
            vmax=max(densities)
        )
        
        # Add hexagons to map
        for hex_id, density in self.current_grids.items():
            # Get hex boundary coordinates
            boundary = h3.cell_to_boundary(hex_id)
            # Convert to GeoJSON-compatible format
            geojson_coords = [[coord[1], coord[0]] for coord in boundary]
            # Close the polygon by repeating the first point
            geojson_coords.append(geojson_coords[0])
            
            folium.GeoJson(
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [geojson_coords]
                    }
                },
                style_function=lambda x, density=density: {
                    'fillColor': colormap(density),
                    'color': 'black',
                    'weight': 1,
                    'fillOpacity': 0.6
                },
                tooltip=f"Hex ID: {hex_id}<br>Density: {density:.2f}"
            ).add_to(m)
            
        # Add colormap to map
        colormap.add_to(m)
        return m

    def get_grid_stats(self) -> Dict[str, float]:
        """Calculate statistics for the current grid system."""
        if not self.current_grids:
            return {
                'total_hexagons': 0,
                'avg_density': 0.0,
                'max_density': 0.0,
                'min_density': 0.0,
                'std_density': 0.0
            }
            
        values = list(self.current_grids.values())
        return {
            'total_hexagons': len(self.current_grids),
            'avg_density': np.mean(values),
            'max_density': np.max(values),
            'min_density': np.min(values),
            'std_density': np.std(values)
        }