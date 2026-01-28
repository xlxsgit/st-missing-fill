"""
Missing Pattern Generation Module

This module implements three types of missing data patterns:
1. MCAR (Missing Completely At Random)
2. SEQ (Sequential Missing with configurable length)
3. SPATIAL (Spatially Correlated Sequential Missing)
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict
from scipy.spatial.distance import pdist, squareform


class MissingPatternGenerator:
    """Generate different types of missing patterns for time series data."""
    
    def __init__(self, data_shape: Tuple[int, int], random_state: Optional[int] = None):
        """
        Initialize the missing pattern generator.
        
        Args:
            data_shape: Shape of the data (n_timesteps, n_stations)
            random_state: Random seed for reproducibility
        """
        self.n_timesteps, self.n_stations = data_shape
        self.random_state = random_state
        self.rng = np.random.RandomState(random_state)
    
    def generate_mcar(self, missing_rate: float) -> np.ndarray:
        """
        Generate Missing Completely At Random (MCAR) pattern.
        
        Args:
            missing_rate: Overall missing rate (0 to 1)
            
        Returns:
            Boolean mask where True indicates missing values
        """
        if not 0 <= missing_rate <= 1:
            raise ValueError("missing_rate must be between 0 and 1")
        
        mask = self.rng.random((self.n_timesteps, self.n_stations)) < missing_rate
        return mask
    
    def generate_seq(self, missing_rate: float, max_length: int = 100) -> np.ndarray:
        """
        Generate Sequential missing pattern.
        
        Single station sequences have continuous missing values with length
        randomly sampled from 1 to max_length.
        
        Args:
            missing_rate: Overall target missing rate (0 to 1)
            max_length: Maximum length of continuous missing sequence
            
        Returns:
            Boolean mask where True indicates missing values
        """
        if not 0 <= missing_rate <= 1:
            raise ValueError("missing_rate must be between 0 and 1")
        if max_length < 1:
            raise ValueError("max_length must be at least 1")
        
        mask = np.zeros((self.n_timesteps, self.n_stations), dtype=bool)
        target_missing_count = int(missing_rate * self.n_timesteps * self.n_stations)
        current_missing_count = 0
        
        # Keep adding sequences until we reach the target missing rate
        # Cap max_attempts to prevent performance issues with large datasets
        max_attempts = min(target_missing_count * 10, 100000)
        attempt = 0
        
        while current_missing_count < target_missing_count and attempt < max_attempts:
            attempt += 1
            
            # Randomly select a station
            station_idx = self.rng.randint(0, self.n_stations)
            
            # Randomly select sequence length from 1 to max_length
            seq_length = self.rng.randint(1, max_length + 1)
            
            # Truncate sequence length if it would exceed target
            remaining = target_missing_count - current_missing_count
            seq_length = min(seq_length, remaining)
            
            # Randomly select start position
            if self.n_timesteps > seq_length:
                start_idx = self.rng.randint(0, self.n_timesteps - seq_length)
            else:
                start_idx = 0
                seq_length = min(seq_length, self.n_timesteps)
            
            # Apply the missing sequence
            end_idx = start_idx + seq_length
            mask[start_idx:end_idx, station_idx] = True
            
            current_missing_count = mask.sum()
        
        return mask
    
    def generate_spatial(
        self, 
        missing_rate: float,
        station_coords: np.ndarray,
        distance_threshold: float,
        max_length: int = 100,
        n_neighbors_range: Tuple[int, int] = (1, 5)
    ) -> np.ndarray:
        """
        Generate Spatially correlated sequential missing pattern.
        
        Missing pattern is related to station geographic locations. When a station
        has missing data, neighboring stations within distance_threshold also have
        sequential missing data together.
        
        Args:
            missing_rate: Overall target missing rate (0 to 1)
            station_coords: Array of shape (n_stations, 2) with (lat, lon) coordinates
            distance_threshold: Maximum distance for neighbors to be affected
            max_length: Maximum length of continuous missing sequence
            n_neighbors_range: Range (min, max) for number of neighbors to affect
            
        Returns:
            Boolean mask where True indicates missing values
        """
        if not 0 <= missing_rate <= 1:
            raise ValueError("missing_rate must be between 0 and 1")
        if max_length < 1:
            raise ValueError("max_length must be at least 1")
        if station_coords.shape[0] != self.n_stations:
            raise ValueError("station_coords must have same number of stations as data")
        
        # Calculate pairwise distances between stations
        distances = squareform(pdist(station_coords, metric='euclidean'))
        
        mask = np.zeros((self.n_timesteps, self.n_stations), dtype=bool)
        target_missing_count = int(missing_rate * self.n_timesteps * self.n_stations)
        current_missing_count = 0
        
        # Cap max_attempts to prevent performance issues with large datasets
        max_attempts = min(target_missing_count * 10, 100000)
        attempt = 0
        no_progress_count = 0  # Track attempts without progress
        last_missing_count = 0
        
        while current_missing_count < target_missing_count and attempt < max_attempts:
            attempt += 1
            
            # Check for progress stagnation
            if current_missing_count == last_missing_count:
                no_progress_count += 1
                if no_progress_count > 100:  # No progress for 100 consecutive attempts
                    break
            else:
                no_progress_count = 0
                last_missing_count = current_missing_count
            
            # Randomly select a seed station
            seed_station = self.rng.randint(0, self.n_stations)
            
            # Find neighbors within distance threshold
            neighbor_mask = distances[seed_station] <= distance_threshold
            neighbor_mask[seed_station] = True  # Include seed station itself
            neighbor_indices = np.where(neighbor_mask)[0]
            
            # Randomly select how many neighbors to affect
            min_neighbors, max_neighbors = n_neighbors_range
            n_neighbors_to_affect = self.rng.randint(
                min(min_neighbors, len(neighbor_indices)),
                min(max_neighbors, len(neighbor_indices)) + 1
            )
            
            # Randomly select which neighbors to affect
            affected_stations = self.rng.choice(
                neighbor_indices, 
                size=n_neighbors_to_affect,
                replace=False
            )
            
            # Randomly select sequence length from 1 to max_length
            seq_length = self.rng.randint(1, max_length + 1)
            
            # Limit sequence length to avoid overshooting target
            remaining = target_missing_count - current_missing_count
            # Approximate remaining per affected station
            max_seq_for_target = remaining // max(1, n_neighbors_to_affect)
            seq_length = min(seq_length, max_seq_for_target, self.n_timesteps)
            
            if seq_length < 1:
                break  # Not enough room for more sequences
            
            # Randomly select start position
            if self.n_timesteps > seq_length:
                start_idx = self.rng.randint(0, self.n_timesteps - seq_length)
            else:
                start_idx = 0
                seq_length = min(seq_length, self.n_timesteps)
            
            # Apply the missing sequence to all affected stations
            end_idx = start_idx + seq_length
            mask[start_idx:end_idx, affected_stations] = True
            
            current_missing_count = mask.sum()
        
        return mask


def calculate_station_distances(station_coords: np.ndarray) -> np.ndarray:
    """
    Calculate pairwise distances between stations.
    
    Args:
        station_coords: Array of shape (n_stations, 2) with (lat, lon) coordinates
        
    Returns:
        Distance matrix of shape (n_stations, n_stations)
    """
    return squareform(pdist(station_coords, metric='euclidean'))


def analyze_distance_distribution(station_coords: np.ndarray) -> Dict[str, float]:
    """
    Analyze the distribution of distances between stations.
    
    Args:
        station_coords: Array of shape (n_stations, 2) with (lat, lon) coordinates
        
    Returns:
        Dictionary with distance statistics
    """
    distances = calculate_station_distances(station_coords)
    # Get upper triangle (excluding diagonal) to avoid double counting
    upper_triangle = distances[np.triu_indices_from(distances, k=1)]
    
    return {
        'mean': float(np.mean(upper_triangle)),
        'median': float(np.median(upper_triangle)),
        'std': float(np.std(upper_triangle)),
        'min': float(np.min(upper_triangle)),
        'max': float(np.max(upper_triangle)),
        'q25': float(np.percentile(upper_triangle, 25)),
        'q75': float(np.percentile(upper_triangle, 75))
    }
