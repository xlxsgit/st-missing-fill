"""
Missing Data Generator Module

This module provides functionality to apply missing patterns to time series data.
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict, Any
from .missing_patterns import MissingPatternGenerator


class MissingDataGenerator:
    """Apply missing patterns to time series data."""
    
    def __init__(self, data: pd.DataFrame, random_state: Optional[int] = None):
        """
        Initialize the missing data generator.
        
        Args:
            data: DataFrame with time series data (rows=time, columns=stations)
            random_state: Random seed for reproducibility
        """
        self.data = data.copy()
        self.original_data = data.copy()
        self.random_state = random_state
        self.pattern_generator = MissingPatternGenerator(
            data_shape=data.shape,
            random_state=random_state
        )
        self.mask = None
    
    def apply_mcar(self, missing_rate: float) -> pd.DataFrame:
        """
        Apply MCAR (Missing Completely At Random) pattern.
        
        Args:
            missing_rate: Overall missing rate (0 to 1)
            
        Returns:
            DataFrame with missing values applied
        """
        self.mask = self.pattern_generator.generate_mcar(missing_rate)
        self.data = self.original_data.copy()
        self.data[self.mask] = np.nan
        return self.data
    
    def apply_seq(self, missing_rate: float, max_length: int = 100) -> pd.DataFrame:
        """
        Apply Sequential missing pattern.
        
        Args:
            missing_rate: Overall target missing rate (0 to 1)
            max_length: Maximum length of continuous missing sequence
            
        Returns:
            DataFrame with missing values applied
        """
        self.mask = self.pattern_generator.generate_seq(missing_rate, max_length)
        self.data = self.original_data.copy()
        self.data[self.mask] = np.nan
        return self.data
    
    def apply_spatial(
        self,
        missing_rate: float,
        station_coords: np.ndarray,
        distance_threshold: float,
        max_length: int = 100,
        n_neighbors_range: Tuple[int, int] = (1, 5)
    ) -> pd.DataFrame:
        """
        Apply Spatially correlated sequential missing pattern.
        
        Args:
            missing_rate: Overall target missing rate (0 to 1)
            station_coords: Array of shape (n_stations, 2) with (lat, lon) coordinates
            distance_threshold: Maximum distance for neighbors to be affected
            max_length: Maximum length of continuous missing sequence
            n_neighbors_range: Range (min, max) for number of neighbors to affect
            
        Returns:
            DataFrame with missing values applied
        """
        self.mask = self.pattern_generator.generate_spatial(
            missing_rate=missing_rate,
            station_coords=station_coords,
            distance_threshold=distance_threshold,
            max_length=max_length,
            n_neighbors_range=n_neighbors_range
        )
        self.data = self.original_data.copy()
        self.data[self.mask] = np.nan
        return self.data
    
    def apply_pattern(
        self,
        pattern_type: str,
        missing_rate: float,
        **kwargs
    ) -> pd.DataFrame:
        """
        Apply a missing pattern to the data.
        
        Args:
            pattern_type: Type of pattern ('mcar', 'seq', 'spatial')
            missing_rate: Overall target missing rate (0 to 1)
            **kwargs: Additional parameters specific to each pattern type
                For 'seq': max_length (default: 100)
                For 'spatial': station_coords, distance_threshold, max_length (default: 100),
                              n_neighbors_range (default: (1, 5))
            
        Returns:
            DataFrame with missing values applied
        """
        pattern_type = pattern_type.lower()
        
        if pattern_type == 'mcar':
            return self.apply_mcar(missing_rate)
        
        elif pattern_type == 'seq':
            max_length = kwargs.get('max_length', 100)
            return self.apply_seq(missing_rate, max_length)
        
        elif pattern_type == 'spatial':
            if 'station_coords' not in kwargs or 'distance_threshold' not in kwargs:
                raise ValueError(
                    "spatial pattern requires 'station_coords' and 'distance_threshold'"
                )
            station_coords = kwargs['station_coords']
            distance_threshold = kwargs['distance_threshold']
            max_length = kwargs.get('max_length', 100)
            n_neighbors_range = kwargs.get('n_neighbors_range', (1, 5))
            
            return self.apply_spatial(
                missing_rate=missing_rate,
                station_coords=station_coords,
                distance_threshold=distance_threshold,
                max_length=max_length,
                n_neighbors_range=n_neighbors_range
            )
        
        else:
            raise ValueError(
                f"Unknown pattern type: {pattern_type}. "
                f"Must be one of: 'mcar', 'seq', 'spatial'"
            )
    
    def get_mask(self) -> Optional[np.ndarray]:
        """
        Get the current missing mask.
        
        Returns:
            Boolean mask where True indicates missing values, or None if no pattern applied
        """
        return self.mask
    
    def get_actual_missing_rate(self) -> float:
        """
        Calculate the actual missing rate in the data.
        
        Returns:
            Actual missing rate (0 to 1)
        """
        if self.mask is None:
            return 0.0
        return self.mask.sum() / self.mask.size
    
    def get_missing_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the missing data.
        
        Returns:
            Dictionary with missing data statistics
        """
        if self.mask is None:
            return {
                'total_values': self.data.size,
                'missing_values': 0,
                'missing_rate': 0.0,
                'missing_per_station': {}
            }
        
        missing_per_station = self.mask.sum(axis=0)
        
        return {
            'total_values': self.mask.size,
            'missing_values': int(self.mask.sum()),
            'missing_rate': float(self.mask.sum() / self.mask.size),
            'missing_per_station': {
                'mean': float(missing_per_station.mean()),
                'std': float(missing_per_station.std()),
                'min': int(missing_per_station.min()),
                'max': int(missing_per_station.max())
            }
        }
    
    def reset(self) -> None:
        """Reset to original data without missing values."""
        self.data = self.original_data.copy()
        self.mask = None
