"""
Unit tests for missing pattern generation.
"""

import numpy as np
import pandas as pd
import pytest
from src.missing_patterns import (
    MissingPatternGenerator,
    calculate_station_distances,
    analyze_distance_distribution
)
from src.missing_generator import MissingDataGenerator


class TestMissingPatternGenerator:
    """Test the MissingPatternGenerator class."""
    
    def setup_method(self):
        """Setup test data."""
        self.n_timesteps = 1000
        self.n_stations = 10
        self.data_shape = (self.n_timesteps, self.n_stations)
        self.generator = MissingPatternGenerator(
            data_shape=self.data_shape,
            random_state=42
        )
    
    def test_mcar_shape(self):
        """Test MCAR pattern has correct shape."""
        mask = self.generator.generate_mcar(missing_rate=0.1)
        assert mask.shape == self.data_shape
    
    def test_mcar_type(self):
        """Test MCAR pattern returns boolean array."""
        mask = self.generator.generate_mcar(missing_rate=0.1)
        assert mask.dtype == bool
    
    def test_mcar_rate(self):
        """Test MCAR pattern achieves approximately correct missing rate."""
        missing_rate = 0.2
        mask = self.generator.generate_mcar(missing_rate=missing_rate)
        actual_rate = mask.sum() / mask.size
        # Allow 5% tolerance for randomness
        assert abs(actual_rate - missing_rate) < 0.05
    
    def test_mcar_invalid_rate(self):
        """Test MCAR raises error for invalid missing rate."""
        with pytest.raises(ValueError):
            self.generator.generate_mcar(missing_rate=1.5)
        with pytest.raises(ValueError):
            self.generator.generate_mcar(missing_rate=-0.1)
    
    def test_seq_shape(self):
        """Test SEQ pattern has correct shape."""
        mask = self.generator.generate_seq(missing_rate=0.1, max_length=50)
        assert mask.shape == self.data_shape
    
    def test_seq_type(self):
        """Test SEQ pattern returns boolean array."""
        mask = self.generator.generate_seq(missing_rate=0.1, max_length=50)
        assert mask.dtype == bool
    
    def test_seq_rate(self):
        """Test SEQ pattern achieves approximately correct missing rate."""
        missing_rate = 0.2
        mask = self.generator.generate_seq(missing_rate=missing_rate, max_length=50)
        actual_rate = mask.sum() / mask.size
        # Allow 10% tolerance for sequential patterns
        assert abs(actual_rate - missing_rate) < 0.1
    
    def test_seq_continuity(self):
        """Test SEQ pattern creates continuous sequences."""
        mask = self.generator.generate_seq(missing_rate=0.3, max_length=20)
        
        # Check at least one station has a continuous sequence
        has_sequence = False
        for station_idx in range(self.n_stations):
            station_mask = mask[:, station_idx]
            if station_mask.sum() > 0:
                # Find continuous sequences
                diff = np.diff(np.concatenate(([False], station_mask, [False])).astype(int))
                starts = np.where(diff == 1)[0]
                ends = np.where(diff == -1)[0]
                lengths = ends - starts
                if len(lengths) > 0 and np.max(lengths) > 1:
                    has_sequence = True
                    break
        
        assert has_sequence, "SEQ pattern should create continuous sequences"
    
    def test_seq_invalid_params(self):
        """Test SEQ raises error for invalid parameters."""
        with pytest.raises(ValueError):
            self.generator.generate_seq(missing_rate=1.5, max_length=50)
        with pytest.raises(ValueError):
            self.generator.generate_seq(missing_rate=0.2, max_length=0)
    
    def test_spatial_shape(self):
        """Test SPATIAL pattern has correct shape."""
        # Create synthetic station coordinates
        station_coords = np.random.RandomState(42).rand(self.n_stations, 2) * 10
        mask = self.generator.generate_spatial(
            missing_rate=0.1,
            station_coords=station_coords,
            distance_threshold=2.0,
            max_length=50
        )
        assert mask.shape == self.data_shape
    
    def test_spatial_type(self):
        """Test SPATIAL pattern returns boolean array."""
        station_coords = np.random.RandomState(42).rand(self.n_stations, 2) * 10
        mask = self.generator.generate_spatial(
            missing_rate=0.1,
            station_coords=station_coords,
            distance_threshold=2.0,
            max_length=50
        )
        assert mask.dtype == bool
    
    def test_spatial_rate(self):
        """Test SPATIAL pattern achieves approximately correct missing rate."""
        missing_rate = 0.2
        station_coords = np.random.RandomState(42).rand(self.n_stations, 2) * 10
        mask = self.generator.generate_spatial(
            missing_rate=missing_rate,
            station_coords=station_coords,
            distance_threshold=2.0,
            max_length=50
        )
        actual_rate = mask.sum() / mask.size
        # Allow 15% tolerance for spatial patterns
        assert abs(actual_rate - missing_rate) < 0.15
    
    def test_spatial_invalid_coords(self):
        """Test SPATIAL raises error for invalid coordinates."""
        # Wrong number of stations
        station_coords = np.random.rand(self.n_stations + 1, 2) * 10
        with pytest.raises(ValueError):
            self.generator.generate_spatial(
                missing_rate=0.1,
                station_coords=station_coords,
                distance_threshold=2.0,
                max_length=50
            )
    
    def test_reproducibility(self):
        """Test that patterns are reproducible with same random state."""
        gen1 = MissingPatternGenerator(self.data_shape, random_state=42)
        gen2 = MissingPatternGenerator(self.data_shape, random_state=42)
        
        mask1 = gen1.generate_mcar(missing_rate=0.2)
        mask2 = gen2.generate_mcar(missing_rate=0.2)
        
        assert np.array_equal(mask1, mask2)


class TestDistanceFunctions:
    """Test distance calculation functions."""
    
    def test_calculate_station_distances(self):
        """Test station distance calculation."""
        # Create simple 2D coordinates
        coords = np.array([[0, 0], [1, 0], [0, 1]])
        distances = calculate_station_distances(coords)
        
        assert distances.shape == (3, 3)
        assert distances[0, 0] == 0  # Distance to self is 0
        assert abs(distances[0, 1] - 1.0) < 1e-10  # Distance between (0,0) and (1,0)
        assert abs(distances[0, 2] - 1.0) < 1e-10  # Distance between (0,0) and (0,1)
    
    def test_analyze_distance_distribution(self):
        """Test distance distribution analysis."""
        coords = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
        stats = analyze_distance_distribution(coords)
        
        assert 'mean' in stats
        assert 'median' in stats
        assert 'std' in stats
        assert 'min' in stats
        assert 'max' in stats
        assert stats['min'] > 0  # Minimum distance should be positive
        assert stats['max'] >= stats['min']  # Max should be >= min


class TestMissingDataGenerator:
    """Test the MissingDataGenerator class."""
    
    def setup_method(self):
        """Setup test data."""
        np.random.seed(42)
        self.n_timesteps = 1000
        self.n_stations = 10
        
        # Create synthetic time series data
        dates = pd.date_range('2023-01-01', periods=self.n_timesteps, freq='10min')
        data = np.random.randn(self.n_timesteps, self.n_stations) * 10 + 20
        self.data = pd.DataFrame(
            data,
            index=dates,
            columns=[f'Station_{i}' for i in range(self.n_stations)]
        )
        self.generator = MissingDataGenerator(self.data, random_state=42)
    
    def test_initialization(self):
        """Test generator initialization."""
        assert self.generator.data.shape == self.data.shape
        assert self.generator.mask is None
    
    def test_apply_mcar(self):
        """Test applying MCAR pattern."""
        result = self.generator.apply_mcar(missing_rate=0.2)
        
        assert result.shape == self.data.shape
        assert result.isnull().sum().sum() > 0  # Should have missing values
        
        actual_rate = result.isnull().sum().sum() / result.size
        assert abs(actual_rate - 0.2) < 0.05
    
    def test_apply_seq(self):
        """Test applying SEQ pattern."""
        result = self.generator.apply_seq(missing_rate=0.2, max_length=50)
        
        assert result.shape == self.data.shape
        assert result.isnull().sum().sum() > 0
    
    def test_apply_spatial(self):
        """Test applying SPATIAL pattern."""
        station_coords = np.random.RandomState(42).rand(self.n_stations, 2) * 10
        result = self.generator.apply_spatial(
            missing_rate=0.2,
            station_coords=station_coords,
            distance_threshold=2.0,
            max_length=50
        )
        
        assert result.shape == self.data.shape
        assert result.isnull().sum().sum() > 0
    
    def test_apply_pattern_mcar(self):
        """Test generic apply_pattern method with MCAR."""
        result = self.generator.apply_pattern('mcar', missing_rate=0.2)
        assert result.isnull().sum().sum() > 0
    
    def test_apply_pattern_seq(self):
        """Test generic apply_pattern method with SEQ."""
        result = self.generator.apply_pattern('seq', missing_rate=0.2, max_length=50)
        assert result.isnull().sum().sum() > 0
    
    def test_apply_pattern_spatial(self):
        """Test generic apply_pattern method with SPATIAL."""
        station_coords = np.random.RandomState(42).rand(self.n_stations, 2) * 10
        result = self.generator.apply_pattern(
            'spatial',
            missing_rate=0.2,
            station_coords=station_coords,
            distance_threshold=2.0,
            max_length=50
        )
        assert result.isnull().sum().sum() > 0
    
    def test_apply_pattern_invalid(self):
        """Test apply_pattern raises error for invalid pattern type."""
        with pytest.raises(ValueError):
            self.generator.apply_pattern('invalid', missing_rate=0.2)
    
    def test_get_mask(self):
        """Test getting the mask."""
        assert self.generator.get_mask() is None
        
        self.generator.apply_mcar(missing_rate=0.2)
        mask = self.generator.get_mask()
        
        assert mask is not None
        assert mask.shape == self.data.shape
    
    def test_get_actual_missing_rate(self):
        """Test getting actual missing rate."""
        assert self.generator.get_actual_missing_rate() == 0.0
        
        self.generator.apply_mcar(missing_rate=0.2)
        actual_rate = self.generator.get_actual_missing_rate()
        
        assert 0 < actual_rate < 1
        assert abs(actual_rate - 0.2) < 0.05
    
    def test_get_missing_statistics(self):
        """Test getting missing statistics."""
        self.generator.apply_mcar(missing_rate=0.2)
        stats = self.generator.get_missing_statistics()
        
        assert 'total_values' in stats
        assert 'missing_values' in stats
        assert 'missing_rate' in stats
        assert 'missing_per_station' in stats
        assert stats['missing_values'] > 0
    
    def test_reset(self):
        """Test resetting to original data."""
        self.generator.apply_mcar(missing_rate=0.2)
        assert self.generator.data.isnull().sum().sum() > 0
        
        self.generator.reset()
        assert self.generator.data.isnull().sum().sum() == 0
        assert self.generator.mask is None
    
    def test_original_data_unchanged(self):
        """Test that original data is not modified."""
        original_copy = self.data.copy()
        self.generator.apply_mcar(missing_rate=0.2)
        
        # Original data should still be intact
        assert np.array_equal(self.data.values, original_copy.values, equal_nan=True)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
