# Missing Pattern Generation Usage Examples

This document provides comprehensive examples of how to use the missing pattern generation functionality.

## Overview

The missing pattern generation module supports three types of missing patterns:

1. **MCAR (Missing Completely At Random)**: Values are missing completely at random across all time points and stations
2. **SEQ (Sequential Missing)**: Continuous sequences of missing values at individual stations with random lengths (1 to max_length)
3. **SPATIAL (Spatially Correlated Sequential Missing)**: Missing sequences occur at spatially neighboring stations simultaneously

## Basic Usage

### 1. Import Required Modules

```python
import pandas as pd
import numpy as np
from src.missing_patterns import MissingPatternGenerator
from src.missing_generator import MissingDataGenerator
```

### 2. Load Your Data

```python
# Load time series data (rows=time, columns=stations)
data = pd.read_parquet('data/processed-data/temperature.parquet')

# Load station coordinates for spatial patterns
stations = pd.read_parquet('data/meteo-swiss-description/stations.parquet')
station_coords = stations[['latitude', 'longitude']].values
```

### 3. Generate Missing Patterns

#### MCAR Pattern

```python
# Create generator with your data
generator = MissingDataGenerator(data, random_state=42)

# Apply MCAR pattern with 20% missing rate
data_with_missing = generator.apply_mcar(missing_rate=0.2)

# Check actual missing rate
print(f"Actual missing rate: {generator.get_actual_missing_rate():.2%}")
```

#### SEQ Pattern

```python
# Create generator
generator = MissingDataGenerator(data, random_state=42)

# Apply SEQ pattern with 30% missing rate and max sequence length of 100
data_with_missing = generator.apply_seq(
    missing_rate=0.3,
    max_length=100  # Maximum continuous missing length
)

# Get missing statistics
stats = generator.get_missing_statistics()
print(stats)
```

#### SPATIAL Pattern

```python
# Create generator
generator = MissingDataGenerator(data, random_state=42)

# Apply SPATIAL pattern
data_with_missing = generator.apply_spatial(
    missing_rate=0.25,
    station_coords=station_coords,
    distance_threshold=0.5,  # Stations within this distance are neighbors
    max_length=100,  # Max sequence length
    n_neighbors_range=(1, 5)  # Randomly select 1-5 neighbors
)
```

### 4. Using the Generic Interface

```python
# Create generator
generator = MissingDataGenerator(data, random_state=42)

# Apply pattern using string identifier
data_with_missing = generator.apply_pattern(
    pattern_type='spatial',
    missing_rate=0.3,
    station_coords=station_coords,
    distance_threshold=0.5,
    max_length=100,
    n_neighbors_range=(2, 4)
)
```

## Advanced Examples

### Example 1: Comparing Different Missing Rates

```python
import matplotlib.pyplot as plt
import seaborn as sns

missing_rates = [0.1, 0.2, 0.3, 0.4]
fig, axes = plt.subplots(len(missing_rates), 1, figsize=(15, 12))

for i, rate in enumerate(missing_rates):
    generator = MissingDataGenerator(data, random_state=42)
    data_missing = generator.apply_mcar(missing_rate=rate)
    
    # Visualize missing pattern
    sns.heatmap(
        data_missing.iloc[:1000, :20].isnull().T,
        cmap='RdYlGn_r',
        cbar=False,
        ax=axes[i]
    )
    axes[i].set_title(f'Missing Rate: {rate:.0%}')
    axes[i].set_ylabel('Station')
    
plt.tight_layout()
plt.show()
```

### Example 2: Analyzing Spatial Neighborhood

```python
from src.missing_patterns import calculate_station_distances, analyze_distance_distribution

# Calculate station distances
distances = calculate_station_distances(station_coords)

# Analyze distance distribution
dist_stats = analyze_distance_distribution(station_coords)

print("Distance Statistics:")
for key, value in dist_stats.items():
    print(f"  {key}: {value:.4f}")

# Determine appropriate threshold (e.g., median or Q25)
threshold = dist_stats['median']
print(f"\nSuggested threshold: {threshold:.4f}")

# Check how many neighbors each station has
neighbors_count = (distances <= threshold).sum(axis=1) - 1
print(f"Average neighbors per station: {neighbors_count.mean():.1f}")
```

### Example 3: Multiple Pattern Comparison

```python
# Prepare subset of data for visualization
subset_data = data.iloc[:1000, :10]
subset_coords = station_coords[:10]

patterns = ['mcar', 'seq', 'spatial']
fig, axes = plt.subplots(len(patterns), 1, figsize=(15, 10))

for i, pattern in enumerate(patterns):
    generator = MissingDataGenerator(subset_data, random_state=42)
    
    if pattern == 'mcar':
        data_missing = generator.apply_mcar(missing_rate=0.3)
        title = 'MCAR Pattern'
    elif pattern == 'seq':
        data_missing = generator.apply_seq(missing_rate=0.3, max_length=50)
        title = 'SEQ Pattern (max_length=50)'
    else:  # spatial
        data_missing = generator.apply_spatial(
            missing_rate=0.3,
            station_coords=subset_coords,
            distance_threshold=0.5,
            max_length=50
        )
        title = 'SPATIAL Pattern'
    
    sns.heatmap(
        data_missing.isnull().T,
        cmap='RdYlGn_r',
        cbar=False,
        xticklabels=False,
        ax=axes[i]
    )
    axes[i].set_title(title)
    axes[i].set_ylabel('Station')

axes[-1].set_xlabel('Time')
plt.tight_layout()
plt.show()
```

### Example 4: Getting and Using the Mask

```python
# Generate pattern
generator = MissingDataGenerator(data, random_state=42)
data_missing = generator.apply_seq(missing_rate=0.2, max_length=100)

# Get the boolean mask
mask = generator.get_mask()  # True where values are missing

# Use the mask for analysis
print(f"Total missing values: {mask.sum()}")
print(f"Missing per station:\n{mask.sum(axis=0)}")

# Apply the same mask to different variables
other_data = pd.read_parquet('data/processed-data/pressure.parquet')
other_data_missing = other_data.copy()
other_data_missing[mask] = np.nan
```

### Example 5: Reproducible Experiments

```python
# Use random_state for reproducibility
def run_experiment(pattern_type, missing_rate, seed):
    generator = MissingDataGenerator(data, random_state=seed)
    
    if pattern_type == 'mcar':
        data_missing = generator.apply_mcar(missing_rate)
    elif pattern_type == 'seq':
        data_missing = generator.apply_seq(missing_rate, max_length=100)
    else:
        data_missing = generator.apply_spatial(
            missing_rate,
            station_coords=station_coords,
            distance_threshold=0.5,
            max_length=100
        )
    
    return generator.get_missing_statistics()

# Run multiple experiments with same seed
results = []
for seed in [42, 43, 44]:
    stats = run_experiment('seq', 0.3, seed)
    results.append(stats)
    print(f"Seed {seed}: Missing rate = {stats['missing_rate']:.2%}")
```

## Tips and Best Practices

1. **Choosing Missing Rate**: Start with lower rates (10-20%) for testing, then increase as needed
2. **SEQ max_length**: Set based on your data frequency and expected gap lengths
3. **SPATIAL threshold**: Use distance distribution analysis to choose appropriate threshold
4. **Random State**: Always set `random_state` for reproducible experiments
5. **Visualization**: Always visualize patterns to ensure they match expectations

## Common Parameters

| Parameter | Description | Typical Range |
|-----------|-------------|---------------|
| `missing_rate` | Overall proportion of missing values | 0.1 - 0.5 |
| `max_length` | Maximum continuous missing sequence | 10 - 200 timesteps |
| `distance_threshold` | Max distance for spatial neighbors | Q25 - Median of distance distribution |
| `n_neighbors_range` | Range of neighbors to affect | (1, 3) - (1, 5) |

## Troubleshooting

### Issue: Actual missing rate differs from target

This is normal, especially for SEQ and SPATIAL patterns. The algorithms try to approximate the target rate while maintaining pattern characteristics. Allow 10-15% tolerance.

### Issue: No neighbors found in SPATIAL pattern

The `distance_threshold` might be too small. Analyze the distance distribution and increase the threshold.

### Issue: Pattern doesn't look right

- Check your data orientation (rows should be time, columns should be stations)
- Verify station coordinates are in the correct format (n_stations × 2)
- Visualize small subsets first to understand the pattern
