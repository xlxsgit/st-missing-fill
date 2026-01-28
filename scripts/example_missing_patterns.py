"""
Example script demonstrating missing pattern generation.

This script shows how to:
1. Load time series data
2. Generate different missing patterns
3. Visualize the patterns
4. Save results
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Import missing pattern modules
from src.missing_patterns import analyze_distance_distribution
from src.missing_generator import MissingDataGenerator
from src.utils import load_config

# Set plotting style
sns.set_theme(style="whitegrid", palette="muted")


def load_data():
    """Load processed time series data and station information."""
    config = load_config('config.yaml')
    
    # Load temperature data
    data_path = Path(config['paths']['save_data_folder']) / 'temperature.parquet'
    data = pd.read_parquet(data_path)
    
    # Load station coordinates
    stations_path = Path(config['paths']['data_desc_folder']) / 'stations.parquet'
    stations = pd.read_parquet(stations_path)
    station_coords = stations[['latitude', 'longitude']].values
    
    return data, station_coords, stations


def demonstrate_patterns(data, station_coords, output_dir='outputs'):
    """
    Demonstrate all three missing patterns.
    
    Args:
        data: Time series DataFrame
        station_coords: Station coordinates array
        output_dir: Directory to save outputs
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Use subset of data for demonstration
    subset_timesteps = 1000
    subset_stations = 10
    demo_data = data.iloc[:subset_timesteps, :subset_stations].copy()
    demo_coords = station_coords[:subset_stations]
    
    print(f"Demo data shape: {demo_data.shape}")
    print(f"Time range: {demo_data.index[0]} to {demo_data.index[-1]}")
    
    # Analyze distances for spatial pattern
    dist_stats = analyze_distance_distribution(demo_coords)
    threshold = (dist_stats['q25'] + dist_stats['median']) / 2
    print(f"\nUsing spatial threshold: {threshold:.4f}")
    
    # Configure patterns
    missing_rate = 0.3
    max_length = 100
    patterns = {
        'MCAR': {
            'type': 'mcar',
            'params': {'missing_rate': missing_rate}
        },
        'SEQ': {
            'type': 'seq',
            'params': {'missing_rate': missing_rate, 'max_length': max_length}
        },
        'SPATIAL': {
            'type': 'spatial',
            'params': {
                'missing_rate': missing_rate,
                'station_coords': demo_coords,
                'distance_threshold': threshold,
                'max_length': max_length,
                'n_neighbors_range': (1, 3)
            }
        }
    }
    
    # Generate and visualize patterns
    fig, axes = plt.subplots(len(patterns), 1, figsize=(15, 10))
    
    results = {}
    for i, (name, config) in enumerate(patterns.items()):
        print(f"\nGenerating {name} pattern...")
        
        # Generate pattern
        generator = MissingDataGenerator(demo_data, random_state=42)
        data_missing = generator.apply_pattern(config['type'], **config['params'])
        
        # Get statistics
        stats = generator.get_missing_statistics()
        results[name] = {
            'data': data_missing,
            'stats': stats,
            'mask': generator.get_mask()
        }
        
        print(f"  Actual missing rate: {stats['missing_rate']:.2%}")
        print(f"  Missing values: {stats['missing_values']}")
        
        # Visualize
        ax = axes[i] if len(patterns) > 1 else axes
        sns.heatmap(
            data_missing.isnull().T,
            cmap='RdYlGn_r',
            cbar=False,
            xticklabels=False,
            yticklabels=data_missing.columns,
            ax=ax
        )
        ax.set_title(
            f'{name} Pattern (Missing Rate: {stats["missing_rate"]:.2%})',
            fontsize=12,
            fontweight='bold'
        )
        ax.set_ylabel('Station')
    
    axes[-1].set_xlabel('Time')
    plt.tight_layout()
    
    # Save figure
    output_file = output_path / 'missing_patterns_comparison.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nFigure saved to: {output_file}")
    plt.show()
    
    # Save missing data to files
    for name, result in results.items():
        output_file = output_path / f'{name.lower()}_missing_data.parquet'
        result['data'].to_parquet(output_file)
        print(f"Saved {name} data to: {output_file}")
    
    return results


def demonstrate_parameter_sensitivity(data, station_coords, output_dir='outputs'):
    """
    Demonstrate how parameters affect the patterns.
    
    Args:
        data: Time series DataFrame
        station_coords: Station coordinates array
        output_dir: Directory to save outputs
    """
    # Use subset
    subset_timesteps = 1000
    subset_stations = 10
    demo_data = data.iloc[:subset_timesteps, :subset_stations].copy()
    
    # Test different missing rates for MCAR
    missing_rates = [0.1, 0.2, 0.3, 0.4]
    fig, axes = plt.subplots(len(missing_rates), 1, figsize=(15, 10))
    
    print("\nTesting different missing rates with MCAR pattern:")
    for i, rate in enumerate(missing_rates):
        generator = MissingDataGenerator(demo_data, random_state=42)
        data_missing = generator.apply_mcar(missing_rate=rate)
        stats = generator.get_missing_statistics()
        
        print(f"  Target: {rate:.0%}, Actual: {stats['missing_rate']:.2%}")
        
        sns.heatmap(
            data_missing.isnull().T,
            cmap='RdYlGn_r',
            cbar=False,
            xticklabels=False,
            yticklabels=False,
            ax=axes[i]
        )
        axes[i].set_title(
            f'MCAR with {rate:.0%} Missing Rate (Actual: {stats["missing_rate"]:.2%})',
            fontsize=11
        )
        axes[i].set_ylabel('Station')
    
    axes[-1].set_xlabel('Time')
    plt.tight_layout()
    
    output_path = Path(output_dir)
    output_file = output_path / 'missing_rate_sensitivity.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Figure saved to: {output_file}")
    plt.show()


def main():
    """Main execution function."""
    print("=" * 60)
    print("Missing Pattern Generation Example")
    print("=" * 60)
    
    try:
        # Load data
        print("\nLoading data...")
        data, station_coords, stations = load_data()
        print(f"Loaded data shape: {data.shape}")
        print(f"Number of stations: {len(stations)}")
        
        # Demonstrate patterns
        print("\n" + "=" * 60)
        print("Demonstrating Missing Patterns")
        print("=" * 60)
        results = demonstrate_patterns(data, station_coords)
        
        # Demonstrate parameter sensitivity
        print("\n" + "=" * 60)
        print("Demonstrating Parameter Sensitivity")
        print("=" * 60)
        demonstrate_parameter_sensitivity(data, station_coords)
        
        print("\n" + "=" * 60)
        print("Example completed successfully!")
        print("=" * 60)
        
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("\nPlease ensure you have:")
        print("1. Run the data processing script first: uv run scripts/run_processing.py")
        print("2. Data files exist in data/processed-data/ and data/meteo-swiss-description/")
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
