import os
import pandas as pd
from pathlib import Path
from collections import Counter
from typing import List, Tuple, Dict, Any
from src.utils import setup_logger

logger = setup_logger(__name__)

class DataProcessor:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.paths = config['paths']
        # Convert string paths to Path objects
        self.raw_data_folder = Path(self.paths['raw_data_folder'])
        self.geo_data_folder = Path(self.paths['geo_data_folder'])
        self.processed_data_folder = Path(self.paths['processed_data_folder'])
        
        # Ensure directories exist
        self.processed_data_folder.mkdir(parents=True, exist_ok=True)

    def get_dic_vars(self) -> Tuple[Dict[str, str], List[str]]:
        """Load parameter dictionary."""
        file_path = self.geo_data_folder / 'vars.csv'
        df = pd.read_csv(file_path)
        dic_vars = dict(zip(df['param_short'], df['name']))
        return dic_vars, list(dic_vars.values())

    def _get_valid_stations(self, raw_cols: List[Tuple[str, str]], n_vars: int) -> Tuple[List[str], List[Tuple[str, str]]]:
        """Filter stations that have all variables."""
        # raw_cols elements are like ('station_name', 'param_name') or similar structure based on previous code
        # But wait, parquet columns usually are strings. Let's check the original code.
        # Original: raw_station_cnt = Counter([i[0] for i in raw_cols])
        # It implies columns are tuples or MultiIndex. 
        # Checking `load_single_data`: df.columns.tolist(). 
        # Parquet doesn't natively support tuple columns easily unless MultiIndex.
        # Assuming the dataframe loaded from parquet has MultiIndex columns or tuples as columns.
        
        raw_station_cnt = Counter([i[0] for i in raw_cols])
        valid_stations = [k for k, v in raw_station_cnt.items() if v == n_vars]
        valid_cols = [i for i in raw_cols if i[0] in valid_stations]
        return valid_stations, valid_cols

    def load_single_data(self, file_path: Path, dic_vars: Dict[str, str]) -> Tuple[pd.DataFrame, List[str]]:
        """Load data for a single year and filter valid stations."""
        df = pd.read_parquet(file_path)
        valid_stations, valid_cols = self._get_valid_stations(df.columns.tolist(), len(dic_vars)) # pyright: ignore[reportArgumentType]
        df = df.loc[:, valid_cols]
        # Rename columns to format: station~variable
        df.columns = [f'{s}~{dic_vars[p]}' for s, p in df.columns]
        return df, valid_stations

    def merge_data(self) -> Tuple[pd.DataFrame, List[str]]:
        """Merge data across multiple years."""
        start_year = self.config['processing']['start_year']
        end_year = self.config['processing']['end_year']
        dic_vars, _ = self.get_dic_vars()
        
        df_merge = None
        merge_valid_stations = []

        logger.info(f"Merging data from {start_year} to {end_year}...")
        
        for year in range(start_year, end_year + 1):
            file_path = self.raw_data_folder / f'{year}.parquet'
            if not file_path.exists():
                logger.warning(f"Data file for year {year} not found: {file_path}")
                continue
                
            df, valid_stations = self.load_single_data(file_path, dic_vars)
            
            if df_merge is None:
                df_merge = df.copy()
                merge_valid_stations = valid_stations[:]
            else:
                # Validation
                if set(merge_valid_stations) != set(valid_stations):
                    raise ValueError(f'Station mismatch between years at {year}')
                
                # Align columns before concat if necessary, but robust code implies checking
                if df_merge.columns.tolist() != df.columns.tolist():
                     raise ValueError(f'Columns mismatch between years at {year}')
                
                df_merge = pd.concat([df_merge, df], axis=0)

        if df_merge is None:
            raise ValueError("No data loaded.")

        # Re-index
        datetime_index = pd.date_range(start=df_merge.index.min(), end=df_merge.index.max(), freq='10min')
        
        # Check for missing time steps
        expected_rows = (datetime_index[-1] - datetime_index[0]).total_seconds() / 600 + 1
        if df_merge.shape[0] != expected_rows:
             logger.warning(f"Time index missing entries! Expected {expected_rows}, got {df_merge.shape[0]}")
             
        df_merge.index = datetime_index
        return df_merge, merge_valid_stations

    def process_and_save(self) -> None:
        """Main processing flow: merge data and save into a single file."""
        df_merge, _ = self.merge_data()
        
        logger.info(f"Saving processed data to {self.processed_data_folder}...")
        
        # Save as a single parquet file
        save_path = self.processed_data_folder / 'all_data.parquet'
        df_merge.to_parquet(save_path)
        logger.info(f"Saved merged data to: {save_path.name}")

    def generate_all_stations(self) -> None:
        """Generate all_stations.csv from merged data and geo data."""
        all_data_path = self.processed_data_folder / 'all_data.parquet'
        stations_csv_path = self.geo_data_folder / 'stations.csv'
        output_path = self.processed_data_folder / 'all_stations.csv'
        
        if not all_data_path.exists():
            logger.error(f"all_data.parquet not found at {all_data_path}")
            return
        
        if not stations_csv_path.exists():
            logger.error(f"stations.csv not found at {stations_csv_path}")
            return
        
        logger.info("Generating all_stations.csv...")
        
        df_all_data = pd.read_parquet(all_data_path)
        stations = set([col.split('~')[0] for col in df_all_data.columns])
        
        df_stations = pd.read_csv(stations_csv_path)
        
        if 'station_name' in df_stations.columns:
            del df_stations['station_name']
        
        df_stations = df_stations.loc[:, 
                        ['nat_abbr', 'latitude', 'longitude', 'station_height', 'swiss_easting', 'swiss_northing', 'dem',
                        'TPI_2000M', 'ASPECT_2000M_SIGRATIO1', 'SLOPE_2000M_SIGRATIO1', 'STD_2000M', 
                        'WE_DERIVATIVE_2000M_SIGRATIO1', 'SN_DERIVATIVE_2000M_SIGRATIO1']]
        
        df_stations.columns = ['Station', 'Lat', 'Lon', 'Height', 'X', 'Y', 'Z',
                        'TPI', 'ASPECT', 'SLOPE', 'STD', 
                        'D_WE', 'D_SN']
        
        df_stations = df_stations[df_stations['Station'].isin(stations)]
        # 按照 x y z kmeans聚类，将站点分为10类
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=10, random_state=42)
        df_stations['cluster'] = kmeans.fit_predict(df_stations[['X', 'Y', 'Z']])
        df_stations = df_stations.sort_values(by='Station')
        
        df_stations.to_csv(output_path, index=False)
        logger.info(f"Saved all_stations.csv to: {output_path}")
    
