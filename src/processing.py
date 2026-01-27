import os
import pandas as pd
from pathlib import Path
from collections import Counter
from typing import List, Tuple, Dict, Any
from .utils import setup_logger

logger = setup_logger(__name__)

class DataProcessor:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.paths = config['paths']
        # Convert string paths to Path objects
        self.data_folder = Path(self.paths['data_folder'])
        self.data_desc_folder = Path(self.paths['data_desc_folder'])
        self.save_data_folder = Path(self.paths['save_data_folder'])
        
        # Ensure directories exist
        self.save_data_folder.mkdir(parents=True, exist_ok=True)

    def convert_data_desc_to_excel(self) -> None:
        """Convert parquet description files to Excel format for easier reading."""
        if not self.data_desc_folder.exists():
            logger.warning(f"Data description folder not found: {self.data_desc_folder}")
            return

        files = list(self.data_desc_folder.glob('*.parquet'))
        if not files:
            logger.info("No parquet files found in description folder.")
            return

        # Check if excel files already exist (simple check)
        if any(f.with_suffix('.xlsx').exists() for f in files):
            return

        logger.info('Converting data description files from parquet to excel...')
        for file_path in files:
            try:
                df = pd.read_parquet(file_path)
                excel_path = file_path.with_suffix('.xlsx')
                df.to_excel(excel_path)
                logger.debug(f"Converted {file_path.name} to {excel_path.name}")
            except Exception as e:
                logger.error(f"Failed to convert {file_path.name}: {e}")
        
        logger.info(f'Conversion complete. Files saved to: {self.data_desc_folder}')

    def get_dic_vars(self) -> Tuple[Dict[str, str], List[str]]:
        """Load parameter dictionary."""
        file_path = self.data_desc_folder / 'parameters.parquet'
        df = pd.read_parquet(file_path)
        dic_vars = {v: k for k, v in df['param_short'].items()}
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
        valid_stations, valid_cols = self._get_valid_stations(df.columns.tolist(), len(dic_vars))
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
            file_path = self.data_folder / f'{year}.parquet'
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
        """Main processing flow: merge data and save by variable."""
        self.convert_data_desc_to_excel()
        
        _, vars_list = self.get_dic_vars()
        df_merge, _ = self.merge_data()
        
        logger.info(f"Saving processed data to {self.save_data_folder}...")
        for var in vars_list:
            # Filter columns for specific variable
            # Columns are named 'Station~Variable'
            df_var = df_merge.filter(like=f'~{var}')
            
            # Clean column names to just 'Station'
            df_var.columns = [col.split('~')[0] for col in df_var.columns]
            
            save_path = self.save_data_folder / f'{var}.parquet'
            df_var.to_parquet(save_path)
            logger.info(f"Saved {var} data: {save_path.name}")
