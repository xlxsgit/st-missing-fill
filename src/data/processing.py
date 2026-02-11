from pathlib import Path
from collections import Counter
from typing import Dict, List, Tuple

import pandas as pd
from sklearn.cluster import KMeans

from src.utils import setup_logger

logger = setup_logger(__name__)


class DataProcessor:
    def __init__(self):
        """Centralized configuration: modify everything here."""
        # ===== Paths =====
        self.raw_data_folder = Path("data/raw/ts_data")
        self.geo_data_folder = Path("data/raw/geo_data")
        self.processed_data_folder = Path("data/processed")
        self.processed_data_folder.mkdir(parents=True, exist_ok=True)

        # ===== Time range =====
        self.start_year = 2023
        self.end_year = 2024
        self.years = range(self.start_year, self.end_year + 1)

        # ===== Load variable dictionary once =====
        self.dic_vars, self.var_names = self._load_var_dict()

    # ------------------------------------------------------------------
    # Basic loaders
    # ------------------------------------------------------------------
    def _load_var_dict(self) -> Tuple[Dict[str, str], List[str]]:
        file_path = self.geo_data_folder / "vars.csv"
        df = pd.read_csv(file_path)
        dic_vars = dict(zip(df["param_short"], df["name"]))
        return dic_vars, list(dic_vars.values())

    @staticmethod
    def _get_valid_stations(
        cols: List[Tuple[str, str]], n_vars: int
    ) -> Tuple[List[str], List[Tuple[str, str]]]:
        """Keep stations that contain all variables."""
        station_cnt = Counter(s for s, _ in cols)
        valid_stations = [s for s, c in station_cnt.items() if c == n_vars]
        valid_cols = [c for c in cols if c[0] in valid_stations]
        return valid_stations, valid_cols

    # ------------------------------------------------------------------
    # Core processing
    # ------------------------------------------------------------------
    def load_single_year(self, year: int) -> Tuple[pd.DataFrame, List[str]]:
        file_path = self.raw_data_folder / f"{year}.parquet"
        if not file_path.exists():
            raise FileNotFoundError(file_path)

        df = pd.read_parquet(file_path)

        valid_stations, valid_cols = self._get_valid_stations(
            df.columns.tolist(), len(self.dic_vars)
        )
        df = df.loc[:, valid_cols]
        df.columns = [f"{s}~{self.dic_vars[p]}" for s, p in df.columns]

        return df, valid_stations

    def merge_data(self) -> Tuple[pd.DataFrame, List[str]]:
        logger.info(f"Merging data from {self.start_year} to {self.end_year}...")

        dfs = []
        ref_stations = None
        ref_cols = None

        for year in self.years:
            df, stations = self.load_single_year(year)

            if ref_stations is None:
                ref_stations = stations
                ref_cols = df.columns.tolist()
            else:
                if set(stations) != set(ref_stations):
                    raise ValueError(f"Station mismatch in year {year}")
                if df.columns.tolist() != ref_cols:
                    raise ValueError(f"Column mismatch in year {year}")

            dfs.append(df)

        df_merge = pd.concat(dfs, axis=0)

        # Rebuild complete time index
        full_index = pd.date_range(
            df_merge.index.min(),
            df_merge.index.max(),
            freq="10min",
        )
        if len(full_index) != len(df_merge):
            logger.warning(
                f"Time index incomplete: expected {len(full_index)}, got {len(df_merge)}"
            )

        df_merge.index = full_index
        return df_merge, ref_stations

    def process_and_save(self) -> None:
        df_merge, _ = self.merge_data()
        save_path = self.processed_data_folder / "all_data.parquet"
        df_merge.to_parquet(save_path)
        logger.info(f"Saved merged data → {save_path}")

    # ------------------------------------------------------------------
    # Station metadata
    # ------------------------------------------------------------------
    def generate_all_stations(self) -> None:
        all_data_path = self.processed_data_folder / "all_data.parquet"
        stations_csv = self.geo_data_folder / "stations.csv"
        output_path = self.processed_data_folder / "all_stations.csv"

        if not all_data_path.exists():
            raise FileNotFoundError(all_data_path)

        df_data = pd.read_parquet(all_data_path)
        stations_used = {c.split("~")[0] for c in df_data.columns}

        df = pd.read_csv(stations_csv)
        df = df[df["nat_abbr"].isin(stations_used)]

        cols = [
            "nat_abbr", "latitude", "longitude", "station_height",
            "swiss_easting", "swiss_northing", "dem",
            "TPI_2000M", "ASPECT_2000M_SIGRATIO1", "SLOPE_2000M_SIGRATIO1",
            "STD_2000M", "WE_DERIVATIVE_2000M_SIGRATIO1", "SN_DERIVATIVE_2000M_SIGRATIO1",
        ]
        df = df[cols]
        df.columns = [
            "Station", "Lat", "Lon", "Height",
            "X", "Y", "Z",
            "TPI", "ASPECT", "SLOPE", "STD",
            "D_WE", "D_SN",
        ]

        kmeans = KMeans(n_clusters=10, random_state=42)
        df["cluster"] = kmeans.fit_predict(df[["X", "Y", "Z"]])

        df = df.sort_values(["cluster", "Station"]).reset_index(drop=True)
        df.to_csv(output_path, index=False)

        logger.info(f"Saved station metadata → {output_path}")
