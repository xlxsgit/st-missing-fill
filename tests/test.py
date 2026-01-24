import pandas as pd

file = 'data/2025.parquet'

df = pd.read_parquet(file, engine="fastparquet")

