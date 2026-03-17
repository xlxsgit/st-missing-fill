import pandas as pd
import os

csv_path = 'docs/body/hpo_parameters.csv'
xlsx_path = 'docs/body/hpo_parameters.xlsx'

if os.path.exists(csv_path):
    df = pd.read_csv(csv_path)
    # Excel might struggle with some characters if not careful, but pandas handles them well generally.
    df.to_excel(xlsx_path, index=False)
    print(f"Successfully converted {csv_path} to {xlsx_path}")
else:
    print(f"Error: {csv_path} not found")
