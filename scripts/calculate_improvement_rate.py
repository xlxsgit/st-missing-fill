import pandas as pd
import numpy as np
from pathlib import Path

def calculate_improvement():
    csv_path = Path("logs/xlx/summary.csv")
    if not csv_path.exists():
        print(f"Error: {csv_path} not found.")
        return

    df = pd.read_csv(csv_path)
    
    # Standardize column names if necessary (though the CSV seemed to have space in names)
    # The columns are: 模型,缺失模式,缺失率,test rmse etc.
    # Group by missing type (缺失模式) and rate (缺失率)
    
    # Model name mapping for identification
    # STAT-Net is 'mymodel' in the raw logs
    
    # We need 3 patterns * 3 rates * 7 models = 63 improvement rates
    
    patterns = ['mcar', 'seq', 'scm']
    rates = [0.1, 0.3, 0.5]
    
    stat_net_name = 'mymodel'
    other_models = [m for m in df['模型'].unique() if m != stat_net_name]
    
    results = []
    
    for pattern in patterns:
        for rate in rates:
            # Get STAT-Net RMSE for this case
            stat_row = df[(df['模型'] == stat_net_name) & 
                          (df['缺失模式'] == pattern) & 
                          (np.isclose(df['缺失率'], rate))]
            
            if stat_row.empty:
                continue
            
            stat_rmse = stat_row['test rmse'].values[0]
            
            for other in other_models:
                other_row = df[(df['模型'] == other) & 
                               (df['缺失模式'] == pattern) & 
                               (np.isclose(df['缺失率'], rate))]
                
                if not other_row.empty:
                    other_rmse = other_row['test rmse'].values[0]
                    # Improvement rate: (Other - STAT-Net) / Other
                    improvement = (other_rmse - stat_rmse) / other_rmse
                    results.append(improvement)
    
    if not results:
        print("No improvement rates calculated. Please check the data.")
        return

    avg_improvement = np.mean(results)
    count = len(results)
    
    print(f"Total cases analyzed: {count}")
    print(f"Average Improvement Rate of STAT-Net: {avg_improvement:.2%}")
    
    # Save result to a file
    with open("logs/xlx/improvement_report.md", "w") as f:
        f.write("# STAT-Net Performance Improvement Report\n\n")
        f.write(f"- **Total Cases**: {count}\n")
        f.write(f"- **Average Improvement Rate**: {avg_improvement:.2%}\n")

if __name__ == "__main__":
    calculate_improvement()
