import pandas as pd


def load_data():
    df_all_data = pd.read_parquet('data/processed/all_data.parquet') # 所有的时序数据

    df_stations = pd.read_csv('data/processed/all_stations.csv') # 所有的站点的信息
    all_stations = df_stations['Station'].tolist() # 所有的站点名称（已经按照cluster排序了）
    all_stations_cluster = df_stations['cluster'].tolist() # 所有的站点对应的cluster

    df_vars = pd.read_csv('data/raw/geo_data/vars.csv') # 所有的变量信息
    vars = {'y': 'wind_speed', 
            'x': [var for var in df_vars['name'] if var != 'wind_speed']}


    # y的处理
    ground_y = df_all_data[[f'{s}~{vars["y"]}' for s in all_stations]].copy()
    missing_rate_y = ground_y.isna().mean().mean()
    print(f'Missing rate in ground_y: {missing_rate_y:.2%}')
    ground_y = ground_y.ffill().bfill()
    ground_y = ground_y.to_numpy().transpose()

    ground_X = df_all_data[[f'{s}~{var}' for s in all_stations for var in vars['x']]]
    missing_rate_X = ground_X.isna().mean().mean()
    print(f'Missing rate in ground_X: {missing_rate_X:.2%}')
    ground_X = ground_X.ffill().bfill()
    ground_X = ground_X.to_numpy()
    
    num_timesteps, num_stations = ground_X.shape[0], len(all_stations) # 时间步长、站点数量，变量数自动补齐即可
    ground_X = ground_X.reshape(num_timesteps, num_stations, -1).transpose(1, 2, 0)


    return ground_X, ground_y, all_stations, all_stations_cluster, vars