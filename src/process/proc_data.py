import os
import pandas as pd
import yaml
from collections import Counter


with open('config.yaml', 'r') as file:
    paths = yaml.safe_load(file)['paths']

def convert_data_desc_to_excel():  # 将数据描述文件从parquet格式转换为excel格式
    data_desc_folder = paths['data_desc_folder']
    files = os.listdir(data_desc_folder)
    if any(file.endswith('.xlsx') for file in files):
        return
    print('当前数据描述文件均为parquet格式，正在转换为excel格式以便阅读...')
    for file in os.listdir(data_desc_folder):
        if not file.endswith('.parquet'):
            continue
        df = pd.read_parquet(os.path.join(data_desc_folder, file))
        excel_file = file.replace('.parquet', '.xlsx')
        df.to_excel(os.path.join(data_desc_folder, excel_file))
    print(f'数据描述文件转换完成，已保存为excel格式，路径：{data_desc_folder}')

def load_stations_info(file='data/meteo-swiss-description/stations.parquet'):  # 加载气象站信息
    return pd.read_parquet(file)


def get_dic_vars(file='data/meteo-swiss-description/parameters.parquet'):  # 加载变量字典
    df = pd.read_parquet(file)
    dic_vars = {v: k for k, v in df['param_short'].items()}
    return dic_vars, list(dic_vars.values())


def get_valid_stations(raw_cols, n_vars):  # 保留包含所有变量的站点
    raw_station_cnt = Counter([i[0] for i in raw_cols])
    valid_stations = [k for k, v in raw_station_cnt.items() if v == n_vars]
    valid_cols = [i for i in raw_cols if i[0] in valid_stations]
    return valid_stations, valid_cols


def load_single_data(file, dic_vars):  # 读取单年数据
    df = pd.read_parquet(file)
    valid_stations, valid_cols = get_valid_stations(df.columns.tolist(), len(dic_vars))
    df = df.loc[:, valid_cols]
    df.columns = [f'{s}~{dic_vars[p]}' for s, p in df.columns]
    return df, valid_stations


def merge_data(dic_vars, start_year=2023, end_year=2024):  # 合并多年份、多站点、多变量的数据
    data_folder = paths['data_folder']
    for year in range(start_year, end_year + 1):
        df, valid_stations = load_single_data(os.path.join(data_folder, f'{year}.parquet'), dic_vars)
        if year == start_year:
            df_merge = df.copy()
            merge_valid_stations = valid_stations.copy()
        else:
            assert set(merge_valid_stations) == set(valid_stations), f'Station mismatch between {year - 1} and {year}'
            assert df_merge.columns.tolist() == df.columns.tolist(), f'Columns mismatch between {year - 1} and {year}'
            df_merge = pd.concat([df_merge, df], axis=0)
    datetime_index = pd.date_range(start=df_merge.index.min(), end=df_merge.index.max(), freq='10min')
    assert df_merge.shape[0] == (
                df_merge.index.max() - df_merge.index.min()).total_seconds() / 600 + 1, 'Time index has missing entries'
    df_merge.index = datetime_index
    return df_merge, merge_valid_stations


def save_vars_separately():  # 按变量保存数据
    dic_vars, vars = get_dic_vars()
    df_merge, stations = merge_data(dic_vars, start_year=2023, end_year=2024)
    save_data_folder = paths['save_data_folder']
    for var in vars:
        df_var = df_merge.filter(like=f'~{var}')
        df_var.columns = [col.split('~')[0] for col in df_var.columns]
        df_var.to_parquet(os.path.join(save_data_folder, f'{var}.parquet'))