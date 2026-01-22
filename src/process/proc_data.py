# path: src/process/proc_data.py
import pandas as pd
import os
import yaml


with open('config.yaml', 'r') as file:
    paths = yaml.safe_load(file)['paths']

def convert_data_desc_to_excel():
    """
    将数据描述文件从parquet格式转换为excel格式，保存到同文件夹，便于阅读
    :return:
    """
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
        df.to_excel(os.path.join(data_desc_folder, excel_file), index=False)
    print(f'数据描述文件转换完成，已保存为excel格式，路径：{data_desc_folder}')

def load_data():
    data_folder = paths['data_folder']
    for file in os.listdir(data_folder):
        df = pd.read_parquet(os.path.join(data_folder, file))
        break
    return df


