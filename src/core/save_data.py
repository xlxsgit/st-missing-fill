# path: src/core/save_data.py
from src.process.proc_data import convert_data_desc_to_excel, save_vars_separately


if __name__ == '__main__':
    # 1. 运行数据描述文件转换为Excel格式（如果尚未转换）
    convert_data_desc_to_excel()
    # 2. 按变量保存数据
    save_vars_separately()