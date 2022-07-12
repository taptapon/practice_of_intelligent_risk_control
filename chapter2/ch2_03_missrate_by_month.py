# -*- coding: utf-8 -*- 

import sys
sys.path.append("./")
sys.path.append("../")
from utils import data_utils

def missrate_by_month(x_with_month, month_col, x_cols):
    """
    按月统计缺失率
    :param x_cols: x变量列名
    :param month_col: 月份时间列名
    :param x_with_month: 包含月份的数据
    :return:
    """
    df = x_with_month.groupby(month_col)[x_cols].apply(lambda x: x.isna().sum() / len(x))
    df = df.T
    df['miss_rate_std'] = df.std(axis=1)
    return df

def main():
    """
    主函数
    """
    # 导入添加month列的数据
    model_data = data_utils.get_data()
    miss_rate_by_month = missrate_by_month(model_data, month_col='month', x_cols=data_utils.numeric_cols)
    print("按月统计缺失率结果: \n", miss_rate_by_month)

if __name__ == "__main__":
    main()

