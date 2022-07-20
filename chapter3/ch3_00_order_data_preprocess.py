# -*- coding: utf-8 -*-

import sys
sys.path.append("./")
sys.path.append("../")

import numpy as np
import pandas as pd
from utils.data_utils import stamp_to_date
from utils.data_utils import date_to_week


def data_preprocess(data, time_col, back_time, dtypes_dict):
    """
    数据预处理函数
    :param data: 待处理的数据
    :param time_col: 回溯依据的时间列名称
    :param back_time: 特征计算时间，datetime.datetime时间格式
    :param dtypes_dict: 指定列字段类型的字典，如{'col1':int}
    :return: 清洗完成的数据
    """
    # 删除time_col为空的行
    data = data[~data[time_col].isin(['nan', np.nan, 'NAN', 'null', 'NULL', 'Null'])]
    # 将时间列的时间戳转为日期格式
    data[time_col] = data[time_col].apply(stamp_to_date)
    # 过滤订单创建时间在back_time之后的数据，避免特征穿越
    data = data[data[time_col] <= back_time]
    # 删除整条缺失的数据
    data.dropna(how='all', inplace=True)
    # 空字符串替换为np.nan
    data.replace('', np.nan, inplace=True)
    # 单个字段缺失填充为0
    data.fillna(0, inplace=True)
    # 去重
    data.drop_duplicates(keep='first', inplace=True)
    # 字段格式转换
    data = data.astype(dtypes_dict)
    # 补充字段
    data['create_time_week'] = data[time_col].apply(date_to_week)
    data['is_weekend'] = data['create_time_week'].apply(lambda x: 1 if x > 5 else 0)

    return data


if __name__ == '__main__':
    # 原始数据读入
    orders = pd.read_excel('data/order_data.xlsx')
    # 取一个用户的历史订单数据
    raw_data = pd.DataFrame(eval(orders['data'][1]))
    # 数据预处理
    data_processed = data_preprocess(raw_data, time_col='create_time',
                                     back_time='2020-12-14',
                                     dtypes_dict={'has_overdue': int,
                                                  'application_term': float,
                                                  'application_amount': float})
    print(data_processed.shape)
