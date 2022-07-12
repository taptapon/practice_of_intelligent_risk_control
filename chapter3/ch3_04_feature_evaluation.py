# -*- coding: utf-8 -*-

import sys
import time
import numpy as np
import pandas as pd
from scipy.stats import variation
sys.path.append("./")
sys.path.append("../")

def cover_ratio(x):
    """
    计算特征覆盖度
    :param x: 特征向量
    :return: cover_ratio, 特征覆盖度
    """
    len_x = len(x)
    len_nan = sum(pd.isnull(x))
    ratio = 1 - len_nan / float(len_x)
    return ratio


def get_datestamps(begin_date, end_date):
    """
    返回[begin_date,end_date]之间日期的时间戳
    :param begin_date: 开始时间
    :param end_date: 结束时间
    :return: [begin_date,end_date]日期的时间戳
    """
    date_arr = [int(time.mktime(x.timetuple())) for x in list(pd.date_range(start=begin_date, end=end_date))]
    return date_arr


if __name__ == '__main__':
    # 模拟生成几个特征
    fea_1 = [-1, -1, -1, 0, 1, 1, 1]  # 特征均值为0
    fea_2 = [1, 1, 1, 1, 1, 1, 1]  # 所有特征均为唯一指
    fea_3 = [1, 2, 3, 4, 5, 6, 7]  # 与时间正相关
    fea_4 = [7, 6, 5, 4, 3, 2, 1]  # 与时间负相关
    fea_5 = [1, 2, 1, 2, np.nan, 2, np.nan]  # 与时间无线性关系

    x_all = pd.DataFrame([fea_1, fea_2, fea_3, fea_4, fea_5]).T
    x_all.columns = ['fea_1', 'fea_2', 'fea_3', 'fea_4', 'fea_5']

    # 特征覆盖度
    fea_cover = x_all.apply(cover_ratio).to_frame('cover_ratio')
    print("特征覆盖度: ", fea_cover)

    # 特征离散度
    fea_variation = variation(fea_2)
    print("特征离散度: ", fea_variation)

    # 计算时间相关性
    x_all['tm_col'] = get_datestamps('2020-10-01', '2020-10-07')

    # 计算三个特征与时间的Peason系数
    fea_time_corr = x_all.loc[:, ['fea_3', 'fea_4', 'fea_5', 'tm_col']].corr().loc[:, ['tm_col']]

    print("构造的特征为: \n", x_all)
    print("特征与时间的Peason系数计算结果: \n", fea_time_corr)
