# -*- coding: utf-8 -*-

import sys
sys.path.append("./")
sys.path.append("../")

import toad
import numpy as np
import pandas as pd
from utils import data_utils
from toad.plot import bin_plot
from matplotlib import pyplot as plt


def cal_iv(x, y):
    """ 
    IV计算函数  
    :param x: feature 
    :param y: label 
    :return: 
    """
    crtab = pd.crosstab(x, y, margins=True)
    crtab.columns = ['good', 'bad', 'total']
    crtab['factor_per'] = crtab['total'] / len(y)
    crtab['bad_per'] = crtab['bad'] / crtab['total']
    crtab['p'] = crtab['bad'] / crtab.loc['All', 'bad']
    crtab['q'] = crtab['good'] / crtab.loc['All', 'good']
    crtab['woe'] = np.log(crtab['p'] / crtab['q'])
    crtab2 = crtab[abs(crtab.woe) != np.inf]

    crtab['IV'] = sum(
        (crtab2['p'] - crtab2['q']) * np.log(crtab2['p'] / crtab2['q']))
    crtab.reset_index(inplace=True)
    crtab['varname'] = crtab.columns[0]
    crtab.rename(columns={crtab.columns[0]: 'var_level'}, inplace=True)
    crtab.var_level = crtab.var_level.apply(str)
    return crtab


german_credit_data = data_utils.get_data()

# 生成分箱初始化对象  
bin_transformer = toad.transform.Combiner()

# 采用等距分箱训练  
bin_transformer.fit(german_credit_data,
                    y='creditability',
                    n_bins=6,
                    method='step',
                    empty_separate=True)

# 分箱数据  
trans_data = bin_transformer.transform(german_credit_data, labels=True)

# 查看Credit amount分箱结果  
bin_plot(trans_data, x='credit.amount', target='creditability')
plt.show()

# 查看Credit amount分箱数据  
cal_iv(trans_data['credit.amount'], trans_data['creditability'])

# 构建单规则
german_credit_data['credit.amount.rule'] = np.where(german_credit_data['credit.amount'] > 12366.0, 1, 0)
