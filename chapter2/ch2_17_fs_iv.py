# -*- coding: utf-8 -*- 

import sys
import toad
sys.path.append("./")
sys.path.append("../")

from utils import data_utils

# 导入数值型样例数据
all_x_y = data_utils.get_all_x_y()
# 利用toad库quality()方法计算IV
var_iv = toad.quality(all_x_y,
                      target='creditability',
                      method='quantile',
                      n_bins=6,
                      iv_only=True)

selected_cols = var_iv[var_iv.iv > 0.1].index.tolist()
print("各特征的iv值计算结果: \n", var_iv)
print("设置iv阈值为0.1, 筛选得到%s个特征: \n" % len(selected_cols), selected_cols)
