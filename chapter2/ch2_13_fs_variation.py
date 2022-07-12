# -*- coding: utf-8 -*- 

import sys
sys.path.append("./")
sys.path.append("../")

from utils import data_utils
from scipy.stats import variation

# 导入数值型样例数据
all_x_y = data_utils.get_all_x_y()
x = all_x_y.drop(data_utils.label, axis=1)
# 计算各个特征的变异系数
x_var = variation(x, nan_policy='omit')
result = dict(zip(x.columns ,x_var))
print("变异系数结果: \n", result)