# -*- coding: utf-8 -*- 

import sys
sys.path.append("./")
sys.path.append("../")

from utils import data_utils
from statsmodels.stats.outliers_influence import variance_inflation_factor

# 导入数值型样例数据
all_x_y = data_utils.get_all_x_y()
x = all_x_y.drop(data_utils.label, axis=1)
vif = [variance_inflation_factor(x.values, ix) for ix in range(x.shape[1])]
print("各特征的vif值计算结果: \n", dict(zip(x.columns, vif)))

# 筛选阈值小于10的特征
selected_cols = x.iloc[:, [f < 10 for f in vif]].columns.tolist()
print("设置vif阈值为10, 筛选得到%s个特征: \n" % len(selected_cols), selected_cols)
