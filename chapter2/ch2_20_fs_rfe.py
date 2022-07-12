# -*- coding: utf-8 -*- 

import sys
import toad
sys.path.append("./")
sys.path.append("../")

from utils import data_utils
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression


# 导入数值型样例数据
all_x_y = data_utils.get_all_x_y()
y = all_x_y.pop(data_utils.label)
x = all_x_y
# 递归特征消除法，返回特征选择后的数据
# 参数estimator为基模型
# 参数n_features_to_select为选择的特征个数
rfe = RFE(estimator=LogisticRegression(), n_features_to_select=10)
x_new = rfe.fit_transform(x, y)

selected_cols = x.columns[rfe.get_support()].tolist()
print("通过递归特征消除法筛选得到%s个特征: \n" % len(selected_cols), selected_cols)
