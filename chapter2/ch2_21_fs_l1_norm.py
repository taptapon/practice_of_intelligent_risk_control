# -*- coding: utf-8 -*- 

import sys
import toad
sys.path.append("./")
sys.path.append("../")

from utils import data_utils
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel

# 导入数值型样例数据
all_x_y = data_utils.get_all_x_y()
y = all_x_y.pop(data_utils.label)
x = all_x_y
# 带L1惩罚项的逻辑回归作为基模型的特征选择
LR = LogisticRegression(penalty='l1', C=0.1, solver='liblinear')
sf = SelectFromModel(LR)
x_new = sf.fit_transform(x, y)

selected_cols = x.columns[sf.get_support()].tolist()
print("基于L1范数筛选得到%s个特征: \n" % len(selected_cols), selected_cols)
