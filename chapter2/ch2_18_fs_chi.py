# -*- coding: utf-8 -*- 

import sys
sys.path.append("./")
sys.path.append("../")

from utils import data_utils
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest

# 导入数值型样例数据
all_x_y = data_utils.get_all_x_y()
y = all_x_y.pop(data_utils.label)
# 选择K个最好的特征，返回选择特征后的数据
fs_chi = SelectKBest(chi2, k=5)
fs_chi.fit(all_x_y, y)
x_new = fs_chi.transform(all_x_y)

selected_cols = all_x_y.columns[fs_chi.get_support()].tolist()
print("卡方检验筛选得到%s个特征: \n" % len(selected_cols), selected_cols)
