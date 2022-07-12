# -*- coding: utf-8 -*- 

import sys
sys.path.append("./")
sys.path.append("../")

import pandas as pd
from utils import data_utils
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

# 导入数值型样例数据
data = data_utils.get_data()
# max-min标准化
X_MinMaxScaler = MinMaxScaler().fit_transform(data[data_utils.numeric_cols])
max_min_df = pd.DataFrame(X_MinMaxScaler, columns=data_utils.numeric_cols)
print("max-min标准化结果: \n", max_min_df)
# z-score标准化
X_StandardScaler = StandardScaler().fit_transform(data[data_utils.numeric_cols])
standard_df = pd.DataFrame(X_StandardScaler, columns=data_utils.numeric_cols)
print("z-score标准化结果: \n", standard_df)
