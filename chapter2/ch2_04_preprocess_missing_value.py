# -*- coding: utf-8 -*- 

import sys
sys.path.append("./")
sys.path.append("../")

import numpy as np
import pandas as pd
from utils import data_utils
from sklearn.impute import SimpleImputer

# 导入数值型样例数据
data = data_utils.get_data()
# 缺失值处理
imp = SimpleImputer(missing_values=np.nan, strategy='mean')
imped_data = imp.fit_transform(data[data_utils.numeric_cols])
imped_df = pd.DataFrame(imped_data, columns=data_utils.numeric_cols)
print("缺失值填充结果: \n", imped_df)
