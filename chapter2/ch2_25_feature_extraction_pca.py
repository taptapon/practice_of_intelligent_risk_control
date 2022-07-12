# -*- coding: utf-8 -*- 

import sys
import toad
import pandas as pd
sys.path.append("./")
sys.path.append("../")

from utils import data_utils
from sklearn.decomposition import PCA


# 导入数值型样例数据
all_x_y = data_utils.get_all_x_y()
x = all_x_y.drop(data_utils.label, axis=1)
pca = PCA(n_components=0.9)
x_new = pca.fit_transform(x)
x_new_df = pd.DataFrame(x_new)
print("利用sklearn进行PCA特征提取, 保留90%信息后结果: \n", x_new_df)
