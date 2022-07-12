# -*- coding: utf-8 -*- 

import sys
import pandas as pd
sys.path.append("./")
sys.path.append("../")

from utils import data_utils
from sklearn.manifold import LocallyLinearEmbedding

# 导入数值型样例数据
all_x_y = data_utils.get_all_x_y()
x = all_x_y.drop(data_utils.label, axis=1)
lle = LocallyLinearEmbedding(n_neighbors=5, n_components=10)
x_new = lle.fit_transform(x)
x_new_df = pd.DataFrame(x_new)
print("利用sklearn进行LLE特征提取结果: \n", x_new_df)
