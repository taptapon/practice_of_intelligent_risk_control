# -*- coding: utf-8 -*- 

import sys
sys.path.append("./")
sys.path.append("../")

from utils import data_utils
from sklearn.model_selection import train_test_split

# 导入添加month列的数据
model_data = data_utils.get_data()
# 选取OOT样本  
oot_set = model_data[model_data['month'] == '2020-05']
# 划分训练集和测试集
train_valid_set = model_data[model_data['month'] != '2020-05']
X = train_valid_set[data_utils.x_cols]
Y = train_valid_set['creditability']
X_train, X_valid, Y_train, Y_valid = train_test_split(X, Y, test_size=0.3, random_state=88)
model_data.loc[oot_set.index, 'sample_set'] = 'oot'
model_data.loc[X_train.index, 'sample_set'] = 'train'
model_data.loc[X_valid.index, 'sample_set'] = 'valid'
