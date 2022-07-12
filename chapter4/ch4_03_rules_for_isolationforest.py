# -*- coding: utf-8 -*-

import sys
sys.path.append("./")
sys.path.append("../")

import numpy as np
from chapter4.ch4_01_rules_for_outliers import rule_discover
from pyod.models.iforest import IForest
from utils import data_utils

# 加载数据
german_credit_data = data_utils.get_data()

# 构造数据集
X = german_credit_data[data_utils.numeric_cols]
y = german_credit_data['creditability']

# 初始化模型
clf = IForest(behaviour='new', bootstrap=False, contamination=0.1, max_features=1.0, max_samples='auto', n_estimators=500, random_state=20, verbose=0)

# 训练模型  
clf.fit(X)

# 预测结果  
german_credit_data['out_pred'] = clf.predict_proba(X)[:, 1]
# 将预测概率大于0.7以上的设为异常值  
german_credit_data['iforest_rule'] = np.where(german_credit_data['out_pred'] > 0.7, 1, 0)

# 效果评估  
rule_iforest = rule_discover(data_df=german_credit_data, var='iforest_rule', target='creditability', rule_term='==1')
print("孤立森林评估结果: \n", rule_iforest.T)
