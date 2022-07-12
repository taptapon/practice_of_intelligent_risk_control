# -*- coding: utf-8 -*- 

import sys
import numpy as np
import pandas as pd
sys.path.append("./")
sys.path.append("../")

from utils import data_utils
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from category_encoders.woe import WOEEncoder

# 导入数值型样例数据
train_x, test_x, train_y, test_y = data_utils.get_x_y_split(test_rate=0.2)

# woe特征处理
encoder = WOEEncoder(cols=train_x.columns)
train_x = encoder.fit_transform(train_x, train_y)
test_x = encoder.transform(test_x)

# 利用梯度下降法训练逻辑回归模型
lr = SGDClassifier(loss="log",
                   penalty="l2",
                   learning_rate='optimal',
                   max_iter=100,
                   tol=0.001,
                   epsilon=0.1,
                   random_state=1)
clf = make_pipeline(StandardScaler(), lr)
clf.fit(train_x, train_y)
auc_score = roc_auc_score(test_y, clf.predict_proba(test_x)[:, 1])
print("梯度下降法训练逻辑回归模型 AUC: ", auc_score)

# 利用牛顿法训练逻辑回归模型
lr = LogisticRegression(penalty="l2",
                        solver='lbfgs',
                        max_iter=100,
                        tol=0.001,
                        random_state=1)
clf = make_pipeline(StandardScaler(), lr)
clf.fit(train_x, train_y)
auc_score = roc_auc_score(test_y, clf.predict_proba(test_x)[:, 1])
print("牛顿法训练逻辑回归模型 AUC: ", auc_score)
