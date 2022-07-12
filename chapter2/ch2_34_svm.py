# -*- coding: utf-8 -*- 

import sys
import numpy as np
import pandas as pd
sys.path.append("./")
sys.path.append("../")

from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from utils import data_utils
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from category_encoders.woe import WOEEncoder

# 导入数值型样例数据
train_x, test_x, train_y, test_y = data_utils.get_x_y_split(test_rate=0.2)

# woe特征处理
encoder = WOEEncoder(cols=train_x.columns)
train_x = encoder.fit_transform(train_x, train_y)
test_x = encoder.transform(test_x)

# 线性SVM, Linear Support Vector Classification
line_svm = LinearSVC(penalty='l2',
                     loss='hinge',
                     C=0.2,
                     tol=0.001)
clf = make_pipeline(StandardScaler(), line_svm)
clf.fit(train_x, train_y)
acc_score = accuracy_score(test_y, clf.predict(test_x))
print("线性SVM模型 ACC: ", acc_score)


# 支持核函数的SVM, C-Support Vector Classification
svm = SVC(C=0.2,
          kernel='rbf',
          tol=0.001,
          probability=True)
clf = make_pipeline(StandardScaler(), svm)
clf.fit(train_x, train_y)
auc_score = roc_auc_score(test_y, clf.predict_proba(test_x)[:, 1])
print("支持核函数SVM模型 AUC: ", auc_score)
