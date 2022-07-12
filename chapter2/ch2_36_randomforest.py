# -*- coding: utf-8 -*- 

import sys
sys.path.append("./")
sys.path.append("../")

from sklearn.ensemble import RandomForestClassifier
from utils import data_utils
from sklearn.metrics import roc_auc_score

# 导入数值型样例数据
train_x, test_x, train_y, test_y = data_utils.get_x_y_split(test_rate=0.2)
clf = RandomForestClassifier(n_estimators=200,
                             criterion='gini',
                             max_depth=6,
                             min_samples_leaf=15,
                             bootstrap=True,
                             oob_score=True,
                             random_state=88)
clf.fit(train_x, train_y)
auc_score = roc_auc_score(test_y, clf.predict_proba(test_x)[:, 1])
print("随机森林模型 AUC: ", auc_score)
