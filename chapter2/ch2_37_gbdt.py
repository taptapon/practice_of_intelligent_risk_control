# -*- coding: utf-8 -*- 

import sys
sys.path.append("./")
sys.path.append("../")

from sklearn.ensemble import GradientBoostingClassifier
from utils import data_utils
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import GradientBoostingClassifier

# 导入数值型样例数据
train_x, test_x, train_y, test_y = data_utils.get_x_y_split(test_rate=0.2)
clf = GradientBoostingClassifier(n_estimators=100,
                                 learning_rate=0.1,
                                 subsample=0.9,
                                 max_depth=4,
                                 min_samples_leaf=20,
                                 random_state=88)
clf.fit(train_x, train_y)
auc_score = roc_auc_score(test_y, clf.predict_proba(test_x)[:, 1])
print("GBDT模型 AUC: ", auc_score)
