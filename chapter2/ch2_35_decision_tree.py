# -*- coding: utf-8 -*- 

import sys
sys.path.append("./")
sys.path.append("../")


from sklearn.tree import DecisionTreeClassifier
from utils import data_utils
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeClassifier

# 导入数值型样例数据
train_x, test_x, train_y, test_y = data_utils.get_x_y_split(test_rate=0.2)
# 导入数值型样例数据
clf = DecisionTreeClassifier(criterion='gini',
                             max_depth=8,
                             min_samples_leaf=15,
                             random_state=88)
clf.fit(train_x, train_y)
auc_score = roc_auc_score(test_y, clf.predict_proba(test_x)[:, 1])
print("决策树模型 AUC: ", auc_score)
