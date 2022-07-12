# -*- coding: utf-8 -*- 

import sys
sys.path.append("./")
sys.path.append("../")

import lightgbm as lgb
from utils import data_utils
from sklearn.metrics import roc_auc_score

# 导入数值型样例数据
train_x, test_x, train_y, test_y = data_utils.get_x_y_split(test_rate=0.2)
clf = lgb.LGBMClassifier(objective='binary',
                         boosting_type='gbdt',
                         max_depth=3,
                         n_estimators=1000,
                         subsample=1,
                         colsample_bytree=1)
lgb_model = clf.fit(train_x, train_y, eval_set=[(test_x, test_y)], eval_metric='auc', early_stopping_rounds=30)
auc_score = roc_auc_score(test_y, lgb_model.predict_proba(test_x)[:, 1])
print("LightGBM模型 AUC: ", auc_score)
