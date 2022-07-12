# -*- coding: utf-8 -*-

import sys
sys.path.append("./")
sys.path.append("../")

# 使用GBDT算法做特征衍生
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import GradientBoostingClassifier


def gbdt_fea_gen(train_data, label, n_estimators=100):
    # 训练GBDT模型
    gbc_model = GradientBoostingClassifier(n_estimators=n_estimators, random_state=1)
    gbc_model.fit(train_data, label)

    # 得到样本元素落在叶节点中的位置
    train_leaf_fea = gbc_model.apply(train_data).reshape(-1, n_estimators)

    # 借用编码将位置信息转化为0，1
    one_hot_encoder = OneHotEncoder()
    one_hot_encoder.fit(train_leaf_fea)
    return gbc_model, one_hot_encoder


def gbdt_fea_appy(data, model, encoder):
    # 获得GBDT特征
    new_feature_train = encoder.transform(model.apply(data).reshape(-1, model.n_estimators)).toarray()
    # new_feas为生成的新特征
    new_fea = pd.DataFrame(new_feature_train)
    new_fea.index = data.index
    new_fea.columns = ['fea_%s' % i for i in range(1, new_fea.shape[1] + 1)]
    return new_fea


if __name__ == '__main__':
    # 读取原始特征数据
    all_x_y = pd.read_excel('data/order_feas.xlsx')
    all_x_y.set_index('order_no', inplace=True)
    # 生成训练数据
    x_train = all_x_y.drop(columns='label')
    x_train.fillna(0, inplace=True)
    y = all_x_y['label']
    # 获取特征
    gbr, encode = gbdt_fea_gen(x_train, y, n_estimators=100)
    new_features = gbdt_fea_appy(x_train, gbr, encode)
    print("使用GBDT算法衍生特征结果: \n", new_features.head())
