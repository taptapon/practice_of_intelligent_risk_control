# -*- coding: utf-8 -*-

import sys
sys.path.append("./")
sys.path.append("../")

# 文本分类算法：朴素贝叶斯
import pandas as pd
from utils.text_utils import sentences_prepare_x_y
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import roc_auc_score


def get_model(x, y):
    # 训练朴素贝叶斯分类器
    clf = GaussianNB()
    bayes_model = clf.fit(x, y)
    return bayes_model


def text_sample_split(texts, y, rate=0.75):
    # 文本向量化
    cv = TfidfVectorizer(binary=True)
    sentence_vec = cv.fit_transform(texts)

    # 划分训练集和测试集
    split_size = int(len(texts) * rate)
    x_train = sentence_vec[:split_size].toarray()
    y_train = y[:split_size]
    x_test = sentence_vec[split_size:].toarray()
    y_test = y[split_size:]
    return x_train, y_train, x_test, y_test


if __name__ == '__main__':
    # 加载语料
    sentences, target = sentences_prepare_x_y()
    print("文本数目: %s" % len(sentences))
    # 训练模型
    x_train, y_train, x_test, y_test = text_sample_split(pd.Series(sentences), pd.Series(target))
    model = get_model(x_train, y_train)
    # 预测
    y_pred = model.predict_proba(x_test)[:, 1]
    auc_score = roc_auc_score(y_test, y_pred)
    print("AUC结果: ", auc_score)
