# -*- coding: utf-8 -*-

import sys
sys.path.append("./")
sys.path.append("../")

import sklearn.tree as st
import graphviz
from utils import data_utils


def decision_tree_resolve(train_x, train_y, class_names=None, max_depth=3, fig_path=''):
    """
    基于决策树可视化
    :param train_x: data of train
    :param train_y: data of y
    :param class_names:  标签名称
    :param max_depth: 树最大深度
    :param fig_path: 图片路径和名称
    :return:
    """
    if class_names is None:
        class_names = ['good', 'bad']
    clf = st.DecisionTreeClassifier(max_depth=max_depth,
                                    min_samples_leaf=0.01,
                                    min_samples_split=0.01,
                                    criterion='gini',
                                    splitter='best',
                                    max_features=None)
    clf = clf.fit(train_x, train_y)

    # 比例图
    dot_data = st.export_graphviz(clf, out_file=None,
                                  feature_names=train_x.columns.tolist(),
                                  class_names=class_names,
                                  filled=True,
                                  rounded=True,
                                  node_ids=True,
                                  special_characters=True,
                                  proportion=True,
                                  leaves_parallel=True)
    graph = graphviz.Source(dot_data, filename=fig_path)
    return graph


# 加载数据
german_credit_data = data_utils.get_data()

# 构造数据集
X = german_credit_data[data_utils.numeric_cols].copy()
y = german_credit_data['creditability']

graph = decision_tree_resolve(X, y, fig_path='data/tree')
graph.view()

# 转化为规则
X['node_5'] = X.apply(lambda x: 1 if x['duration.in.month'] <= 34.5 and x['credit.amount'] > 8630.5 else 0, axis=1)
X['node_9'] = X.apply(
    lambda x: 1 if x['duration.in.month'] > 34.5 and x['age.in.years'] <= 29.5 and x['credit.amount'] > 4100.0 else 0,
    axis=1)
X['node_12'] = X.apply(lambda x: 1 if x['duration.in.month'] > 34.5 and x['age.in.years'] > 56.5 else 0, axis=1)
