# -*- coding: utf-8 -*-

import sys
sys.path.append("./")
sys.path.append("../")

# 使用聚类算法衍生特征
import pandas as pd
from sklearn.cluster import KMeans


def cluster_fea_gen(data, selected_cols, n_clusters):
    """
    使用聚类算法生成特征
    :param data: 用作输入的x,y
    :param selected_cols: 选取用来做聚类的特征列
    :param n_clusters: 聚类类别数
    :return: 聚类算法生成的特征
    """
    x_cluster_feas = data.loc[:, selected_cols]
    # 拟合聚类模型
    clf = KMeans(n_clusters=n_clusters, random_state=1)
    clf.fit(x_cluster_feas)
    return clf


def cluster_fea_apply(data, selected_cols, clf):
    """
    使用聚类算法生成特征
    :param data: 用作输入的x,y
    :param selected_cols: 选取用来做聚类的特征列
    :param clf: 聚类模型
    :return: 聚类算法生成的特征
    """
    # 对原数据表进行类别标记
    data['group'] = clf.predict(data[selected_cols])

    # 距质心距离特征的计算
    centers_df = pd.DataFrame(clf.cluster_centers_)
    centers_df.columns = [x + '_center' for x in selected_cols]

    for item in selected_cols:
        data[item + '_center'] = data['group'].apply(
            lambda x: centers_df.iloc[x, :][item + '_center'])
        data[item + '_distance'] = data[item] - data[item + '_center']

    fea_cols = ['group']
    fea_cols.extend([x + '_distance' for x in selected_cols])

    return data.loc[:, fea_cols]


if __name__ == '__main__':
    # 数据读取
    all_x_y = pd.read_excel('data/order_feas.xlsx')
    all_x_y.set_index('order_no', inplace=True)
    # 取以下几个特征做聚类
    chose_cols = ['orderv1_age', 'orderv1_90_workday_application_amount_mean', 'orderv1_history_order_num',
                  'orderv1_max_overdue_days']
    all_x_y.fillna(0, inplace=True)

    # 生成聚类特征
    model = cluster_fea_gen(all_x_y, chose_cols, n_clusters=5)
    fea_cluster = cluster_fea_apply(all_x_y, chose_cols, model)
    print("使用聚类算法衍生特征数: \n", fea_cluster.shape[1])
    print("使用聚类算法衍生特征结果: \n", fea_cluster.head())
