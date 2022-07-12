# -*- coding: utf-8 -*- 

import sys
sys.path.append("./")
sys.path.append("../")

import toad
import pandas as pd
from utils import data_utils


# 导入添加month列的数据
model_data = data_utils.get_data()

x = model_data[data_utils.x_cols]
y = model_data[data_utils.label]

# 分箱
Combiner = toad.transform.Combiner()
x_cat = Combiner.fit_transform(x, y, n_bins=6, method='quantile', empty_separate=True)

# 合并标签和month
x_cat_with_month = x_cat.merge(model_data[['month', 'creditability']], left_index=True, right_index=True)

# 单个特征对比逾期率
feature_col = 'age.in.years'
x_cat_one = x_cat_with_month[[feature_col, 'month', 'creditability']]
feature_var = x_cat_one.pivot_table(index=feature_col,
                                columns='month',
                                values='creditability',
                                aggfunc=['mean'])
print("特征'age.in.years'的按月分箱逾期率统计结果: \n", feature_var)


# 计算特征按月逾期率波动值
def variation_by_month(df, time_col, columns, label_col):
    variation_dict = {}
    for col in columns:
        feature_v = df.pivot_table(
            index=col, columns=time_col, values=label_col, aggfunc=['mean'])
        variation_dict[col] = feature_v.rank().std(axis=1).mean()

    return pd.DataFrame([variation_dict], index=['variation']).T


var_badrate = variation_by_month(x_cat_with_month, 'month', data_utils.x_cols, 'creditability')
print("各特征按月逾期率的标准差: \n", var_badrate)

selected_cols = var_badrate[var_badrate['variation'] < 0.8].index.tolist()
print("设置标准差阈值为0.8, 筛选得到%s个特征: \n" % len(selected_cols), selected_cols)
