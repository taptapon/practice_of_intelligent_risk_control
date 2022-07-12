# -*- coding: utf-8 -*-

import sys
sys.path.append("./")
sys.path.append("../")

# 时间序列特征挖掘
import pandas as pd
from tsfresh.feature_extraction import extract_features

if __name__ == '__main__':
    # 读取数据
    orders = pd.read_excel('data/order_data.xlsx')
    orders_new = []
    for i in range(len(orders)):
        sub_data = pd.DataFrame.from_records(eval(orders['data'][i]))
        sub_data['uid'] = orders['uid'][i]
        orders_new.append(sub_data)
    orders_new_df = pd.concat(orders_new)
    # 数据格式
    orders_new_df['application_amount'] = orders_new_df['application_amount'].astype(float)
    orders_new_df['has_overdue'] = orders_new_df['has_overdue'].astype(float)

    # 调用extract_features生成时间序列特征:order_feas
    order_feas = extract_features(orders_new_df[['uid', 'create_time', 'application_amount', 'has_overdue']], column_id="uid", column_sort="create_time")
    print("时间序列挖掘特征数: \n", order_feas.shape[1])
    print("时间序列特征挖掘结果: \n", order_feas.head())
