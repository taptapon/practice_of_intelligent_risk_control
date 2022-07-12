# -*- coding: utf-8 -*-

import sys

sys.path.append("./")
sys.path.append("../")

# 根据业务逻辑自动生成用户历史订单特征
import pandas as pd
import numpy as np
from dateutil.parser import parse
from utils.data_utils import stamp_to_date
from chapter3.ch3_00_order_data_preprocess import data_preprocess

func_trans = {'sum': np.sum,
              'mean': np.mean,
              'cnt': np.size,
              'max': np.max,
              'min': np.min,
              'std': np.std,
              }


def get_name_map(type_k, type_v):
    """
    类别变量取值含义，此处直接根据case给出(实际应用中会定制配置)
    :param type_k: 待切分数据
    :param type_v: 具体切分类别
    :return:
    """
    if 'name_map' in globals():
        new_name = name_map.get('%s_%s' % (type_k, type_v), '%s_%s' % (type_k, type_v))
    else:
        new_name = '%s_%s' % (type_k, type_v)
    return new_name


def apply_func(f, *args):
    return f(*args)


def rfm_cut(data, time_col, back_time, type_dict, comp_dict, time_arr, fea_prefix='f'):
    """
    基于RFM思想切分数据，生成特征
    :param DataFrame data: 待切分的数据，时间列为create_time(timestamp)，距今天数列为gap_days
    :param str time_col: 回溯依据的时间列名称
    :param datetime.datetime back_time: 回溯时间点，datetime.datetime时间格式
    :param dict type_dict: 类别变量，以及其对应的取值类别，用于划分数据，类别列名必须在data中
    :param dict comp_dict: 指定计算字段以及对该字段采用的计算方法, 计算变量名必须在data中
    :param list time_arr: 切分时间列表(近N天)
    :param fea_prefix: 特征前缀
    :return dict: 特征
    """
    data[time_col] = data[time_col].apply(stamp_to_date)
    # 业务时间距back_time天数
    data['gap_days'] = data[time_col].apply(lambda x: (back_time - x).days)

    res_feas = {}
    for col_time in time_arr:
        for col_comp in comp_dict.keys():
            for type_k, type_v in type_dict.items():
                # 按类别和时间维度切分,筛选数据
                for item in type_v:
                    data_cut = data[(data['gap_days'] < col_time) & (data[type_k] == item)]
                    for func_k in comp_dict[col_comp]:
                        func_v = func_trans.get(func_k, np.size)
                        # 对筛选出的数据, 在各统计指标上做聚合操作生成特征
                        fea_name = '%s_%s_%s_%s_%s' % (
                            fea_prefix, col_time, get_name_map(type_k, item), col_comp, func_k)
                        if data_cut.empty:
                            res_feas[fea_name] = np.nan
                        else:
                            res_feas[fea_name] = apply_func(func_v, data_cut[col_comp])
    return res_feas


def gen_order_feature_auto(raw_data, time_col, back_time, dtypes_dict, type_dict, comp_dict, time_arr,
                           fea_prefix='f'):
    """
    基于RFM切分，自动生成订单特征
    :param pd.DataFrame raw_data: 原始数据
    :param str time_col: 回溯依据的时间列名称
    :param str back_time: 回溯时间点，字符串格式
    :param dict dtypes_dict: 指定列字段类型的字典，如{'col1':int}
    :param list time_arr: 切分时间列表(近N天)
    :param dict type_dict: 类别变量，以及其对应的取值类别，用于划分数据，类别列名必须在data中
    :param dict comp_dict: 指定计算字段以及对该字段采用的计算方法,计算变量名必须在data中
    :param fea_prefix: 特征前缀
    :return: res_feas 最终生成的特征
    """
    if raw_data.empty:
        return {}
    back_time = parse(str(back_time))

    order_df = data_preprocess(raw_data, time_col=time_col, back_time=back_time, dtypes_dict=dtypes_dict)
    if order_df.empty:
        return {}

    # 特征衍生：使用rfm切分
    res_feas = rfm_cut(order_df, time_col, back_time, type_dict, comp_dict, time_arr, fea_prefix)
    return res_feas


if __name__ == '__main__':
    # 原始数据读入
    orders = pd.read_excel('data/order_data.xlsx')
    # 取一个用户的历史订单数据
    raw_orders = pd.DataFrame(eval(orders['data'][1]))

    # 设置自动特征的参数
    # 类别字段及其取值
    type_dict_param = {
        'has_overdue': [0, 1],
        'is_weekend': [0, 1]
    }
    # 计算字段及其计算函数
    comp_dict_param = {
        'order_no': ['cnt'],
        'application_amount': ['sum', 'mean', 'max', 'min']
    }
    time_cut = [30, 90, 180, 365]

    cols_dtypes_dict = {'has_overdue': int, 'application_term': float, 'application_amount': float}

    # 根据业务逻辑生成用户历史订单特征
    features_auto = gen_order_feature_auto(raw_orders, 'create_time', '2020-12-14', cols_dtypes_dict,
                                           type_dict_param, comp_dict_param, time_cut)
    print("特征维度: ", len(features_auto.keys()))
    print(features_auto)

    # 批量生成特征
    feature_dict = {}
    for i, row in orders.iterrows():
        feature_dict[i] = gen_order_feature_auto(pd.DataFrame(eval(row['data'])), 'create_time', row['back_time'],
                                                 cols_dtypes_dict, type_dict_param, comp_dict_param, time_cut,
                                                 'order_auto')
    feature_df_auto = pd.DataFrame(feature_dict).T
    # feature_df_auto.to_excel('data/features_auto.xlsx', index=True)
