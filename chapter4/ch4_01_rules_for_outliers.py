# -*- coding: utf-8 -*-

import sys
sys.path.append("./")
sys.path.append("../")

import pandas as pd
from utils import data_utils

def rule_evaluate(selected_df, total_df, target, rate=0.15, amount=10000):
    """
    :param selected_df: 子特征列表
    :param total_df: 特征宽表
    :param target: 目标变量
    :param rate: 息费（%）
    :param amount: 平均每笔借款金额
    :return:
    """
    # 命中规则的子群体指标统计
    hit_size = selected_df.shape[0]
    hit_bad_size = selected_df[target].sum()
    hit_bad_rate = selected_df[target].mean()
    # 总体指标统计
    total_size = total_df.shape[0]
    total_bad_size = total_df[target].sum()
    total_bad_rate = total_df[target].mean()
    # 命中率
    hit_rate = hit_size / total_size
    # 提升度
    lift = hit_bad_rate / total_bad_rate
    # 收益
    profit = hit_bad_size * amount - (hit_size - hit_bad_size) * rate * amount
    res = [total_size, total_bad_size, total_bad_rate,
           hit_rate, hit_size, hit_bad_size, hit_bad_rate, lift, profit]
    return res


def rule_discover(data_df, var, target, rule_term, rate=0.15, amount=10000):
    """
    :param data_df: 特征宽表
    :param var: 特征名称
    :param target: 目标变量
    :param rule_term: 分位数列表或规则条件
    :param rate: 息费（%）
    :param amount: 平均每笔借款金额
    :return:
    """
    res_list = []
    if rule_term is None:
        rule_term = [0.005, 0.01, 0.02, 0.05, 0.95, 0.98, 0.99, 0.995]
    if isinstance(rule_term, list):
        for q in rule_term:
            threshold = data_df[var].quantile(q).round(2)
            if q < 0.5:
                temp = data_df.query("`{0}` <= @threshold".format(var))
                rule = "<= {0}".format(threshold)
            else:
                temp = data_df.query("`{0}` >= @threshold".format(var))
                rule = ">= {0}".format(threshold)
            res = rule_evaluate(temp, data_df, target, rate, amount)
            res_list.append([var, rule] + res)
    else:
        temp = data_df.query("`{0}` {1}".format(var, rule_term))
        rule = rule_term
        res = rule_evaluate(temp, data_df, target, rate, amount)
        res_list.append([var, rule] + res)
    columns = ['var', 'rule', 'total_size', 'total_bad_size', 'total_bad_rate',
               'hit_rate', 'hit_size', 'hit_bad_size', 'hit_bad_rate', 'lift',
               'profit']
    result_df = pd.DataFrame(res_list, columns=columns)
    return result_df


if __name__ == '__main__':
    # 数据读入
    german_credit_data = data_utils.get_data()
    german_credit_data.loc[german_credit_data.sample(
        frac=0.2, random_state=0).index, 'sample_set'] = 'Train'
    german_credit_data['sample_set'].fillna('OOT', inplace=True)
    # 使用分位数列表构建规则集
    rule_table = rule_discover(data_df=german_credit_data, var='credit.amount',
                               target='creditability',
                               rule_term=[0.005, 0.01, 0.02, 0.05, 0.95, 0.98, 0.99, 0.995])
    print(rule_table)
    # 规则效果评估
    rule_analyze = german_credit_data.groupby('sample_set').apply(
        lambda x: rule_discover(data_df=x, var='credit.amount',
                                target='creditability', rule_term='>12366.0'))
    print(rule_analyze)
