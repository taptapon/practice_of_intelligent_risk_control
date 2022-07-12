import toad
import numpy as np
import pandas as pd
import scorecardpy as sc
import datetime as dt
import pytz
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from dateutil.parser import parse

numeric_cols = ['duration.in.month',
                'credit.amount',
                'age.in.years',
                'present.residence.since',
                'number.of.existing.credits.at.this.bank',
                'installment.rate.in.percentage.of.disposable.income',
                'number.of.people.being.liable.to.provide.maintenance.for']

category_cols = ['status.of.existing.checking.account', 'credit.history',
                 'savings.account.and.bonds', 'present.employment.since',
                 'personal.status.and.sex', 'other.debtors.or.guarantors',
                 'property', 'other.installment.plans', 'housing', 'job',
                 'telephone', 'foreign.worker', 'purpose']

x_cols = numeric_cols + category_cols

label = 'creditability'


def get_data():
    """
    导入原始数据集
    """
    german_credit_data = sc.germancredit()
    german_credit_data[label] = np.where(
        german_credit_data[label] == 'bad', 1, 0)
    # 设置随机数种子, 确保结果可复现
    np.random.seed(0)
    month_list = ['2020-01', '2020-02', '2020-03', '2020-04', '2020-05']
    # 随机分配月份
    german_credit_data['month'] = np.random.choice(
        month_list, len(german_credit_data))
    return german_credit_data


def get_all_x_y(transform_method='minmax'):
    """
    加载数据
    :param transform_method: 数据标准化方式
    """
    german_credit_data = sc.germancredit()
    # 类别型变量转化成数值型索引变量
    encoder = OrdinalEncoder()
    category_result = encoder.fit_transform(german_credit_data[category_cols])
    category_result = pd.DataFrame(data=category_result, columns=category_cols)
    numeric_result = german_credit_data[numeric_cols + [label]].copy()
    # 将标签creditability映射为数值
    numeric_result[label] = np.where(numeric_result[label] == 'bad', 1, 0)
    all_x_y = pd.merge(category_result, numeric_result, left_index=True, right_index=True)
    x_cols = [f for f in all_x_y.columns if f != label]
    # 数据标准化
    if transform_method == 'minmax':
        encoder = MinMaxScaler()
        all_x_y[x_cols] = encoder.fit_transform(all_x_y[x_cols])
    elif transform_method == 'standard':
        encoder = StandardScaler()
        all_x_y[x_cols] = encoder.fit_transform(all_x_y[x_cols])
    elif transform_method == 'origin':
        pass
    return all_x_y


def get_data_after_fs(empty=0.5, iv=0.02, corr=0.7):
    """
    加载特征选择后的数据
    :param empty: 缺失率阈值
    :param iv: iv阈值
    :param corr: 相关性阈值
    """
    all_x_y = get_all_x_y()
    selected_data, drop_lst = toad.selection.select(
        all_x_y, target=label, empty=0.5,
        iv=0.02, corr=0.7, return_drop=True)
    return selected_data


def get_x_y_split(test_rate=0.2, transform_method='minmax'):
    """
    划分训练集和测试集
    :param test_rate: 测试集样本占比
    :param transform_method: 数据标准化方式
    """
    german_credit_data = get_all_x_y(transform_method)
    y = german_credit_data.pop(label)
    x = german_credit_data
    x_train, x_valid, y_train, y_valid = train_test_split(
        x, y, test_size=test_rate, random_state=88)
    return x_train, x_valid, y_train, y_valid


def stamp_to_date(time_stamp, timezone=None):
    """
    时间戳转日期函数
    :param time_stamp:int，时间戳
    :param timezone:string，时区
    :return:datetime
    """
    try:
        if timezone is None:
            stamp_str = str(time_stamp)
            if len(stamp_str) >= 10:
                stamp_str = stamp_str[:10]
            else:
                stamp_str = stamp_str
            time_stamp = int(stamp_str)
            date = dt.datetime.fromtimestamp(time_stamp)
            return date
        else:
            stamp_str = str(time_stamp)
            if len(stamp_str) >= 10:
                stamp_str = stamp_str[:10]
            else:
                stamp_str = stamp_str
            time_stamp = int(stamp_str)
            tz = pytz.timezone(timezone)
            date = dt.datetime.fromtimestamp(time_stamp, tz).strftime('%Y-%m-%d %H:%M:%S')
            date = parse(date)
            return date
    except:
        return parse('2100-01-01')


def date_to_week(date):
    """
    日期转换为星期
    :param date:datetime，string
    :return:int
    """
    try:
        if isinstance(date, str):
            date = parse(date)
        if_weekend = date.weekday()
        return if_weekend
    except:
        return np.nan
