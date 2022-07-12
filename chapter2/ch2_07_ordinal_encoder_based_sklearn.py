# -*- coding: utf-8 -*- 

import sys
sys.path.append("./")
sys.path.append("../")

import pandas as pd
from utils import data_utils
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder

def label_encode(x):
    """
    将原始分类变量用数字编码
    :param str x: 需要编码的原始变量
    :returns: x_encoded 数字编码后的变量
    """
    le = LabelEncoder()
    x_encoded = le.fit_transform(x.astype(str))
    class_ = le.classes_
    return class_, pd.DataFrame(x_encoded, columns=x.columns)

def ordinal_encode(x):
    """
    将原始分类变量用数字编码
    :param str x: 需要编码的原始变量，shape为[m,n]
    :returns: x_encoded 数字编码后的变量
    """
    enc = OrdinalEncoder()
    x_encoded = enc.fit_transform(x.astype(str))
    return pd.DataFrame(x_encoded).values


def main():
    """
    主函数
    """
    # 加载数据
    german_credit_data = data_utils.get_data()
    # 以特征purpose为例，进行类别编码
    class_, label_encode_x = label_encode(german_credit_data[['purpose']])
    print("特征'purpose'的类别编码结果: \n", label_encode_x)
    print("特征'purpose'编码顺序为: \n", class_)
    # 以特征purpose、credit.history为例，进行类别编码
    ordinal_encode_x = ordinal_encode(german_credit_data[['purpose', 'credit.history']])
    print("特征'purpose'和'credit.history'的类别编码结果: \n", ordinal_encode_x)


if __name__ == "__main__":
    main()

