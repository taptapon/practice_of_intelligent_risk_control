# -*- coding: utf-8 -*- 

import sys
sys.path.append("./")
sys.path.append("../")

import pandas as pd
from utils import data_utils
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder


def one_hot_encode(x):
    """
    将原始类别变量进行one-hot编码
    :param str x: 需要编码的原始变量
    :returns: x_oht one-hot编码后的变量
    """
    # 首先将类别值进行数值化
    re = OrdinalEncoder()
    x_encoded = re.fit_transform(x.astype(str))
    x_encoded = pd.DataFrame(x_encoded).values
    # 在对数值化后的类别变量进行one-hot编码
    ohe = OneHotEncoder(handle_unknown='ignore')
    x_oht = ohe.fit_transform(x_encoded).toarray()
    return x_oht

def main():
    """
    主函数
    """
    # 加载数据
    german_credit_data = data_utils.get_data()
    # 以特征purpose为例，进行one-hot编码
    label_encode_x = one_hot_encode(german_credit_data[['purpose']])
    label_encode_df = pd.DataFrame(label_encode_x)
    print("特征purpose的one-hot编码结果: \n", label_encode_df)


if __name__ == "__main__":
    main()