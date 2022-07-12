# -*- coding: utf-8 -*- 

import sys
sys.path.append("./")
sys.path.append("../")

import toad
from utils import data_utils

# 加载数据集
german_credit_data = data_utils.get_data()
detect_res = toad.detector.detect(german_credit_data)
# 打印前5行, 前4列

print("前5行, 前4列:")
print(detect_res.iloc[:5, :4])
print("前5行, 第5至9列:")
# 打印前5行, 第5至9列
print(detect_res.iloc[:5, 4:9])
# 打印前5行, 第10至14列
print("前5行, 第10至14列:")
print(detect_res.iloc[:5, 9:])

