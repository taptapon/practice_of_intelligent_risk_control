# -*- coding: utf-8 -*- 

import sys
sys.path.append("./")
sys.path.append("../")

from utils import data_utils
from category_encoders.woe import WOEEncoder

# 加载数据
german_credit_data = data_utils.get_data()
y = german_credit_data['creditability']
x = german_credit_data[['purpose', 'personal.status.and.sex']]

# WOE编码
encoder = WOEEncoder(cols=x.columns)
result = encoder.fit_transform(x, y)
print("WOE编码结果: \n", result)
