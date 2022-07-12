# -*- coding: utf-8 -*- 

import sys
sys.path.append("./")
sys.path.append("../")

from utils import data_utils
from category_encoders.target_encoder import TargetEncoder


# 加载数据
german_credit_data = data_utils.get_data()
y = german_credit_data['creditability']
x = german_credit_data[['purpose', 'personal.status.and.sex']]
# 目标编码
enc = TargetEncoder(cols=x.columns)
result = enc.fit_transform(x, y)
print("目标编码结果: \n", result)
