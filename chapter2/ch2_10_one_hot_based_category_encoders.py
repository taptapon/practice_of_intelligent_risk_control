# -*- coding: utf-8 -*- 

import sys
sys.path.append("./")
sys.path.append("../")

from utils import data_utils
from category_encoders.one_hot import OneHotEncoder


# 加载数据
german_credit_data = data_utils.get_data()
# 初始化OneHotEncoder类
encoder = OneHotEncoder(cols=['purpose', 'personal.status.and.sex'],
                        handle_unknown='indicator',
                        handle_missing='indicator',
                        use_cat_names=True)
# 转换数据集
result = encoder.fit_transform(german_credit_data)
print("one-hot编码结果: \n", result)