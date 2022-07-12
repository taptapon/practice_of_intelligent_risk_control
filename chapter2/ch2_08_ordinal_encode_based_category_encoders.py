# -*- coding: utf-8 -*- 

import sys
sys.path.append("./")
sys.path.append("../")

from utils import data_utils
from category_encoders.ordinal import OrdinalEncoder

# 加载数据
german_credit_data = data_utils.get_data()
# 初始化OrdinalEncoder类
encoder = OrdinalEncoder(cols=['purpose', 'personal.status.and.sex'],
                         handle_unknown='value',
                         handle_missing='value')
# 将 handle_unknown设为"value"，即测试集中的未知特征值将被标记为-1
# 将 handle_missing设为"value"，即测试集中的缺失值将被标记为-2
# 当设为"error"，即报错；当设为"return_nan"，即未知值/缺失值被标记为nan
result = encoder.fit_transform(german_credit_data)
category_mapping = encoder.category_mapping
print("类别编码结果: \n", result)
print("类别编码映射关系: \n", category_mapping)
