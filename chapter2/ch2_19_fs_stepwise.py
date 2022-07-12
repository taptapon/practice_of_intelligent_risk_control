# -*- coding: utf-8 -*- 

import sys
import toad
sys.path.append("./")
sys.path.append("../")

from utils import data_utils

# 导入数值型样例数据
all_x_y = data_utils.get_all_x_y()
final_data = toad.selection.stepwise(all_x_y,
                                     target=data_utils.label,
                                     estimator='lr',
                                     direction='both',
                                     criterion='aic',
                                     return_drop=False)
selected_cols = final_data.columns
print("通过stepwise筛选得到%s个特征: \n" % len(selected_cols), selected_cols)
