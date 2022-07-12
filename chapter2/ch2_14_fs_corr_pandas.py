# -*- coding: utf-8 -*- 

import sys
sys.path.append("./")
sys.path.append("../")

from utils import data_utils


# 导入数值型样例数据
all_x_y = data_utils.get_all_x_y()
x = all_x_y.drop(data_utils.label, axis=1)
# 利用pandas库计算相关系数
# pearson相关系数
pearson_corr = x.corr(method='pearson')
print("pandas库计算 pearson相关系数: \n", pearson_corr)
# spearman相关系数
spearman_corr = x.corr(method='spearman')  
print("pandas库计算 spearman相关系数: \n", spearman_corr)
# kendall相关系数
kendall_corr = x.corr(method='kendall')  
print("pandas库计算 kendall相关系数: \n", kendall_corr)
