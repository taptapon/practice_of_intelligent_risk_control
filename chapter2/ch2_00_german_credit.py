# -*- coding: utf-8 -*- 

import scorecardpy as sc

# 加载数据集
german_credit_data = sc.germancredit()
# 打印前5行, 前4列和最后一列
print(german_credit_data.iloc[:5, list(range(-1, 4))])