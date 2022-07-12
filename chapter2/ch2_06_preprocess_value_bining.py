# -*- coding: utf-8 -*- 

import sys
sys.path.append("./")
sys.path.append("../")

import toad
from toad.plot import bin_plot
from utils import data_utils

german_credit_data = data_utils.get_data()
# 利用toad库等频分箱
# 初始化分箱对象
c = toad.transform.Combiner()
c.fit(german_credit_data[data_utils.x_cols],
      y=german_credit_data[data_utils.label], n_bins=6, method='quantile', empty_separate=True)
# 特征age.in.years分箱结果画图
data_binned = c.transform(german_credit_data, labels=True)
bin_plot(data_binned, x='age.in.years', target=data_utils.label)
