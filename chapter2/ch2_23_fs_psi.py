# -*- coding: utf-8 -*- 

import sys
import toad
sys.path.append("./")
sys.path.append("../")

from utils import data_utils

# 加载数据
all_x_y = data_utils.get_all_x_y()
# 定义分箱方法
Combiner = toad.transform.Combiner()
Combiner.fit(all_x_y,
             y=data_utils.label,
             n_bins=6,
             method='quantile',
             empty_separate=True)
# 计算psi
var_psi = toad.metrics.PSI(all_x_y.iloc[:500, :],
                           all_x_y.iloc[500:, :],
                           combiner=Combiner)
var_psi_df = var_psi.to_frame(name='psi')

selected_cols = var_psi[var_psi_df.psi < 0.1].index.tolist()
print("各特征的psi值计算结果: \n", var_psi_df)
print("设置psi阈值为0.1, 筛选得到%s个特征: \n" % len(selected_cols), selected_cols)
