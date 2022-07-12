# -*- coding: utf-8 -*-
"""
使用DeepWalk算法生成特征(可以直接在shell命令窗口中运行deepwalk命令)
"""

import os
import sys
import pandas as pd
sys.path.append("./")
sys.path.append("../")

size = 8
os.system(
    "deepwalk --input data/graph_data/graph_demo.adjlist "
    f"--output data/graph_data/graph_demo.embeddings --representation-size {size}")

fea_vec = pd.read_csv('data/graph_data/graph_demo.embeddings', sep=' ', skiprows=1, index_col=0,
                      names=['fea_%s' % i for i in range(size)]).sort_index()
print('词向量维度：', fea_vec.shape)
print('词向量结果：', fea_vec)
