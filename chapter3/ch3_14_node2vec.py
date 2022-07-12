# -*- coding: utf-8 -*-

import sys
sys.path.append("./")
sys.path.append("../")

# 使用Node2Vec算法生成特征
import networkx as nx
import pandas as pd
from node2vec import Node2Vec
import matplotlib.pyplot as plt


def adj_to_graph(adj_table):
    # 根据邻接表生成图G
    graph = nx.Graph()
    # 添加边
    for i in range(0, len(adj_table)):
        node_edgs = adj_table[i]
        for j in range(0, len(node_edgs)):
            graph.add_edge(node_edgs[0], node_edgs[j])
    return graph


def gen_node2vec_fea(graph, dimensions=8):
    # 生成随机游走序列
    node2vec = Node2Vec(graph, dimensions=dimensions, walk_length=30, num_walks=100, workers=4)
    # 向量化
    model = node2vec.fit(window=10, min_count=1, batch_words=4)
    return model.wv.vectors


if __name__ == '__main__':
    # 数据读取
    adj_tbl = []
    with open('data/graph_data/graph_demo.adjlist') as f:
        for line in f.readlines():
            adj_tbl.append(line.replace('\n', '').split(' '))
    G = adj_to_graph(adj_tbl)
    # 使用networkx展示图结构
    nx.draw(G, with_labels=True)
    plt.show()
    feas = gen_node2vec_fea(G, dimensions=8)
    print(pd.DataFrame(feas))
