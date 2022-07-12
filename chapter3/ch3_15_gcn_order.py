# -*- coding: utf-8 -*-

import sys
sys.path.append("./")
sys.path.append("../")

# GCN关系网络节点预测
import pickle
import os
import itertools
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.optim as optim
import matplotlib.pyplot as plt
from collections import namedtuple

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
cpu_type = "cuda" if torch.cuda.is_available() else "cpu"


def numpy_to_tensor(x):
    return torch.from_numpy(x).to(cpu_type)


def build_adjacency(adj_dict):
    """
    根据邻接表创建邻接矩阵
    :param adj_dict: 输入的邻接表
    :return: 邻接矩阵
    """
    edge_index = []
    node_counts = len(adj_dict)
    for src, dst in adj_dict.items():
        edge_index.extend([src, v] for v in dst)
        edge_index.extend([v, src] for v in dst)
    # 去重
    edge_index = list(k for k, _ in itertools.groupby(sorted(edge_index)))
    edge_index = np.asarray(edge_index)
    # 构建邻接矩阵，相接的节点值为1
    adjacency = sp.coo_matrix((np.ones(len(edge_index)),
                               (edge_index[:, 0], edge_index[:, 1])),
                              shape=(node_counts, node_counts), dtype="double")
    return adjacency


def read_data(path_of_data):
    """
    数据读取
    :param path_of_data: 文件路径
    :return:
    """
    out = pickle.load(open(path_of_data, "rb"), encoding="latin1")
    out = out.toarray() if hasattr(out, "toarray") else out
    return out


def data_preprocess():
    print("Start data preprocess.")
    filenames = ["order.{}".format(name) for name in ['x', 'y', 'graph']]
    # 图有2000个节点，每个节点有104维特征，y值为0或1，graph用字典表示，字典key为节点编号，value为关联的节点编号list
    root_path = 'data/graph_data'
    x, y, graph = [read_data(os.path.join(root_path, name)) for name in filenames]

    # 划分train，validation和test节点编号
    train_index = list(range(0, 700))
    val_index = list(range(700, 1000))
    test_index = list(range(1000, 2000))

    num_nodes = x.shape[0]
    train_mask = np.zeros(num_nodes, dtype=bool)
    val_mask = np.zeros(num_nodes, dtype=bool)
    test_mask = np.zeros(num_nodes, dtype=bool)

    train_mask[train_index] = True
    val_mask[val_index] = True
    test_mask[test_index] = True

    adjacency = build_adjacency(graph)
    print("特征维度: ", x.shape)
    print("标签长度: ", y.shape)
    print("邻接矩阵维度: ", adjacency.shape)
    # 构建带字段名的元组
    Data = namedtuple('Data', ['x', 'y', 'adjacency',
                               'train_mask', 'val_mask', 'test_mask'])
    return Data(x=x, y=y, adjacency=adjacency,
                train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)


def adj_norm(adjacency):
    """
    正则化：公式L=D^-0.5 * (A+I) * D^-0.5
    :param torch.sparse.FloatTensor adjacency:
    :return:
    """
    adjacency += sp.eye(adjacency.shape[0])
    degree = np.array(adjacency.sum(1))
    d_hat = sp.diags(np.power(degree, -0.5).flatten())
    return d_hat.dot(adjacency).dot(d_hat).tocoo()


class GraphConv(nn.Module):
    def __init__(self, input_dim, output_dim, use_bias=True):
        """
        # 图卷积层定义
        :param int input_dim: 输入特征维度
        :param int output_dim: 输出特征维度
        :param bool use_bias: 偏置
        :return:
        """
        super(GraphConv, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_bias = use_bias
        self.weight = nn.Parameter(torch.Tensor(input_dim, output_dim))
        if self.use_bias:
            self.bias = nn.Parameter(torch.Tensor(output_dim))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight)
        if self.use_bias:
            init.zeros_(self.bias)

    def forward(self, adjacency, fea_input):
        """
        :param torch.sparse.FloatTensor adjacency : 邻接矩阵
        :param torch.Tensor fea_input: 输入特征
        :return:
        """
        support = torch.mm(fea_input, self.weight)
        output = torch.sparse.mm(adjacency, support)
        if self.use_bias:
            output += self.bias
        return output


class GcnNet(nn.Module):
    def __init__(self, input_dim):
        """
        模型定义
        :param int input_dim: 输入特征维度
        """
        super(GcnNet, self).__init__()
        self.gcn1 = GraphConv(input_dim, 16)
        self.gcn2 = GraphConv(16, 2)

    def forward(self, adjacency, feature):
        h = F.relu(self.gcn1(adjacency, feature))
        lg = self.gcn2(adjacency, h)
        return lg


def model_predict(model, tensor, tensor_adj, mask):
    model.eval()
    with torch.no_grad():
        lg = model(tensor_adj, tensor)
        lg_mask = lg[mask]
        y_pred = lg_mask.max(1)[1]
    return y_pred


def cal_accuracy(y_true, y_pred):
    accuracy = torch.eq(y_pred, y_true).double().mean()
    return accuracy


def model_train(tensor_x, tensor_y, tensor_adjacency, train_mask, val_mask, epochs, learning_rate,
                weight_decay):
    # 模型定义：Model, Loss, Optimizer
    model = GcnNet(tensor_x.shape[1]).to(cpu_type)
    optimizer = optim.Adam(model.parameters(),
                           lr=learning_rate,
                           weight_decay=weight_decay)

    loss_list = []
    test_accuracy_list = []
    model.train()
    train_y = tensor_y[train_mask].long()

    for epoch in range(epochs):
        # 前向传播
        lg = model(tensor_adjacency, tensor_x)
        train_mask_logits = lg[train_mask]
        loss = nn.CrossEntropyLoss().to(cpu_type)(train_mask_logits, train_y)
        optimizer.zero_grad()
        # 反向传播
        loss.backward()
        optimizer.step()
        # 准确率
        train_accuracy = cal_accuracy(tensor_y[train_mask],
                                      model_predict(model, tensor_x, tensor_adjacency, train_mask))
        test_accuracy = cal_accuracy(tensor_y[val_mask],
                                     model_predict(model, tensor_x, tensor_adjacency, val_mask))

        loss_list.append(loss.item())
        test_accuracy_list.append(test_accuracy.item())
        if epoch % 10 == 1:
            print("epoch {:04d}: loss {:.4f}, train accuracy {:.4}, test accuracy {:.4f}".format(
                epoch, loss.item(), train_accuracy.item(), test_accuracy.item()))
    return model, loss_list, test_accuracy_list


def plot_loss_with_acc(loss_history, val_acc_history):
    fig = plt.figure()
    # 坐标系ax1画曲线1
    ax1 = fig.add_subplot(111)  # 指的是将plot界面分成1行1列，此子图占据从左到右从上到下的1位置
    ax1.plot(range(len(loss_history)), loss_history,
             c=np.array([255, 71, 90]) / 255.)  # c为颜色
    plt.ylabel('Loss')

    # 坐标系ax2画曲线2
    ax2 = fig.add_subplot(111, sharex=ax1, frameon=False)  # 其本质就是添加坐标系，设置共享ax1的x轴，ax2背景透明
    ax2.plot(range(len(val_acc_history)), val_acc_history,
             c=np.array([79, 179, 255]) / 255.)
    ax2.yaxis.tick_right()  # 开启右边的y坐标

    ax2.yaxis.set_label_position("right")
    plt.ylabel('ValAcc')

    plt.xlabel('Epoch')
    plt.title('Training Loss & Validation Accuracy')
    plt.show()


if __name__ == '__main__':
    # 数据预处理
    dataset = data_preprocess()

    # x、y规范化
    node_feature = (dataset.x - dataset.x.mean()) / dataset.x.std()
    tensor_x_all = numpy_to_tensor(node_feature).to(torch.float32)
    tensor_y_all = numpy_to_tensor(dataset.y)

    tensor_train_mask = numpy_to_tensor(dataset.train_mask)
    tensor_val_mask = numpy_to_tensor(dataset.val_mask)
    tensor_test_mask = numpy_to_tensor(dataset.test_mask)

    # 邻接矩阵规范化
    normed_adj = adj_norm(dataset.adjacency)

    indices = torch.from_numpy(np.asarray([normed_adj.row,
                                           normed_adj.col]).astype('int64')).long()
    values = torch.from_numpy(normed_adj.data.astype(np.float32))

    tensor_adjacency_all = torch.sparse.FloatTensor(indices, values,
                                                    (node_feature.shape[0], node_feature.shape[0])).to(cpu_type)

    # 训练模型并做预测
    gcn_model, loss_arr, test_accuracy_arr = model_train(tensor_x_all, tensor_y_all, tensor_adjacency_all,
                                                         tensor_train_mask,
                                                         tensor_val_mask, epochs=300,
                                                         learning_rate=0.04, weight_decay=5e-4)
    y_predict = model_predict(gcn_model, tensor_x_all, tensor_adjacency_all, tensor_test_mask)
    test_acc = cal_accuracy(tensor_y_all[tensor_test_mask], y_predict)
    print(test_acc.item())

    plot_loss_with_acc(loss_arr, test_accuracy_arr)
