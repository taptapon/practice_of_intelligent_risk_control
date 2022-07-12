# -*- coding: utf-8 -*- 

import sys
import numpy as np
import pandas as pd
sys.path.append("./")
sys.path.append("../")


# Pickle方式保存和读取模型
def save_model_as_pkl(model, path):
    """
    保存模型到路径path
    :param model: 训练完成的模型
    :param path: 保存的目标路径
    """
    import pickle
    with open(path, 'wb') as f:
        pickle.dump(model, f, protocol=2)


def load_model_from_pkl(path):
    """
    从路径path加载模型
    :param path: 保存的目标路径
    """
    import pickle
    with open(path, 'rb') as f:
        model = pickle.load(f)
    return model

