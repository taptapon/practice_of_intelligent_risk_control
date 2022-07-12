# -*- coding: utf-8 -*- 
import sys
import numpy as np
import pandas as pd
sys.path.append("./")
sys.path.append("../")

def p_to_score(p, pdo, base, odds):
    """ 
    逾期概率转换分数 
    :param p: 逾期概率 
    :param pdo: points double odds. default = 60 
    :param base: base points. default = 600 
    :param odds: odds. default = 1.0/15.0 
    :returns: 模型分数 
    """
    B = pdo / np.log(2)
    A = base + B * np.log(odds)
    score = A - B * np.log(p / (1 - p))
    return round(score, 0)

pros = pd.Series(np.random.rand(100))
pros_score = p_to_score(pros, pdo=60.0, base=600, odds=1.0 / 15.0)
print("随机产生100个概率并转化为score结果: \n", dict(zip(pros, pros_score)))
