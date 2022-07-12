# -*- coding: utf-8 -*-

import sys
sys.path.append("./")
sys.path.append("../")

# 文本特征挖掘：word2vec
import numpy as np
import pandas as pd
from utils.text_utils import sentences_prepare
from gensim.models import word2vec


def sent2vec(words, w2v_model):
    """
    转换成句向量
    :param words: 词列表
    :param w2v_model: word2vec模型
    :return:
    """
    if words == '':
        return np.array([0] * model.wv.vector_size)

    vector_list = []
    for w in words:
        try:
            vector_list.append(w2v_model.wv[w])
        except:
            continue
    vector_list = np.array(vector_list)
    v = vector_list.sum(axis=0)
    return v / np.sqrt((v ** 2).sum())


if __name__ == '__main__':
    # 加载语料
    sentences = sentences_prepare()

    # 获取词向量
    model = word2vec.Word2Vec(sentences, size=100, window=5, min_count=2, workers=2)
    fea_vec = pd.DataFrame([sent2vec(x, model).tolist() for x in sentences])
    fea_vec.columns = ['fea_%s' % i for i in range(model.wv.vector_size)]
    print('词向量维度：', fea_vec.shape)
