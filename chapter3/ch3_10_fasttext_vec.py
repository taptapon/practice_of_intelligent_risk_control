# -*- coding: utf-8 -*-

import sys
sys.path.append("./")
sys.path.append("../")

# 文本特征挖掘：fasttext
import pandas as pd
from utils.text_utils import sentences_prepare
import fasttext

if __name__ == '__main__':
    # 加载语料
    sentences = sentences_prepare()

    # 预处理过后的文本写入文件unsupervised_train_data
    with open('data/text_data/unsupervised_train_data.txt', 'w') as f:
        for sentence in sentences:
            f.write(sentence)
            f.write('\n')

    # 获取fasttext词向量
    model = fasttext.train_unsupervised('data/text_data/unsupervised_train_data.txt', model='skipgram', dim=8)
    fea_vec = pd.DataFrame([model.get_sentence_vector(x).tolist() for x in sentences])
    fea_vec.columns = ['fea_%s' % i for i in range(model.get_dimension())]
    print('词向量维度：', fea_vec.shape)
