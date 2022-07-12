# -*- coding: utf-8 -*-

import sys
sys.path.append("./")
sys.path.append("../")

# 文本特征挖掘：词袋模型示例
import pandas as pd
from utils.text_utils import sentences_prepare
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer


def gen_count_doc_vec(text):
    """
    基于词频统计生成文本的向量表示
    :param text: 输入文本
    :return: 生成的文本向量表示
    """
    cv = CountVectorizer(binary=True)
    document_vec = cv.fit_transform(text)
    return pd.DataFrame(document_vec.toarray())


def gen_tfidf_doc_vec(text):
    """
    基于TfidfVectorizer生成文本向量表示
    :param text: 输入文本
    :return: 生成的文本向量表示
    """
    cv = TfidfVectorizer()
    document_vec = cv.fit_transform(text)
    return pd.DataFrame(document_vec.toarray())


def gen_hash_doc_vec(text, n_features=8):
    """
    基于HashingVectorizer生成文本向量表示
    :param text: 输入文本
    :param n_features: 指定输出特征的维数
    :return: 生成的文本向量表示
    """
    cv = HashingVectorizer(n_features=n_features)
    document_vec = cv.fit_transform(text)
    return pd.DataFrame(document_vec.toarray())


def gen_ngram_doc_vec(text):
    ngram_cv = CountVectorizer(ngram_range=(2, 2), decode_error="ignore",
                               token_pattern=r'\b\w+\b', min_df=1)
    document_vec = ngram_cv.fit_transform(text)
    return pd.DataFrame(document_vec.toarray())


if __name__ == '__main__':
    sentences = sentences_prepare()
    # 词袋模型应用示例
    # 取前三条文本用于展示
    texts = sentences[0:5]
    fea_vec_count = gen_count_doc_vec(texts)
    print("CountVectorizer词向量:")
    print(fea_vec_count)

    fea_vec_tfidf = gen_tfidf_doc_vec(texts)
    print("TfidfVectorizer词向量:")
    print(fea_vec_tfidf)

    fea_vec_hash = gen_hash_doc_vec(texts, n_features=8)
    print("HashingVectorizer词向量:")
    print(fea_vec_hash)

    fea_vec_ngram = gen_ngram_doc_vec(texts)
    print("CountVectorizer词向量(ngram):")
    print(fea_vec_ngram)
