# -*- coding: utf-8 -*-

import os
import random
import jieba
import pandas as pd

# 读取停用词
stopwords = pd.read_csv("data/text_data/stopwords.txt", index_col=False, quoting=3, sep="\t", names=['stopword'],
                        encoding='utf-8')
stopwords = stopwords['stopword'].values


def cut_words(line, words_min=2):
    line_segments = jieba.lcut(line)
    line_segments = filter(lambda x: len(x) >= words_min, line_segments)
    line_segments = filter(lambda x: x not in stopwords, line_segments)
    return list(line_segments)


def load_corpus():
    """
    加载语料库：取自搜狗新闻语料库(https://www.sogou.com/labs/resource/cs.php)
    :return: sentences 语料库
    """
    # 取样后的文本存储
    df_entertainment = pd.read_csv(os.path.join('data/text_data/entertainment_news.csv'))
    df_sports = pd.read_csv(os.path.join('data/text_data/sports_news.csv'))

    entertainment = df_entertainment.content.values.tolist()
    sports = df_sports.content.values.tolist()
    content_file = {'entertainment': entertainment, 'sports': sports}

    return content_file


def sentences_prepare():
    """
    语料库预处理（无标签）
    """
    sentences = []
    content_file = load_corpus()
    for category in content_file.keys():
        for line in content_file[category]:
            try:
                words_list = cut_words(line)
                sentences.append(" ".join(words_list))
            except Exception as e:
                sentences.append("")
                print(e)
                continue
    random.seed(1)
    random.shuffle(sentences)
    return sentences


def sentences_prepare_with_y():
    """
    语料库预处理（含标签）
    """
    sentences = []
    content_file = load_corpus()
    for category in content_file.keys():
        for line in content_file[category]:
            try:
                words_list = cut_words(line)
                sentences.append("__label__" + str(category) + " , " + " ".join(words_list))
            except Exception as e:
                sentences.append("")
                print(line)
                continue
    random.seed(1)
    random.shuffle(sentences)
    return sentences


def sentences_prepare_x_y():
    """
    语料库预处理（语料和标签分别输出）
    """
    cate_dic = {'entertainment': 0, 'sports': 1}
    content_file = load_corpus()
    # 生成训练数据
    sentences = []
    y = []

    for category in content_file.keys():
        # 文本预处理
        for line in content_file[category]:
            try:
                words_list = cut_words(line)
                sentences.append(" ".join(words_list))
                y.append(str(cate_dic.get(category)))
            except Exception as e:
                print(line)
                continue
    sentences_df = pd.DataFrame({'sentences': sentences, 'target': y})
    sentences_df = sentences_df.sample(frac=1, random_state=1)
    return sentences_df.sentences.tolist(), sentences_df.target.tolist()
