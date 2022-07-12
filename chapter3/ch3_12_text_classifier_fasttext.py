# -*- coding: utf-8 -*-

import sys
sys.path.append("./")
sys.path.append("../")

# 文本分类算法：fasttext
import fasttext
import pandas as pd
from utils.text_utils import sentences_prepare_with_y
from sklearn.metrics import roc_auc_score


def process_sentences(train_path, test_path, rate=0.8):
    sentences = sentences_prepare_with_y()
    # 预处理之后的数据写入文件train_data.txt
    num = int(len(sentences) * rate)
    train_out = open(train_path, 'w')
    test_out = open(test_path, 'w')
    for sentence in sentences[:num]:
        train_out.write(sentence)
        train_out.write("\n")
    for sentence in sentences[num:]:
        test_out.write(sentence)
        test_out.write("\n")
    print("预处理之后的数据已写入文件train_data.txt, test_data.txt")
    print("train文本数目: %s, test文本数目: %s" % (num, len(sentences) - num))


if __name__ == '__main__':
    # 处理文本数据
    process_sentences(train_path='data/train_data.txt', test_path='data/test_data.txt', rate=0.8)

    # 训练、保存模型
    classifier = fasttext.train_supervised('data/train_data.txt', label='__label__', wordNgrams=3, loss='softmax')
    classifier.save_model('data/fasttext_demo.model')

    # 加载模型
    classifier = fasttext.load_model('data/fasttext_demo.model')
    texts = "系列 票房 不差 口碑 生化危机 资深 玩家 张艳 告诉 玩家 很难 承认 一系列 电影 " \
            "电影 原著 面目全非 女主角 爱丽丝 游戏 角色 电影 渐渐 脱离 游戏 打着 游戏 名号 发展 票房 " \
            "号召力 观众 影响力 电影 系列 具备 剧情 世界观 游戏 生硬 强加 角色 背景 "
    print("当前文本所属类别: ", classifier.predict(texts))

    # 测试集
    test_data = pd.read_csv('data/test_data.txt', header=None)
    texts_new = test_data[1].tolist()
    y_true = [1 if x.strip() == '__label__sports' else 0 for x in test_data[0].tolist()]

    # 预测效果评估
    result = classifier.predict(texts_new)
    y_pre = []
    for i in range(len(result[0])):
        if result[0][i][0] == '__label__sports':
            y_pre.append(result[1][i][0])
        else:
            y_pre.append(1 - result[1][i][0])
    auc_score = roc_auc_score(y_true, y_pre)
    print("测试集AUC为: ", auc_score)
