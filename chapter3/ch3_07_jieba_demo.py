# -*- coding: utf-8 -*-

# 结巴分词使用示例
from utils.text_utils import cut_words

text_demo = "通过资料审核与电话沟通用户审批通过借款金额10000元操作人小明审批时间2020年10月5日 经过电话核实用户确认所有资料均为本人提交提交时间2020年11月5日用户当前未逾期"
segs = cut_words(text_demo)
print("原文: ", text_demo)
print("切词后的结果:", list(segs))
