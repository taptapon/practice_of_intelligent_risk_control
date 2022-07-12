# -*- coding: utf-8 -*- 

import sys
sys.path.append("./")
sys.path.append("../")

# https://keras.io

from utils import data_utils
import tensorflow as tf
from sklearn.metrics import roc_auc_score
from tensorflow.keras import layers, models, callbacks

# 加载数据集
train_x, test_x, train_y, test_y = data_utils.get_x_y_split(transform_method='standard')

# 设置随机数种子
tf.random.set_seed(1)
# 设置早停
callback = callbacks.EarlyStopping(monitor='val_loss', patience=30, mode='min')
# 构建DNN模型结构
model = models.Sequential()
model.add(layers.Flatten(input_shape=(train_x.shape[1], 1)))
model.add(layers.Dense(32, activation=tf.nn.relu))
model.add(layers.Dropout(0.3, seed=1))
model.add(layers.Dense(16, activation=tf.nn.relu))
model.add(layers.Dense(1, activation=tf.nn.sigmoid))
# 显示模型的结构
model.summary()
# 设置模型训练参数
model.compile(optimizer='SGD',
              metrics=[tf.metrics.AUC()],
              loss='binary_crossentropy')
# 模型训练
model.fit(train_x, train_y, validation_data=(test_x, test_y), batch_size=16, epochs=240, callbacks=[callback], verbose=2)

# 效果评估
auc_score = roc_auc_score(train_y, model.predict(train_x))
print("训练集AUC", auc_score)
auc_score = roc_auc_score(test_y, model.predict(test_x))
print("测试集AUC", auc_score)
