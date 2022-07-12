# -*- coding: utf-8 -*- 

import sys
sys.path.append("./")
sys.path.append("../")

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from utils import data_utils
from sklearn.metrics import roc_auc_score
from tensorflow.keras import layers, models, callbacks

# 加载数据集
train_x, test_x, train_y, test_y = data_utils.get_x_y_split(transform_method='standard')

# 数据预处理
train_x = train_x.to_numpy().reshape((train_x.shape[0], train_x.shape[1], 1))
test_x = test_x.to_numpy().reshape((test_x.shape[0], test_x.shape[1], 1))
train_y = train_y.values.reshape((train_y.shape[0], 1))
test_y = test_y.values.reshape((test_y.shape[0], 1))

# 设置随机数种子，保证每次运行结果一致
tf.random.set_seed(1)
callback = callbacks.EarlyStopping(monitor='val_loss', patience=30, mode='min')

# 构建CNN模型结构
model = models.Sequential()
model.add(layers.Conv1D(filters=16, kernel_size=4, activation='relu', input_shape=(train_x.shape[1], 1)))
model.add(layers.Conv1D(filters=8, kernel_size=1, activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dropout(0.3, seed=1))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
# 显示模型的结构
model.summary()
# 设置模型训练参数
model.compile(optimizer='SGD',
              metrics=[tf.metrics.AUC()],
              loss='binary_crossentropy')
# 模型训练
model.fit(train_x, train_y, validation_data=(test_x, test_y), batch_size=16, epochs=240, callbacks=[callback], verbose=2)

# 测试集效果评估
auc_score = roc_auc_score(train_y, model.predict(train_x))
print("训练集AUC", auc_score)
auc_score = roc_auc_score(test_y, model.predict(test_x))
print("测试集AUC", auc_score)
