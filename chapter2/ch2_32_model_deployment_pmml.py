# -*- coding: utf-8 -*- 

import sys
sys.path.append("./")
sys.path.append("../")

# PMML方式保存和读取模型
from sklearn2pmml import sklearn2pmml, PMMLPipeline
from sklearn_pandas import DataFrameMapper
from pypmml import Model
from xgboost.sklearn import XGBClassifier
from utils import data_utils
from chapter2.ch2_31_model_deployment_pickle import load_model_from_pkl


# 以xgb模型为例，方式1：
# sklearn接口的xgboost，可使用sklearn2pmml生成pmml文件
def save_model_as_pmml(x, y, save_file_path):
    """
    保存模型到路径save_file_path
    :param x: 训练数据特征
    :param y: 训练数据标签
    :param save_file_path: 保存的目标路径
    """
    # 设置pmml的pipeline
    xgb = XGBClassifier(random_state=88)
    mapper = DataFrameMapper([([i], None) for i in x.columns])
    pipeline = PMMLPipeline([('mapper', mapper), ('classifier', xgb)])
    # 模型训练
    pipeline.fit(x, y)
    # 模型结果保存
    sklearn2pmml(pipeline, pmml=save_file_path, with_repr=True)


# PMML格式读取
def load_model_from_pmml(load_file_path):
    """
    从路径load_file_path加载模型
    :param load_file_path: pmml文件路径
    """
    model = Model.fromFile(load_file_path)
    return model


train_x, test_x, train_y, test_y = data_utils.get_x_y_split(test_rate=0.2)
save_model_as_pmml(train_x, train_y, 'data/model/xgb_model.pmml')
model = load_model_from_pmml('data/model/xgb_model.pmml')
pre = model.predict(test_x)
print(pre.head())

# 方式2：
# 原生xgboost.core库生成的XGBoost模型，不能使用sklearn2pmml生成pmml文件，只能通过jpmml-xgboost包，将已有的.bin或.model
# 格式模型文件转为pmml文件

# step1.获取到xgb模型文件
xgb_model = load_model_from_pkl("data/model/xgb_model.pkl")


# step2.生成fmap文件
def create_feature_map(file_name, features):
    outfile = open(file_name, 'w')
    for i, feat in enumerate(features):
        outfile.write('{0}\t{1}\tq\n'.format(i, feat))


create_feature_map('data/model/xgb_model.fmap', xgb_model.feature_names)

# step3.jpmml-xgboost的环境配置及pmml转换：
# step3.1. 下载jpmml-xgboost
# step3.2. 命令行切换到jpmml-xgboost的项目文件夹，输入代码编译
# mvn clean install
# 该步执行完后，jpmml-xgboost的项目文件夹下会多出一个target文件夹，里面包含生成好的jar包
# step3.3. jar包转换为pmml文件
# java -jar jpmml-xgboost_path/target/jpmml-xgboost-executable-1.5-SNAPSHOT.jar  --X-nan-as-missing False
# --model-input data/model/xgb.model --fmap-input data/model/xgb.fmap --target-name target
# --pmml-output data/model/xgb_pmml.pmml
