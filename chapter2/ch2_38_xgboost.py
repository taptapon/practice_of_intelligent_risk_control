# -*- coding: utf-8 -*- 

import sys
sys.path.append("./")
sys.path.append("../")

import shap
import numpy as np
import pandas as pd
import xgboost as xgb
import bayes_opt as bo
import sklearn.model_selection as sk_ms
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import roc_auc_score
from utils import data_utils
import shap
from chapter2.ch2_31_model_deployment_pickle import save_model_as_pkl


# 确定最优树的颗数
def xgb_cv(param, x, y, num_boost_round=10000):
    dtrain = xgb.DMatrix(x, label=y)
    cv_res = xgb.cv(param, dtrain, num_boost_round=num_boost_round, early_stopping_rounds=30)
    num_boost_round = cv_res.shape[0]
    return num_boost_round

def train_xgb(params, x_train, y_train, x_test=None, y_test=None, num_boost_round=10000, early_stopping_rounds=30, verbose_eval=50):
    """
    训练xgb模型
    """
    dtrain = xgb.DMatrix(x_train, label=y_train)
    if x_test is None:
        num_boost_round = xgb_cv(params, x_train, y_train)
        early_stopping_rounds = None
        eval_sets = ()
    else:
        dtest = xgb.DMatrix(x_test, label=y_test)
        eval_sets = [(dtest, 'test')]
    model = xgb.train(params, dtrain, num_boost_round, evals=eval_sets, early_stopping_rounds=early_stopping_rounds, verbose_eval=verbose_eval)
    return model


def xgboost_grid_search(params_space, x_train, y_train, x_test=None, y_test=None, num_boost_round=10000):
    """
    网格调参, 确定其他参数
    """
    # 设置训练参数
    if x_test is None:
        x_train, x_test, y_train, y_test = sk_ms.train_test_split(x_train, y_train, test_size=0.2, random_state=1)
    score_list = []
    test_params = list(ParameterGrid(params_space))
    for params_try in test_params:
        params_try['eval_metric'] = "auc"
        params_try['random_state'] = 1
        clf_obj = train_xgb(params_try, x_train, y_train, x_test, y_test, num_boost_round=num_boost_round,
                            early_stopping_rounds=30, verbose_eval=0)
        score_list.append(roc_auc_score(y_test, clf_obj.predict(xgb.DMatrix(x_test))))
    result = pd.DataFrame(dict(zip(score_list, test_params))).T
    print(result)
    # 取测试集上效果最好的参数组合
    params = test_params[np.array(score_list).argmax()]
    return params


def xgboost_bayesian_optimization(params_space, x_train, y_train, x_test=None, y_test=None, num_boost_round=10000, nfold=5, init_points=2, n_iter=5, verbose_eval=0, early_stopping_rounds=30):
    """
    贝叶斯调参, 确定其他参数
    """
    # 设置需要调节的参数及效果评价指标
    def xgboost_cv_for_bo(eta, gamma, max_depth, min_child_weight,
                          subsample, colsample_bytree):
        params = {
            'eval_metric': 'auc',
            'booster': 'gbtree',
            'objective': 'binary:logistic',
            'eta': eta,
            'gamma': gamma,
            'max_depth': int(max_depth),
            'min_child_weight': int(min_child_weight),
            'subsample': subsample,
            'colsample_bytree': colsample_bytree,
            'seed': 1
        }
        if x_test is None:
            dtrain = xgb.DMatrix(x_train, label=y_train)
            xgb_cross = xgb.cv(params,
                               dtrain,
                               nfold=nfold,
                               metrics='auc',
                               early_stopping_rounds=early_stopping_rounds,
                               num_boost_round=num_boost_round)
            test_auc = xgb_cross['test-auc-mean'].iloc[-1]
        else:
            clf_obj = train_xgb(params, x_train, y_train, x_test, y_test, num_boost_round=num_boost_round,
                                early_stopping_rounds=early_stopping_rounds, verbose_eval=verbose_eval)
            test_auc = roc_auc_score(y_test, clf_obj.predict(xgb.DMatrix(x_test)))
        return test_auc

    # 指定需要调节参数的取值范围
    xgb_bo_obj = bo.BayesianOptimization(xgboost_cv_for_bo, params_space, random_state=1)
    xgb_bo_obj.maximize(init_points=init_points, n_iter=n_iter)
    best_params = xgb_bo_obj.max['params']
    best_params['max_depth'] = int(best_params['max_depth'])
    best_params['min_child_weight'] = int(best_params['min_child_weight'])
    best_params['eval_metric'] = 'auc'
    best_params['booster'] = 'gbtree'
    best_params['objective'] = 'binary:logistic'
    best_params['seed'] = 1
    return best_params


# 导入数值型样例数据
train_x, test_x, train_y, test_y = data_utils.get_x_y_split(test_rate=0.2)

# 经验参数
exp_params = {
    'eval_metric': 'auc',
    'booster': 'gbtree',
    'objective': 'binary:logistic',
    'eta': 0.1,
    'gamma': 0.01,
    'max_depth': 4,
    'min_child_weight': 1,
    'subsample': 1,
    'colsample_bytree': 1,
    'seed': 1
}
final_xgb_model = train_xgb(exp_params, train_x, train_y, test_x, test_y)
auc_score = roc_auc_score(test_y, final_xgb_model.predict(xgb.DMatrix(test_x)))
print("经验参数模型AUC: ", auc_score)

# 随机搜索调参
choose_tuner = 'bayesian'  # bayesian grid_search
if choose_tuner == 'grid_search':
    params_test = {
        'learning_rate': [0.1, 0.15],
        'gamma': [0.01, 0],
        'max_depth': [4, 3],
        'min_child_weight': [1, 2],
        'subsample': [0.95, 1],
        'colsample_bytree': [1]
    }
    optimal_params = xgboost_grid_search(params_test, train_x, train_y, test_x, test_y)
elif choose_tuner == 'bayesian':
    # 贝叶斯调参
    params_test = {'eta': (0.05, 0.2),
                   'gamma': (0.005, 0.05),
                   'max_depth': (3, 5),
                   'min_child_weight': (0, 3),
                   'subsample': (0.9, 1.0),
                   'colsample_bytree': (0.9, 1.0)}
    optimal_params = xgboost_bayesian_optimization(params_test, train_x, train_y, test_x, test_y, init_points=5, n_iter=8)

print("随机搜索调参最优参数: ", optimal_params)

final_xgb_model = train_xgb(optimal_params, train_x, train_y, test_x, test_y)
auc_score = roc_auc_score(test_y, final_xgb_model.predict(xgb.DMatrix(test_x)))
print("随机搜索调参模型AUC: ", auc_score)

# 保存模型
save_model_as_pkl(final_xgb_model, "./data/xgb_model.pkl")

# SHAP计算
explainer = shap.TreeExplainer(final_xgb_model)
shap_values = explainer.shap_values(train_x)
# SHAP可视化
shap.summary_plot(shap_values, train_x, max_display=5)
