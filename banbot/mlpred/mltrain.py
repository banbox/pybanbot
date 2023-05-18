#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : mltrain.py
# Author: anyongjin
# Date  : 2023/2/14
'''
这里是根据统计特征预测未来一两个单位的趋势。
准确率达到70%。但回测效果较差。

测试过三种特征提取方案：
胜率相似，模型参数对准确率的影响有限，更多来自特征。
效果主要跟pos_thres，neg_thres，min_acc有关
***************  方案一：特征组合的胜率。【目前使用】 *******************
基于features文件的use_policy指标组合，得到每种情况的胜率。然后作为特征直接训练模型。
***************  方案二：特征组合的聚类后分类特征   *********************
基于features文件的use_policy指标组合，得到每种情况的聚类后分类特征，每种特征组合输出一列。然后作为特征直接训练模型。
***************   方案三：最细的技术指标直接作为特征   ******************
基于features的所有涉及到的IndInfo，全部共20+，不做聚类，直接训练。
'''
from typing import Optional

from banbot.compute.utils import *

from banbot.mlpred.mldata import *

save_dir = r'E:\trade\freqtd_data\user_data\models\feaprob4'
if not os.path.isdir(save_dir):
    os.mkdir(save_dir)


train_lgb_params = dict(
    task='train',
    boosting_type='gbdt',
    objective='multiclass',
    metric='multi_logloss',
    num_class=3,
    n_jobs=8,
    n_estimators=900000,
    max_depth=20,
    num_leaves=100,
    min_data_in_leaf=50,
    max_bin=255,
    seed=40,
    learning_rate=0.1,
    feature_fraction=0.8,
    feature_fraction_bynode=0.5,
    bagging_fraction=0.8,
    bagging_freq=1,
    early_stopping_round=50,
    # categorical_feature=','.join([str(v) for v in range(12)])
    # class_weight={0: 1, 1: 2, 2: 1}
)


def start_bayes_opt(opt_rounds=700):
    '''
    针对lightGBM模型进行超参数随机搜索调优
    :param opt_rounds:
    :return:
    '''
    from bayes_opt import BayesianOptimization
    from bayes_opt.logger import JSONLogger
    from bayes_opt.event import Events
    train_data, test_data, col_names = load_data()

    def bayes_opt_eval(max_depth, num_leaves, feature_fraction, feature_fraction_bynode, min_data_in_leaf):
        cur_params = dict(**train_lgb_params)
        cur_params['feature_pre_filter'] = False
        cur_params['n_estimators'] = 200
        cur_params['learning_rate'] = 0.06
        cur_params['max_depth'] = round(max_depth)
        cur_params['num_leaves'] = round(num_leaves)
        cur_params['min_data_in_leaf'] = round(min_data_in_leaf)
        cur_params['feature_fraction'] = max(min(feature_fraction, 1), 0)
        cur_params['feature_fraction_bynode'] = max(min(feature_fraction_bynode, 1), 0)

        callback = [lgb.early_stopping(stopping_rounds=20), lgb.log_evaluation(period=10)]
        cv_res = lgb.cv(cur_params, train_data, seed=None, shuffle=False, stratified=False, callbacks=callback)
        # 这里是损失，越小越好，所以返回负数（优化目标是找最大值）
        return -min(cv_res['multi_logloss-mean'])

    hyper_spaces = dict(
        max_depth=(20, 80),
        num_leaves=(50, 500),
        min_data_in_leaf=(10, 500),
        feature_fraction=(0.6, 1),
        feature_fraction_bynode=(0.15, 0.6)
    )
    opt_model = BayesianOptimization(bayes_opt_eval, hyper_spaces, random_state=0, allow_duplicate_points=True)

    log_path = os.path.join(save_dir, 'bayes_opt_log.json')
    if os.path.isfile(log_path):
        from bayes_opt.util import load_logs
        load_logs(opt_model, logs=[log_path])
    saver = JSONLogger(path=log_path)
    opt_model.subscribe(Events.OPTIMIZATION_STEP, saver)
    opt_model.maximize(n_iter=opt_rounds)

    print(f'best args: {opt_model.max}')
    return opt_model.max


def train_lgb_model(timeframe='5m', pos_thres=40, neg_thres=-30, pred_off: int = 1):
    '''
    :return:
    '''
    plan_1 = dict(
        n_estimators=1000,
        bagging_fraction=0.7,
        bagging_freq=1,
        feature_fraction=0.6867,
        feature_fraction_bynode=0.5,
        lambda_l1=2.35,  # 1.15, 2
        lambda_l2=5,  # 2.55 5
        max_depth=30,  # 51, 13
        min_data_in_leaf=40,
        min_sum_hessian_in_leaf=10,
        num_leaves=150,  # 65,60
        path_smooth=1.8,
        early_stopping_round=70,
        # max_bin=80,
    )

    cur_args = dict(**train_lgb_params)
    cur_args.update(**plan_1)
    cur_args['learning_rate'] = 0.06

    callback = [lgb.early_stopping(stopping_rounds=1000), lgb.log_evaluation(period=10)]

    model_path = os.path.join(save_dir, f'lgb_{timeframe}.txt')
    train_data, test_data, col_names = load_data(timeframe=timeframe, pos_thres=pos_thres, neg_thres=neg_thres,
                                                 pred_off=pred_off)

    # mdl = lgb.LGBMClassifier(**cur_args)
    # train_x, train_y, test_x, test_y, col_names = load_data(False)
    # mdl.fit(train_x, train_y, eval_set=(test_x, test_y), callbacks=callback)
    # mdl.booster_.save_model(model_path)
    mdl = lgb.train(cur_args, train_data, valid_sets=[train_data, test_data], callbacks=callback)
    mdl.save_model(model_path)

    analyze_fea_importances(mdl, col_names)

    test_model(mdl, timeframe, pos_thres, neg_thres, pred_off)


def print_classify_metrics(test_y, pred_y):
    from sklearn import metrics
    acc = metrics.accuracy_score(test_y, pred_y)
    prec = metrics.precision_score(test_y, pred_y, average='weighted')
    recall = metrics.recall_score(test_y, pred_y, average='weighted')
    f1 = metrics.f1_score(test_y, pred_y, average='weighted')
    print(f'acc:    {acc:.3f}')
    print(f'prec:   {prec:.3f}')
    print(f'recall: {recall:.3f}')
    print(f'f1:     {f1:.3f}')


def test_model(model=None, timeframe: Optional[str] = None, pos_thres=40, neg_thres=-30, pred_off: int = 1):
    if not model:
        assert timeframe, '`timeframe` is required to test model'
        model_path = os.path.join(save_dir, f'lgb_{timeframe}.txt')
        model = lgb.Booster(model_file=model_path)
    _, _, test_x, test_y, _ = load_data(False, timeframe, pos_thres, neg_thres, pred_off)
    pred_y = model.predict(test_x).argmax(axis=1)
    view_arr = np.array([test_y, pred_y]).reshape((2, -1)).transpose()
    print(f'  **  all data perf {timeframe} pos:{pos_thres} neg:{neg_thres} off:{pred_off}  **')
    print_classify_metrics(test_y, pred_y)
    print('  **  all pos metrics  **')
    case_ids = np.where(pred_y == 1)[0]
    pred_y = pred_y[case_ids]
    test_y = test_y[case_ids]
    print_classify_metrics(test_y, pred_y)


def analyze_fea_importances(model, cols):
    if isinstance(model, str):
        model = lgb.Booster(model_file=model)
    impts = model.feature_importance()
    feature_imp = pd.DataFrame({'Value': impts, 'Feature': cols})
    fea_imp_data = feature_imp.sort_values(by="Value", ascending=False)
    out_path = os.path.join(save_dir, 'fea_imprt.csv')
    fea_imp_data.to_csv(out_path, index=False)
    print(f'feature importances saved to {out_path}')


if __name__ == '__main__':
    # start_bayes_opt()
    # train_lgb_model('1m', 25, -20, 1)
    # train_lgb_model('5m', 40, -30)
    train_lgb_model('15m', 64, -53, 1)
    # test_model(timeframe='1m', pos_thres=25, neg_thres=-20, pred_off=2)
    bell()
