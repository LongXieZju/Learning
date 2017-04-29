#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2017-04-20 20:29:02
# @Author  : Xielong (xielong@zju.edu.cn)
# @Link    : https://github.com/XieLongZju
# @Version : V0.1

# coding: utf-8

import pandas as pd
from gen_feat import make_train_set
from gen_feat import make_test_set
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
import xgboost as xgb
from gen_feat import report
from score import score
from Rule import del_has_buyed


def xgboost_make_submission():
    train_start_date = '2016-03-10'
    train_end_date = '2016-04-11'

    test_start_date = '2016-04-11'
    test_end_date = '2016-04-16'

    sub_start_date = '2016-03-15'
    sub_end_date = '2016-04-16'

    train_path = make_train_set(train_start_date, train_end_date, test_start_date, test_end_date)
    # user_index = pd.read_csv(train_path[0])
    training_data = pd.read_csv(train_path[1])
    label = pd.read_csv(train_path[2])
    
    X_train, X_test, y_train, y_test = train_test_split(training_data.values, label.values, test_size=0.2, random_state=0)
    del training_data
    del label
    dtrain = xgb.DMatrix(X_train, label=y_train)
    del X_train
    del y_train
    dtest = xgb.DMatrix(X_test, label=y_test)
    del X_test
    del y_test
    
    param = {'learning_rate': 0.1, 'n_estimators': 1000, 'max_depth': 4,
             'min_child_weight': 5, 'gamma': 0, 'subsample': 1.0, 'colsample_bytree': 0.8,
             'scale_pos_weight': 1, 'eta': 0.05, 'silent': 1, 'objective': 'binary:logistic'}
    num_round = 200
    param['nthread'] = 4
    
    plst = param.items()
    plst += [('eval_metric','logloss')]
    evallist = [(dtest, 'eval'),(dtrain, 'train')]
    bst = xgb.train(plst, dtrain, num_round, evallist)
    del dtest
    del dtrain
    del plst
    del evallist
    
    test_path = make_test_set(sub_start_date, sub_end_date)
    sub_training_data = pd.read_csv(test_path[1])
    sub_training_data = xgb.DMatrix(sub_training_data.values)
    joblib.dump(bst,'20170429model.pkl')
    
    #bst = joblib.load('20170429model.pkl')
    y = bst.predict(sub_training_data)
    del sub_training_data
    sub_user_index = pd.read_csv(test_path[0])
    sub_user_index['label'] = y
    pred = sub_user_index[sub_user_index['label'] >= 0.03]
    del sub_user_index
    pred = pred.sort_values(['label'])
    pred = pred[['user_id','sku_id']]
    pred = pred.groupby('user_id').first().reset_index()
    pred['user_id'] = pred['user_id'].astype(int)
    pred = del_has_buyed(pred, sub_end_date)
    pred.to_csv('1submission.csv', index=False,index_label=False)
    
def xgboost_cv():
    #用于训练模型
    train_start_date = '2016-03-10'
    train_end_date = '2016-04-11'

    test_start_date = '2016-04-11'
    test_end_date = '2016-04-16'
    
    #用于测试模型，根据官方分数计算得分
    sub_start_date = '2016-02-05'
    sub_end_date = '2016-03-05'
    sub_test_start_date = '2016-03-05'
    sub_test_end_date = '2016-03-10'
    
    train_path = make_train_set(train_start_date, train_end_date, test_start_date, test_end_date)
    
    training_data = pd.read_csv(train_path[1])
    label = pd.read_csv(train_path[2])
    
    X_train, X_test, y_train, y_test = train_test_split(training_data.values, label.values, test_size=0.2, random_state=0)
    del training_data
    del label
    dtrain = xgb.DMatrix(X_train, label=y_train)
    del X_train
    del y_train
    dtest = xgb.DMatrix(X_test, label=y_test)
    del X_test
    del y_test
    
    param = {'learning_rate': 0.1, 'n_estimators': 1000, 'max_depth': 4,
             'min_child_weight': 5, 'gamma': 0, 'subsample': 1.0, 'colsample_bytree': 0.8,
             'scale_pos_weight': 1, 'eta': 0.05, 'silent': 1, 'objective': 'binary:logistic'}
    num_round = 283
    param['nthread'] = 4
    
    plst = param.items()
    plst += [('eval_metric','logloss')]
    evallist = [(dtest, 'eval'),(dtrain, 'train')]
    bst = xgb.train(plst, dtrain, num_round, evallist)
    del dtest
    del dtrain
    del plst
    del evallist
    joblib.dump(bst,'1model.pkl')
    
    bst = joblib.load('1model.pkl')
    test_path = make_train_set(sub_start_date, sub_end_date, sub_test_start_date, sub_test_end_date)
    sub_training_data = pd.read_csv(test_path[1])
    
    sub_training_data = xgb.DMatrix(sub_training_data.values)   
    y = bst.predict(sub_training_data)
    del sub_training_data
    
    sub_user_index = pd.read_csv(test_path[0])
    sub_user_index['label'] = y
    pred = sub_user_index[sub_user_index['label'] >= 0.03]
    pred = pred[['user_id','sku_id']]
    pred = pred.groupby('user_id').first().reset_index()
    #预测出的用户商品对
    pred['user_id'] = pred['user_id'].astype(int)    
    # pred = del_has_buyed(pred, sub_end_date)
    
    
    sub_label = pd.read_csv(test_path[2])
    sub_user_index['label'] = sub_label
    del sub_label
    truedata = sub_user_index[sub_user_index['label'] == 1]
    truedata.drop(['label'], axis=1, inplace=True)
    score(pred, truedata)
    
if __name__ == '__main__':
    #xgboost_cv()
    xgboost_make_submission()