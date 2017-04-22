#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2017-04-20 20:29:02
# @Author  : Xielong (xielong@zju.edu.cn)
# @Link    : https://github.com/XieLongZju
# @Version : V0.1

# coding: utf-8

from gen_feat import make_train_set
from gen_feat import make_test_set
from sklearn.model_selection import train_test_split
import xgboost as xgb
from gen_feat import report


def xgboost_make_submission():
    train_start_date = '2016-03-10'
    train_end_date = '2016-04-11'

    test_start_date = '2016-04-11'
    test_end_date = '2016-04-16'

    sub_start_date = '2016-03-15'
    sub_end_date = '2016-04-16'

    user_index, training_data, label = make_train_set(
        train_start_date, train_end_date, test_start_date, test_end_date)
    X_train, X_test, y_train, y_test = train_test_split(
        training_data.values, label.values, test_size=0.2, random_state=0)
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    param = {'learning_rate': 0.1, 'n_estimators': 1000, 'max_depth': 3,

             'min_child_weight': 5, 'gamma': 0, 'subsample': 1.0, 'colsample_bytree': 0.8,

             'scale_pos_weight': 1, 'eta': 0.05, 'silent': 1, 'objective': 'binary:logistic'}
