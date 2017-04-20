#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2017-03-30 05:42:31
# @Author  : Xielong (xielong@zju.edu.cn)
# @Link    : https://github.com/XieLongZju
# @Version : V0.1

# coding: utf-8

import numpy as np
import pandas as pd
import time
import pickle
import os
import math
from datatime import datetime
from datetime import timedelta

#%%
# Setting
path = 'F:/Competition/JData/DataSet/'
user_path = path + 'JData_User.csv'
comment_path = path + 'JData_Comment.csv'
product_path = path + 'JData_Product.csv'
action_2_path = path + 'JData_Action_201602.csv'
action_3_path = path + 'JData_Action_201603.csv'
action_4_path = path + 'JData_Action_201604.csv'

comment_date = ["2016-02-01", "2016-02-08", "2016-03-15", "2016-02-22",
                "2016-02-19", "2016-03-07", "2016-03-14", "2016-03-21",
                "2016-03-28", "2016-04-04", "2016-04-11", "2016-04-15"]

# 注册基准时间，京东多媒体网开通时间
baseTime = datetime.datetime.strptime('2004-01-01', "%Y-%m-%d")

#%%
# Analysis
# def convert_temp_type(type):
#     if type == 4:
#         return 1
#     else:
#         return 0

# userAll = pd.read_csv(user_path, encoding='gbk',
#                       parse_dates=['user_reg_tm'])
# userAll.loc[userAll["age"] == "-1", "age"] = 0
# userAll.loc[userAll["age"] == u'56\u5c81\u4ee5\u4e0a', "age"] = 55
# userAll.loc[userAll["age"] == u'46-55\u5c81', "age"] = 50
# userAll.loc[userAll["age"] == u'36-45\u5c81', "age"] = 40
# userAll.loc[userAll["age"] == u'26-35\u5c81', "age"] = 30
# userAll.loc[userAll["age"] == u'16-25\u5c81', "age"] = 20
# userAll.loc[userAll["age"] == u'15\u5c81\u4ee5\u4e0b', "age"] = 15
# userAll.loc[userAll["age"] == u'56\u5c81\u4ee5\u4e0a', "age"] = 55
# userAll.drop([34072, 38905, 67704], axis=0, inplace=True)
# #userAll['user_reg_dt'] = user['user_reg_dt'].astype('str')
# userAll['user_reg_dt'] = userAll['user_reg_dt'].apply(lambda x: (
#     datetime.datetime.strptime(x, "%Y-%m-%d") - baseTime).days)
# userAction_4 = pd.read_csv(action_4_path, encoding='gbk')
# userAction_4['user_id'] = userAction_4['user_id'].astype('int')
# temp = pd.DataFrame()
# temp['user_id'] = userAction_4['user_id']
# temp['type'] = userAction_4['type']
# temp['type'] = temp['type'].map(convert_temp_type)
# del userAction_4
# temp = temp.groupby(['user_id']).sum()['type'].reset_index()
# user_action = pd.merge(userAll, temp, on='user_id', how='left')
# age = user_action.groupby(['age']).sum()['type'].reset_index()
# sex = user_action.groupby(['sex']).sum()['type'].reset_index()
# print temp.head()

#%%


def convertUserAge(age):
    # 根据购买力对年龄段进行排序，数值越大代表购买力越强
    if age == u'15ËêÒÔÏÂ':
        return 1
    elif age == u'56ËêÒÔÉÏ':
        return 2
    elif age == u'46-55Ëê':
        return 3
    elif age == u'16-25Ëê':
        return 4
    elif age == u'-1':
        return 5
    elif age == u'36-45Ëê':
        return 6
    elif age == u'26-35Ëê':
        return 7


def convertUserRegTM(age):
    # 处理用户注册时间异常数据
    age1 = age.loc[age['age'] == 1]
    age2 = age.loc[age['age'] == 2]
    age3 = age.loc[age['age'] == 3]
    age4 = age.loc[age['age'] == 4]
    age5 = age.loc[age['age'] == 5]
    age6 = age.loc[age['age'] == 6]
    age7 = age.loc[age['age'] == 7]
    age1.fillna(age1['user_reg_tm'].mean(), inplace=True)
    age2.fillna(age2['user_reg_tm'].mean(), inplace=True)
    age3.fillna(age3['user_reg_tm'].mean(), inplace=True)
    age4.fillna(age4['user_reg_tm'].mean(), inplace=True)
    age5.fillna(age5['user_reg_tm'].mean(), inplace=True)
    age6.fillna(age6['user_reg_tm'].mean(), inplace=True)
    age7.fillna(age7['user_reg_tm'].mean(), inplace=True)
    frames = [age1, age2, age3, age4, age5, age6, age7]
    result = pd.concat(frames)
    return result


def get_basic_user_feat():
    # 对用户特征进行处理
    dump_path = '../cache/basic_user.pkl'
    if os.path.exists(dump_path):
        user = pickle.load(open(dump_path))
    else:
        user = pd.read_csv(user_path, encoding='gbk')
        # 抛弃掉这三个用户
        user.drop([34072, 38905, 67704], axis=0, inplace=True)
        # 对年龄段编码，性别不处理（性别与购买力也有关系）
        user['age'] = user['age'].map(convertUserAge)
        user['user_reg_tm'] = user['user_reg_tm'].astype('str')
        user['user_reg_tm'] = user['user_reg_tm'].apply(lambda x: (
            datetime.datetime.strptime(x, "%Y-%m-%d") - baseTime).days)
        user.loc[user['user_reg_tm'] < 0, "user_reg_tm"] = np.nan
        user = convertUserRegTM(user)
        user = user.astype('int')
        user.sort_values('user_id', inplace=True)
        pickle.dump(user, open(dump_path, 'w'))
    return user


def get_basic_product_feat():
    # 获得产品特征
    dump_path = '../cache/basic_product.pkl'
    if os.path.exists(dump_path):
        product = pickle.load(open(dump_path))
    else:
        product = pd.read_csv(product_path, encoding='gbk')
        attr1_df = pd.get_dummies(product["a1"], prefix="a1")
        attr2_df = pd.get_dummies(product["a2"], prefix="a2")
        attr3_df = pd.get_dummies(product["a3"], prefix="a3")
        frames = ['sku_id', 'cate', 'brand']
        product = pd.concat(
            [product[frames], attr1_df, attr2_df, attr3_df], axis=1)
        pickle.dump(product, open(dump_path, 'w'))
    return product


def get_action_2():
    # 获取用户2月份行为
    actAll = pd.read_csv(action_2_path, iterator=True)
    loop = True
    chunksize = 100000
    chunks = []
    while loop:
        try:
            chunk = actAll.get_chunk(chunksize)
            chunks.append(chunk)
        except StopIteration:
            loop = False
            print "get_action_2 Iteration is stopped!"
    action = pd.concat(chunks, ignore_index=True)
    return action


def get_action_3():
    # 获取用户3月份行为
    actAll = pd.read_csv(action_3_path, iterator=True)
    loop = True
    chunksize = 100000
    chunks = []
    while loop:
        try:
            chunk = actAll.get_chunk(chunksize)
            chunks.append(chunk)
        except StopIteration:
            loop = False
            print "get_action_3 Iteration is stopped!"
    action = pd.concat(chunks, ignore_index=True)
    return action


def get_action_4():
    # 获取用户4月份行为
    actAll = pd.read_csv(action_4_path, iterator=True)
    loop = True
    chunksize = 100000
    chunks = []
    while loop:
        try:
            chunk = actAll.get_chunk(chunksize)
            chunks.append(chunk)
        except StopIteration:
            loop = False
            print "get_action_4 Iteration is stopped!"
    action = pd.concat(chunks, ignore_index=True)
    return action


def get_actions(start_date, end_date):
    # 获取用户所有月份行为
    dump_path = '../cache/all_action_%s_%s.pkl' % (start_date, end_date)
    if os.path.exists(dump_path):
        actions = pickle.load(open(dump_path))
    else:
        actions3 = get_action_3()
        actions = get_action_4()
        actions = pd.concat([actions3, actions])
        del actions3
        actions2 = get_action_2()
        actions = pd.concat([actions2, actions])
        del actions2
        actions = actions[(actions.time >= start_date)
                          & (actions.time < end_date)]
        pickle.dump(actions, open(dump_path, 'w'))
    return actions


def get_action_feat(start_date, end_date):
    # 获得用户特征，cate未进行one-hot编码
    dump_path = '../cache/action_accumulate_%s_%s.pkl' % (start_date, end_date)
    if os.path.exists(open(dump_path)):
        actions = pickle.load(open(dump_path))
    else:
        actions = get_actions(start_date, end_date)
        actions.drop(['time', 'model_id'], axis=1, inplace=True)
        user_action_type = pd.get_dummies(
            actions['type'], prefix='%s-%s-action' % (start_date, end_date))
        actions = pd.concat([actions, user_action_type], axis=1)
        del user_action_type
        del actions['type']
        actions = actions.groupby(
            ['user_id', 'sku_id', 'cate', 'brand'], as_index=False).sum()
        pickle.dump(actions, open(dump_path, 'w'))
    return actions


def get_accumulate_action_feat(start_date, end_date):
    # 获得用户累积特征
    # 参考选手没有用这一块，可以自己改进下
    dump_path = '../cache/action_accumulate_%s_%s.pkl' % (start_date, end_date)
    if os.path.exists(open(dump_path)):
        actions = pickle.load(open(dump_path))
    else:
        actions = get_actions(start_date, end_date)
        user_action_type = pd.get_dummies(actions['type'], prefix='action')
        actions = pd.concat([actions, user_action_type], axis=1)
        del user_action_type
        # 近期行为按时间衰减
        actions['weights'] = actions['time'].map(lambda x: datetime.strptime(
            end_date, '%Y-%m-%d') - datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
        # actions['weights'] = time.strptime(end_date,'%Y-%m-%d') -
        # actions['datetime']
        actions['weights'] = actions['weights'].map(
            lambda x: math.exp(-x.days))
        print actions.head(10)
        actions['action_1'] = actions['action_1'] * actions['weights']
        actions['action_2'] = actions['action_2'] * actions['weights']
        actions['action_3'] = actions['action_3'] * actions['weights']
        actions['action_4'] = actions['action_4'] * actions['weights']
        actions['action_5'] = actions['action_5'] * actions['weights']
        actions['action_6'] = actions['action_6'] * actions['weights']
        actions.drop(['model_id', 'type', 'time', 'datetime',
                     'weights'], axis=1, inplace=True)
        actions = actions.groupby(
            ['user_id', 'sku_id', 'cate', 'brand'], as_index=False).sum()
        pickle.dump(actions, open(dump_path, 'w'))
    return actions


def get_comments(start_date, end_date):
    # 获取商品评论信息
    comment_date_end = end_date
    comment_date_begin = comment_date[0]
    for date in reversed(comment_date):
        if date < comment_date_end:
            comment_date_begin = date
            break
    commentAll = pd.read_csv(comment_path, iterator=True)
    loop = True
    chunksize = 100000
    chunks = []
    while loop:
        try:
            chunk = commentAll.get_chunk(chunksize)
            chunks.append(chunk)
        except StopIteration:
            loop = False
            print "get_comments Iteration is stopped!"
    comments = pd.concat(chunks, ignore_index=True)
    comments = comments[(comments.dt >= comment_date_begin)
                         & (comments.dt < comment_date_end)]
    return comments


def get_comments_product_feat(start_date, end_date):
    # 商品评论特征
    dump_path = '../cache/comments_accumulate_%s_%s.pkl' % (
        start_date, end_date)
    if os.path.exists(open(dump_path)):
        comments = pickle.load(open(dump_path))
    else:
        comments = get_comments(start_date, end_date)
        comment_num = pd.get_dummies(
            comments['comment_num'], prefix='comment_num')
        comments = pd.concat([comments, comment_num], axis=1)
        # frames = ['sku_id', 'has_bad_comment', 'bad_comment_rate',
        # 'comment_num_1', 'comment_num_2', 'comment_num_3', 'comment_num_4']
        comments.drop(['dt'], axis=1, inplace=True)
        pickle.dump(comments, open(dump_path, 'w'))
    return comments


def get_accumulate_user_feat(start_date, end_date):
    # 获得用户累积特征
    """
    : user_action_1_ratio:购买/浏览
    : user_action_2_ratio:购买/加入购物车
    : user_action_3_ratio:购买/删除购物车（选择：加入购物车-删除购物车）
    : user_action_5_ratio:购买/关注
    : user_action_6_ratio:购买/点击

    """
    feature=['user_id', 'user_action_1_ratio', 'user_action_2_ratio',
        'user_action_3_ratio', 'user_action_5_ratio', 'user_action_6_ratio']
    dump_path = '../cache/user_feat_accumulate_%s_%s.pkl' % (start_date, end_date)
    if os.path.exists(dump_path):
        actions = pickle.load(open(dump_path))
    else:
        actions = get_actions(start_date, end_date)
        user_action_type = pd.get_dummies(actions['type'], prefix='action')
        actions = pd.concat([actions['user_id'], user_action_type], axis=1)
        actions = actions.groupby(['user_id'], as_index=False).sum()
        actions['user_action_1_ratio'] = actions['action_4'] / actions['action_1']
        actions['user_action_2_ratio'] = actions['action_4'] / actions['action_2']
        actions['user_action_3_ratio'] = actions['action_4'] / actions['action_3']
        actions['user_action_5_ratio'] = actions['action_4'] / actions['action_5']
        actions['user_action_6_ratio'] = actions['action_4'] / actions['action_6']
        actions = actions[feature]
        pickle.dump(actions, open(dump_path, 'w'))
    return actions

def get_accumulate_product_feat(start_date, end_date):
    # 获得商品累积特征
    """
    : product_action_1_ratio:购买/浏览
    : product_action_2_ratio:购买/加入购物车
    : product_action_3_ratio:购买/删除购物车（选择：加入购物车-删除购物车）
    : product_action_5_ratio:购买/关注
    : product_action_6_ratio:购买/点击

    """
    feature=['sku_id', 'product_action_1_ratio', 'product_action_2_ratio',
        'product_action_3_ratio', 'product_action_5_ratio', 'product_action_6_ratio']
    dump_path = '../cache/product_feat_accumulate_%s_%s.pkl' % (start_date, end_date)
    if os.path.exists(dump_path):
        actions = pickle.load(open(dump_path))
    else:
        actions = get_actions(start_date, end_date)
        user_action_type = pd.get_dummies(actions['type'], prefix='action')
        actions = pd.concat([actions['sku_id'], user_action_type], axis=1)
        del user_action_type
        actions = actions.groupby(['sku_id'], as_index=False).sum()
        actions['product_action_1_ratio'] = actions['action_4'] / actions['action_1']
        actions['product_action_2_ratio'] = actions['action_4'] / actions['action_2']
        actions['product_action_3_ratio'] = actions['action_4'] / actions['action_3']
        actions['product_action_5_ratio'] = actions['action_4'] / actions['action_5']
        actions['product_action_6_ratio'] = actions['action_4'] / actions['action_6']
        actions = actions[feature]
        pickle.dump(actions, open(dump_path, 'w'))
    return actions    

def get_labels(start_date, end_date):
    #获取label标签
    dump_path = '../cache/labels_%s_%s.pkl' % (start_date, end_date)
    if os.path.exists(dump_path):
        actions = pickle.load(open(dump_path))
    else:
        actions = get_actions(start_date, end_date)
        actions = actions[actions['type'] == 4]
        actions = actions.groupby(['user_id', 'sku_id'], as_index=False).sum()
        actions['label'] = 1
        actions = actions[['user_id', 'sku_id', 'label']]
        pickle.dump(actions, open(dump_path, 'w'))
    return actions

#%%
def make_test_set(train_start_date, train_end_date):
    #验证集
    dump_path = '../cache/test_set_%s_%s.pkl' % (train_start_date, train_end_date)
    if os.path.exists(dump_path):
        actions = pickle.load(open(dump_path))
    else:
        start_days = "2016-02-01"
        #labels = get_labels(test_start_date, test_end_date)
        # generate 时间窗口
        # actions = get_accumulate_action_feat(train_start_date, train_end_date)
        actions = None
        for i in (1, 2, 3, 5, 7, 10, 15, 21, 30):
            start_days = datetime.strptime(train_end_date, '%Y-%m-%d') - timedelta(days=i)
            start_days = start_days.strftime('%Y-%m-%d')
            if actions is None:
                actions = get_action_feat(start_days, train_end_date)
            else:
                #对一天前出现的数据进行统计，时间越长出现的用户商品对越多
                actions = pd.merge(actions, get_action_feat(start_days, train_end_date), how='outer',on=['user_id', 'sku_id'])
        # 用户商品对特征融合   
        user = get_basic_user_feat()
        actions = pd.merge(actions, user, how='left', on='user_id')
        del user
        user_acc = get_accumulate_user_feat(start_days, train_end_date)
        actions = pd.merge(actions, user_acc, how='left', on='user_id')
        del user_acc
        product = get_basic_product_feat()
        actions = pd.merge(actions, product, how='left', on='sku_id')
        del product
        product_acc = get_accumulate_product_feat(start_days, train_end_date)
        actions = pd.merge(actions, product_acc, how='left', on='sku_id')
        del product_acc 
        comment_acc = get_comments_product_feat(train_start_date, train_end_date)
        actions = pd.merge(actions, comment_acc, how='left', on='sku_id')
        del comment_acc
        actions.fillna(0,inplace=True)
        actions = actions[actions['cate'] == 8]
    
    users = actions[['user_id', 'sku_id']].copy()
    actions.drop(['user_id', 'sku_id'],axis=1,inplace=True)
    return users, actions

#%%
def make_train_set(train_start_date, train_end_date, test_start_date, test_end_date, days=30):
    #训练集
    dump_path = '../cache/train_set_%s_%s_%s_%s.pkl' % (train_start_date, train_end_date, test_start_date, test_end_date)
    if os.path.exists(dump_path):
        actions = pickle.load(open(dump_path))
    else:
        start_days = "2016-02-01"
        #labels = get_labels(test_start_date, test_end_date)
        # generate 时间窗口
        # actions = get_accumulate_action_feat(train_start_date, train_end_date)
        actions = None
        for i in (1, 2, 3, 5, 7, 10, 15, 21, 30):
            start_days = datetime.strptime(train_end_date, '%Y-%m-%d') - timedelta(days=i)
            start_days = start_days.strftime('%Y-%m-%d')
            if actions is None:
                actions = get_action_feat(start_days, train_end_date)
            else:
                #对一天前出现的数据进行统计，时间越长出现的用户商品对越多
                actions = pd.merge(actions, get_action_feat(start_days, train_end_date), how='outer',on=['user_id', 'sku_id'])
        # 用户商品对特征融合   
        user = get_basic_user_feat()
        actions = pd.merge(actions, user, how='left', on='user_id')
        del user
        user_acc = get_accumulate_user_feat(start_days, train_end_date)
        actions = pd.merge(actions, user_acc, how='left', on='user_id')
        del user_acc
        product = get_basic_product_feat()
        actions = pd.merge(actions, product, how='left', on='sku_id')
        del product
        product_acc = get_accumulate_product_feat(start_days, train_end_date)
        actions = pd.merge(actions, product_acc, how='left', on='sku_id')
        del product_acc 
        comment_acc = get_comments_product_feat(train_start_date, train_end_date)
        actions = pd.merge(actions, comment_acc, how='left', on='sku_id')
        del comment_acc
        labels = get_labels(test_start_date, test_end_date)
        actions = pd.merge(actions, labels, how='left', on=['user_id', 'sku_id'])
        del labels
        actions.fillna(0,inplace=True)
        actions = actions[actions['cate'] == 8]
    
    users = actions[['user_id', 'sku_id']].copy()
    actions.drop(['user_id', 'sku_id'],axis=1,inplace=True)
    labels = actions['label'].copy()
    actions.drop(['label'],axis=1,inplace=True)
    return users, actions, labels     

#%%
def report(pred, label):
    actions = label
    result = pred
    
    # 所有用户商品对
    all_user_item_pair = actions['user_id'].map(str) + '-' + actions['sku_id'].map(str)
    all_user_item_pair = np.array(all_user_item_pair)
    
    # 所有购买用户
    all_user_set = actions['user_id'].unique()
    
    # 所有品类中预测购买的用户
    all_user_test_set = result['user_id'].unique()
    all_user_test_item_pair = result['user_id'].map(str) + '-' + result['sku_id'].map(str)
    all_user_test_item_pair = np.array(all_user_test_item_pair)
    
    # 计算所有用户购买评价指标
    pos, neg = 0,0
    for user_id in all_user_test_set:
        if user_id in all_user_set:
            pos += 1
        else:
            neg += 1
    all_user_acc = 1.0 * pos / ( pos + neg)
    all_user_recall = 1.0 * pos / len(all_user_set)
    print '所有用户中预测购买用户的准确率为 ' + str(all_user_acc)
    print '所有用户中预测购买用户的召回率' + str(all_user_recall)
    
    pos, neg = 0, 0
    for user_item_pair in all_user_test_item_pair:
        if user_item_pair in all_user_item_pair:
            pos += 1
        else:
            neg += 1
    all_item_acc = 1.0 * pos / ( pos + neg)
    all_item_recall = 1.0 * pos / len(all_user_item_pair)
    print '所有用户中预测购买商品的准确率为 ' + str(all_item_acc)
    print '所有用户中预测购买商品的召回率' + str(all_item_recall)
    
    F11 = 6.0 * all_user_recall * all_user_acc / (5.0 * all_user_recall + all_user_acc)
    F12 = 5.0 * all_item_acc * all_item_recall / (2.0 * all_item_recall + 3 * all_item_acc)
    score = 0.4 * F11 + 0.6 * F12
    print 'F11=' + str(F11)
    print 'F12=' + str(F12)
    print 'score=' + str(score)
    
#%%

if __name__ == '__main__':
    train_start_date = '2016-02-01'
    train_end_date = '2016-03-01'
    test_start_date = '2016-03-01'
    test_end_date = '2016-03-05'
    user, action, label = make_train_set(train_start_date, train_end_date, test_start_date, test_end_date)
    print user.head(10)
    print action.head(10)