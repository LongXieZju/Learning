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
from datetime import datetime
from datetime import timedelta
from sklearn.preprocessing import OneHotEncoder

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
baseTime = datetime.strptime('2004-01-01', "%Y-%m-%d")

# type属性编码器
X_type = [[1], [2], [3], [4], [5], [6]]
encoder = OneHotEncoder(dtype=np.int, sparse=False)
encoder.fit(X_type)

#%%


def convertUserAge(age):
    # 根据购买力对年龄段进行排序，数值越大代表购买力越强
    if age == u'15岁以下':
        return 1
    elif age == u'56岁以上':
        return 2
    elif age == u'46-55岁':
        return 3
    elif age == u'16-25岁':
        return 4
    elif age == u'-1':
        return 5
    elif age == u'36-45岁':
        return 6
    elif age == u'26-35岁':
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


def read_csv_file(path, name):
    # 分块读取文件
    All = pd.read_csv(path, iterator=True, encoding='gbk')
    loop = True
    chunksize = 100000
    chunks = []
    while loop:
        try:
            chunk = All.get_chunk(chunksize)
            chunks.append(chunk)
        except StopIteration:
            loop = False
            print name + " has loaded!"
    action = pd.concat(chunks, ignore_index=True)
    return action


def get_basic_user_feat():
    # 对用户特征进行处理
    print "get_basic_user_feat start"
    name = 'basic_user'
    dump_path = name + '.csv'
    if os.path.exists(dump_path):
        user = read_csv_file(dump_path, name)
    else:
        user = read_csv_file(user_path, name)
        # 抛弃掉这三个用户
        user.drop([34072, 38905, 67704], axis=0, inplace=True)
        # 对年龄段编码，性别不处理（性别与购买力也有关系）
        user['age'] = user['age'].map(convertUserAge)
        user['user_reg_tm'] = user['user_reg_tm'].astype('str')
        user['user_reg_tm'] = user['user_reg_tm'].apply(lambda x: (
            datetime.strptime(x, "%Y-%m-%d") - baseTime).days)
        user.loc[user['user_reg_tm'] < 0, "user_reg_tm"] = np.nan
        user = convertUserRegTM(user)
        user = user.astype('int')
        user.sort_values('user_id', inplace=True)
        user.to_csv(dump_path, header=True, index=False)
        #pickle.dump(user, open(dump_path, 'w'))
    return user


def get_basic_product_feat():
    # 获得产品特征
    print "get_basic_product_feat start"
    name = 'basic_product'
    dump_path = name + '.csv'
    if os.path.exists(dump_path):
        product = read_csv_file(dump_path, name)
    else:
        product = read_csv_file(product_path, name)
        attr1_df = pd.get_dummies(product["a1"], prefix="a1")
        attr2_df = pd.get_dummies(product["a2"], prefix="a2")
        attr3_df = pd.get_dummies(product["a3"], prefix="a3")
        frames = ['sku_id', 'cate', 'brand']
        product = pd.concat(
            [product[frames], attr1_df, attr2_df, attr3_df], axis=1)
        product.sort_values(['sku_id'], inplace=True)
        product.to_csv(dump_path, header=True, index=False)
    return product


def get_action_2():
    # 获取用户2月份行为
    print "get_action_2 start"
    name = 'action'
    dump_path = name + '.csv'
    actAll = pd.read_csv(action_2_path, iterator=True)
    loop = True
    chunksize = 100000
    flag = 1
    while loop:
        try:
            chunk = actAll.get_chunk(chunksize)
            chunk = chunk.sort_values(['user_id'])
            if flag == 1:
                chunk.to_csv(dump_path, mode='a', header=True, index=False)
                flag = 2
            else:
                chunk.to_csv(dump_path, mode='a', header=False, index=False)
        except StopIteration:
            loop = False
            print "action_2 has been stored!"


def get_action_3():
    # 获取用户3月份行为
    print "get_action_3 start"
    name = 'action'
    dump_path = name + '.csv'
    actAll = pd.read_csv(action_3_path, iterator=True)
    loop = True
    chunksize = 100000
    while loop:
        try:
            chunk = actAll.get_chunk(chunksize)
            chunk = chunk.sort_values(['user_id'])
            chunk.to_csv(dump_path, mode='a', header=False, index=False)
        except StopIteration:
            loop = False
            print "action_3 has been stored!"


def get_action_4():
    # 获取用户4月份行为
    print "get_action_4 start"
    name = 'action'
    dump_path = name + '.csv'
    actAll = pd.read_csv(action_4_path, iterator=True)
    loop = True
    chunksize = 100000
    while loop:
        try:
            chunk = actAll.get_chunk(chunksize)
            chunk = chunk.sort_values(['user_id'])
            chunk.to_csv(dump_path, mode='a', header=False, index=False)
        except StopIteration:
            loop = False
            print "action_4 has been stored!"


def get_Allactions():
    # 存储用户三个月的行为
    print "get_Allactions start"
    dump_path = 'action.csv'
    if os.path.exists(dump_path):
        dump_path
    else:
        get_action_2()
        get_action_3()
        get_action_4()
    print "All actions have been stored!"


def get_actions(start_date, end_date):
    # 获取用户所有月份行为
    print "get_actions(%s, %s) start" % (start_date, end_date)
    name = 'all_action_%s_%s' % (start_date, end_date)
    dump_path = name + '.csv'
    if os.path.exists(dump_path):
        dump_path
    else:
        actAll = pd.read_csv('action.csv', iterator=True)
        loop = True
        chunksize = 1000000
        chunks = []
        flag = 1
        while loop:
            try:
                chunk = actAll.get_chunk(chunksize)
                chunks = chunk[(chunk.time >= start_date)& (chunk.time < end_date)]
                chunks =  chunks.sort_values(['user_id'])
                if flag == 1:
                    chunks.to_csv(dump_path, mode='a',header=True, index=False)
                    flag = 2
                else:
                    chunks.to_csv(dump_path, mode='a',header=False, index=False)
            except StopIteration:
                loop = False
                print name + " has been stored!"
    print "get_actions(%s, %s) finished" % (start_date, end_date)
    return dump_path
#%%
# 获得用户特征，cate未进行one-hot编码


def get_action_feat(start_date, end_date):

    # 使用自定义属性值来训练编码器，因为可能分块读取时属性值缺失
    print "get_action_feat(%s, %s) start" % (start_date, end_date)
    
    X_type = [[1], [2], [3], [4], [5], [6]]
    encoder = OneHotEncoder(dtype=np.int, sparse=False)
    encoder.fit(X_type)

    # 分块处理，节约内存
    name = 'action_accumulate_%s_%s' % (start_date, end_date)
    dump_path = name + '.csv'
    if os.path.exists(dump_path):
        dump_path
    else:
        actAll = pd.read_csv('action.csv', iterator=True)
        loop = True
        chunksize = 1000000
        actions = []
        flag = 1
        while loop:
            try:
                chunk = actAll.get_chunk(chunksize)
                actions = chunk[(chunk.time >= start_date)
                                & (chunk.time < end_date)]
                
                if actions.shape[0] != 0:
                    # 如果有数据则保存一次列名和数据，否则什么也不做
                    #user_action_type = encoder.transform(actions['type'])
                    # user_action_type = pd.get_dummies(
                    # actions['type'], prefix='%s-%s-action' % (start_date,
                    # end_date))
                    actions = actions.drop(['time', 'model_id'], axis=1)
                    actions = actions.sort_values(['user_id'])
                    
                    action_1 = '%s-%s-action_1' % (start_date, end_date)
                    action_2 = '%s-%s-action_2' % (start_date, end_date)
                    action_3 = '%s-%s-action_3' % (start_date, end_date)
                    action_4 = '%s-%s-action_4' % (start_date, end_date)
                    action_5 = '%s-%s-action_5' % (start_date, end_date)
                    action_6 = '%s-%s-action_6' % (start_date, end_date)
                    
                    user_action_type = encoder.transform([[i] for i in actions['type']])
                    user_action_type = pd.DataFrame(user_action_type)
                    user_action_type.columns = ([action_1, action_2, action_3, action_4, action_5, action_6])
                    user_action_type.index = actions.index
                    
                    actions = pd.concat([actions, user_action_type], axis=1)
                    del user_action_type
                    del actions['type']
                    actions = actions.groupby(
                        ['user_id', 'sku_id', 'cate', 'brand'], as_index=False).sum()
                    if flag == 1:
                        actions.to_csv(dump_path, mode='a',
                                       header=True, index=False)
                        flag = 2
                    else:
                        actions.to_csv(dump_path, mode='a',
                                       header=False, index=False)
            except StopIteration:
                loop = False
                print name + " has been stored!"
    print "get_action_feat(%s, %s) finished" % (start_date, end_date)
    return dump_path

#%%需要处理
# 获得用户累积特征
# 参考选手没有用这一块，可以自己改进下


def get_accumulate_action_feat(start_date, end_date):
    
    print "get_accumulate_action_feat(%s, %s) start" % (start_date, end_date)

    dump_path = '../cache/action_accumulate_%s_%s.pkl' % (start_date, end_date)
    if os.path.exists(dump_path):
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

#%%
# 获取商品评论信息


def get_comments(start_date, end_date):
    
    print "get_comments(%s, %s) start" % (start_date, end_date)

    name = 'comments_%s_%s' % (start_date, end_date)
    dump_path = name + '.csv'

    if os.path.exists(dump_path):
        dump_path
    else:
        comment_date_end = end_date
        comment_date_begin = comment_date[0]
        for date in reversed(comment_date):
            if date < comment_date_end:
                comment_date_begin = date
                break
        commentAll = pd.read_csv(comment_path, iterator=True)
        loop = True
        chunksize = 100000
        flag = 1
        while loop:
            try:
                chunk = commentAll.get_chunk(chunksize)
                comments = chunk[(chunk.dt >= comment_date_begin)
                                 & (chunk.dt < comment_date_end)]
                if comments.shape[0] != 0:
                    # 如果有数据则保存一次列名和数据，否则什么也不做
                    comments =  comments.sort_values(['sku_id'])
                    if flag == 1:
                        comments.to_csv(dump_path, mode='a',
                                        header=True, index=False)
                        flag = 2
                    else:
                        comments.to_csv(dump_path, mode='a',
                                        header=False, index=False)
            except StopIteration:
                loop = False
                print name + " has been stored!"
    print "get_comments(%s, %s) finished" % (start_date, end_date)
    return dump_path

#%%
# 商品评论特征，数据量较小，直接缓存返回


def get_comments_product_feat(start_date, end_date):

    name = 'comments_accumulate_%s_%s' % (start_date, end_date)
    dump_path = name + '.csv'
    
    # type属性编码器
    X_type = [[0], [1], [2], [3], [4]]
    Cencoder = OneHotEncoder(dtype=np.int, sparse=False)
    Cencoder.fit(X_type)
    
    if os.path.exists(dump_path):
        comments = pd.read_csv(dump_path)
    else:
        # 存储时间段内的评论
        path = get_comments(start_date, end_date)
        # 由于评论数数据量不大，可直接读取
        comments = pd.read_csv(path)
        
        user_comment = Cencoder.transform([[i] for i in comments['comment_num']])
        user_comment = pd.DataFrame(user_comment)
        user_comment.columns = (['comment_num_0', 'comment_num_1', 'comment_num_2', 'comment_num_3', 'comment_num_4'])
        user_comment.index = comments.index
        
        #comment_num = pd.get_dummies(comments['comment_num'], prefix='comment_num')
        comments = pd.concat([comments, user_comment], axis=1)
        comments.drop(['dt', 'comment_num'], axis=1, inplace=True)
        comments.sort_values(['sku_id'], inplace=True)
        comments.to_csv(dump_path, mode='a',
                        header=True, index=False)
        print name + " has been stored!"
    return comments

#%%需要处理
# onthot编码分块读取属性值缺失


def get_accumulate_user_feat(start_date, end_date):
    # 获得用户累积特征
    """
    : user_action_1_ratio:购买/浏览
    : user_action_2_ratio:购买/加入购物车
    : user_action_3_ratio:购买/删除购物车（选择：加入购物车-删除购物车）
    : user_action_5_ratio:购买/关注
    : user_action_6_ratio:购买/点击

    """

    feature = ['user_id', 'user_action_1_ratio', 'user_action_2_ratio',
               'user_action_3_ratio', 'user_action_5_ratio', 'user_action_6_ratio', 'action_4']

    name = 'user_feat_accumulate_%s_%s' % (start_date, end_date)
    dump_path = name + '.csv'

    if os.path.exists(dump_path):
        dump_path
        #actions = pickle.load(open(dump_path))
    else:
        path = get_actions(start_date, end_date)
        usersAll = pd.read_csv(path, iterator=True)

        loop = True
        chunksize = 2000000
        chunks=[]
        while loop:
            try:
                actions = usersAll.get_chunk(chunksize)
                if actions.shape[0] != 0:
                    # 如果有数据则保存一次列名和数据，否则什么也不做
                    actions =  actions.sort_values(['user_id'])
                    
                    user_action_type = encoder.transform([[i] for i in actions['type']])
                    user_action_type = pd.DataFrame(user_action_type)
                    user_action_type.columns = (['action_1', 'action_2', 'action_3', 'action_4', 'action_5', 'action_6'])
                    user_action_type.index = actions.index
                    actions = pd.concat([actions['user_id'], user_action_type], axis=1)
                    actions = actions.groupby(['user_id'], as_index=False).sum()
                    chunks.append(actions)
            except StopIteration:
                loop = False
                print name + " has been stored!"
        actions = pd.concat(chunks)
        actions = actions.groupby(['user_id'], as_index=False).sum()
        actions['user_action_1_ratio'] = actions['action_4'] / actions['action_1']
        actions['user_action_2_ratio'] = actions['action_4'] / actions['action_2']
        actions['user_action_3_ratio'] = actions['action_4'] / actions['action_3']
        actions['user_action_5_ratio'] = actions['action_4'] / actions['action_5']
        actions['user_action_6_ratio'] = actions['action_4'] / actions['action_6']
        actions = actions[feature]
        actions.to_csv(dump_path, mode='a',header=True, index=False)
    return dump_path


def get_accumulate_product_feat(start_date, end_date):
    # 获得商品累积特征
    """
    : product_action_1_ratio:购买/浏览
    : product_action_2_ratio:购买/加入购物车
    : product_action_3_ratio:购买/删除购物车（选择：加入购物车-删除购物车）
    : product_action_5_ratio:购买/关注
    : product_action_6_ratio:购买/点击

    """
    print "get_accumulate_product_feat start"
    feature = ['sku_id', 'product_action_1_ratio', 'product_action_2_ratio',
               'product_action_3_ratio', 'product_action_5_ratio', 'product_action_6_ratio', 'action_4']
    name = 'product_feat_accumulate_%s_%s' % (start_date, end_date)
    dump_path = name + '.csv'
    if os.path.exists(dump_path):
        dump_path
        #actions = pickle.load(open(dump_path))
    else:
        path = get_actions(start_date, end_date)
        productsAll = pd.read_csv(path, iterator=True)

        loop = True
        chunksize = 1000000
        chunks=[]
        while loop:
            try:
                chunk = productsAll.get_chunk(chunksize)
                
                if chunk.shape[0] != 0:
                    # 如果有数据则保存一次列名和数据，否则什么也不做
                    chunk = chunk.sort_values(['user_id'])
                    
                    user_action_type = encoder.transform(
                        [[i] for i in chunk['type']])
                    user_action_type = pd.DataFrame(user_action_type)
                    user_action_type.columns = (
                        ['action_1', 'action_2', 'action_3', 'action_4', 'action_5', 'action_6'])
                    user_action_type.index = chunk.index
                    chunk = pd.concat([chunk['sku_id'], user_action_type], axis=1)
                    chunk = chunk.groupby(['sku_id'], as_index=False).sum()
                    chunks.append(chunk)
            except StopIteration:
                loop = False
        actions = pd.concat(chunks)
        actions = actions.groupby(['sku_id'], as_index=False).sum()
        actions['product_action_1_ratio'] = actions['action_4'] / actions['action_1']
        actions['product_action_2_ratio'] = actions['action_4'] / actions['action_2']
        actions['product_action_3_ratio'] = actions['action_4'] / actions['action_3']
        actions['product_action_5_ratio'] = actions['action_4'] / actions['action_5']
        actions['product_action_6_ratio'] = actions['action_4'] / actions['action_6']
        actions = actions[feature]
        actions.to_csv(dump_path, mode='a',header=True, index=False)
        print name + " has been stored!"
    print "get_accumulate_product_feat finished"
    return dump_path


def get_labels(start_date, end_date):
    # 获取label标签
    # start_date—end_date时间段内有过购买行为的作为正样本
    print 'get_labels start'
    name = 'labels_%s_%s' % (start_date, end_date)
    dump_path = name + '.csv'
    if os.path.exists(dump_path):
        dump_path
        #actions = pickle.load(open(dump_path))
    else:
        path = get_actions(start_date, end_date)
        labelsAll = pd.read_csv(path, iterator=True)

        loop = True
        chunksize = 1000000
        chunks=[]
        while loop:
            try:
                chunk = labelsAll.get_chunk(chunksize)
                if chunk.shape[0] != 0:
                    # 如果有数据则保存一次列名和数据，否则什么也不做
                    chunk = chunk.sort_values(['user_id'])
                    chunk = chunk[chunk['type'] == 4]
                    chunk = chunk.groupby(['user_id', 'sku_id'], as_index=False).sum()
                    chunks.append(chunk)
            except StopIteration:
                loop = False
        actions = pd.concat(chunks)
        actions = actions.groupby(['user_id', 'sku_id'], as_index=False).sum()
        actions['label'] = 1
        actions = actions[['user_id', 'sku_id', 'label']]
        actions = actions.astype('int')
        actions.to_csv(dump_path, mode='a',header=True, index=False)
        print name + " has been stored!"
    print 'get_labels finished'
    return dump_path

#%%
# 将累积特征分开保存
# （抛弃掉没有任何行为特征的样本）
# 未用前30天的特征


def store_set(start_date, end_date):
    start_days = "2016-02-01"
    paths = []
    for i in (1, 2, 3, 5, 7, 10, 15, 21):
        start_days = datetime.strptime(end_date, '%Y-%m-%d') - timedelta(days=i)
        start_days = start_days.strftime('%Y-%m-%d')
        path = get_action_feat(start_days, end_date)
        paths.append(path)
    for i in (1, 2, 3, 5, 7, 10, 15, 21, 30):
        start_days = datetime.strptime(end_date, '%Y-%m-%d') - timedelta(days=i)
        start_days = start_days.strftime('%Y-%m-%d')
        path = get_accumulate_user_feat(start_days, end_date)
        paths.append(path)
    for i in (1, 2, 3, 5, 7, 10, 15, 21, 30):
        start_days = datetime.strptime(end_date, '%Y-%m-%d') - timedelta(days=i)
        start_days = start_days.strftime('%Y-%m-%d')
        path = get_accumulate_product_feat(start_days, end_date)
        paths.append(path)
    return paths




def after_process():
    # 后续处理，store_set将样本特征分时段存储后可能会有重复的用户商品对，需要再次对结果聚合
    start_date = "2016-02-01"
    end_date = "2016-04-11"
    paths = store_set(start_date, end_date)
    for i in range(8):
        if os.path.exists('after_' + paths[i]):
            print 'after_' + paths[i]
        else:
            All = pd.read_csv(paths[i])
            actions=All.groupby(['user_id', 'sku_id', 'cate', 'brand'], as_index=False).sum()
            actions.to_csv('after_' + paths[i],mode='a', header=True, index=False)

#%%

def merge_set(train_start_date, train_end_date):
    # 融合滑窗得到的特征，保存为actionSet.csv
    print 'merge_set start'
    
    name = 'actionSet'
    dump_path = name + '.csv'
    
    if os.path.exists(dump_path):
        dump_path
    else:
        paths = store_set(train_start_date, train_end_date)
        actionsAll = pd.read_csv('after_' + paths[7])
        actionsAll['user_id'] = actionsAll['user_id'].astype('int')
        for i in reversed(range(26)):
            print i
            print paths[i]
            if (i>=17) & (i<=25):
                action = pd.read_csv(paths[i])
                actionsAll = pd.merge(actionsAll, action, how='left', on=['sku_id'])
                #actionsAll = actionsAll.sort_values(['user_id'])
                del action
            elif (i<=16) & (i>=8):
                action = pd.read_csv(paths[i])
                actionsAll = pd.merge(actionsAll, action, how='left', on=['user_id'])
                #actionsAll = actionsAll.sort_values(['user_id'])
                del action
            elif i == 7:
                continue
            else:
                action = pd.read_csv('after_' + paths[i])
                actionsAll = pd.merge(actionsAll, action, how='left', on=['user_id', 'sku_id', 'cate', 'brand'])
                del action
        actionsAll = actionsAll.sort_values(['user_id'])
        actionsAll.to_csv(dump_path, mode='a', header=True, index=False)
        print name + ' has been stored!'
    print 'merge_set finished'
    return dump_path


def make_test_set(train_start_date, train_end_date):
    # 测试集
    # 前一个月中出现的所有用户商品对
    print "make_test_set start"
    name = 'test_set_%s_%s' % (train_start_date, train_end_date)
    dump_path = name + '.csv'
    paths = ['users_test_set.csv', 'actions_test_set.csv']
    if os.path.exists(paths[0]):
        paths
    else:
        #labels = get_labels(test_start_date, test_end_date)
        # generate 时间窗口
        # actions = get_accumulate_action_feat(train_start_date, train_end_date)
        path = merge_set(train_start_date, train_end_date)
        actionsAll = pd.read_csv(path, iterator=True)
        loop = True
        chunksize = 100000
        start_days = train_start_date#"2016-03-12"
        flag = 1
        while loop:
            try:
                # 用户商品对特征融合
                actions = actionsAll.get_chunk(chunksize)
                actions['user_id'] = actions['user_id'].astype('int')
                actions['sku_id'] = actions['sku_id'].astype('int')
                user = get_basic_user_feat()
                user['user_id'] = user['user_id'].astype('int')
                actions = pd.merge(actions, user,on='user_id',how='left')
                del user
                # path = get_accumulate_user_feat(start_days, train_end_date)
                # user_acc = pd.read_csv(path)
                # user_acc['user_id'] = user_acc['user_id'].astype('int')
                # actions = pd.merge(actions, user_acc,on='user_id',how='left')
                # del user_acc
                product = get_basic_product_feat()
                product['sku_id'] = product['sku_id'].astype('int')
                actions = pd.merge(actions, product,on='sku_id',how='left')
                del product
                # path = get_accumulate_product_feat(start_days, train_end_date)
                # product_acc = pd.read_csv(path)
                # product_acc['sku_id'] = product_acc['sku_id'].astype('int')
                # actions = pd.merge(actions, product_acc,on='sku_id',how='left')
                # del product_acc
                comment_acc = get_comments_product_feat(train_start_date, train_end_date)
                comment_acc['sku_id'] = comment_acc['sku_id'].astype('int')
                actions = pd.merge(actions, comment_acc,on='sku_id',how='left')
                del comment_acc
                
                actions = actions.drop(['cate_y','brand_y'],axis=1)
                actions.rename(columns={'cate_x':'cate', 'brand_x':'brand'},inplace=True)

                actions.fillna(0, inplace=True)
                actions.replace([np.inf,-np.inf], 1, inplace=True)
                actions = actions[actions['cate'] == 8]
                
                users = actions[['user_id', 'sku_id']].copy()
                actions = actions.drop(['user_id', 'sku_id'], axis=1)
                
                if flag == 1:
                    users.to_csv(paths[0], mode='a', header=True, index=False)
                    actions.to_csv(paths[1], mode='a', header=True, index=False)
                    flag = 2
                else:
                    users.to_csv(paths[0], mode='a', header=False, index=False)
                    actions.to_csv(paths[1], mode='a', header=False, index=False)
            except StopIteration:
                loop = False
                print paths[0] + " has been merged!"
                print paths[1] + " has been merged!"
    return paths

#%%


def make_train_set(train_start_date, train_end_date, test_start_date, test_end_date, days=30):
    # 训练集
    name = 'labels'
    dump_path = name + '.csv' 
    
    
    paths = ['users_train_set.csv', 'actions_train_set.csv', 'labels.csv']
    if os.path.exists(paths[0]):
        paths
    else:
        #labels = get_labels(test_start_date, test_end_date)
        # generate 时间窗口
        # actions = get_accumulate_action_feat(train_start_date, train_end_date)
        path = merge_set(train_start_date, train_end_date)
        actionsAll = pd.read_csv(path, iterator=True)
        loop = True
        chunksize = 100000
        start_days = train_start_date#"2016-03-12"
        flag = 1
        while loop:
            try:
                # 用户商品对特征融合
                actions = actionsAll.get_chunk(chunksize)
                actions['user_id'] = actions['user_id'].astype('int')
                actions['sku_id'] = actions['sku_id'].astype('int')
                user = get_basic_user_feat()
                user['user_id'] = user['user_id'].astype('int')
                actions = pd.merge(actions, user,on='user_id',how='left')
                del user
                # path = get_accumulate_user_feat(start_days, train_end_date)
                # user_acc = pd.read_csv(path)
                # user_acc['user_id'] = user_acc['user_id'].astype('int')
                # actions = pd.merge(actions, user_acc,on='user_id',how='left')
                # del user_acc
                product = get_basic_product_feat()
                product['sku_id'] = product['sku_id'].astype('int')
                actions = pd.merge(actions, product,on='sku_id',how='left')
                del product
                # path = get_accumulate_product_feat(start_days, train_end_date)
                # product_acc = pd.read_csv(path)
                # product_acc['sku_id'] = product_acc['sku_id'].astype('int')
                # actions = pd.merge(actions, product_acc,on='sku_id',how='left')
                # del product_acc
                comment_acc = get_comments_product_feat(train_start_date, train_end_date)
                comment_acc['sku_id'] = comment_acc['sku_id'].astype('int')
                actions = pd.merge(actions, comment_acc,on='sku_id',how='left')
                del comment_acc
                path = get_labels(test_start_date, test_end_date)
                labels =  pd.read_csv(path)
                labels['user_id'] = labels['user_id'].astype('int')
                labels['sku_id'] = labels['sku_id'].astype('int')
                actions = pd.merge(actions, labels,on=['user_id', 'sku_id'],how='left')
                del labels
                
                actions = actions.drop(['cate_y','brand_y'],axis=1)
                actions.rename(columns={'cate_x':'cate', 'brand_x':'brand'},inplace=True)
                
                actions.fillna(0, inplace=True)
                actions.replace([np.inf,-np.inf], 1, inplace=True)
                #actions = actions[actions['cate'] == 8]
                
                users = actions[['user_id', 'sku_id']].copy()
                labels = actions['label'].copy()
                actions = actions.drop(['user_id', 'sku_id', 'label'], axis=1)
                
                if flag == 1:
                    users.to_csv(paths[0], mode='a', header=True, index=False)
                    actions.to_csv(paths[1], mode='a', header=True, index=False)
                    labels.to_csv(paths[2], mode='a', header=True, index=False)
                    flag = 2
                else:
                    users.to_csv(paths[0], mode='a', header=False, index=False)
                    actions.to_csv(paths[1], mode='a', header=False, index=False)
                    labels.to_csv(paths[2], mode='a', header=False, index=False)
            except StopIteration:
                loop = False
                print paths[0] + " has been merged!"
                print paths[1] + " has been merged!"
                print paths[2] + " has been merged!"
    return paths
    
    
#%%


def report(pred, label):
    # 真实label
    actions = label
    # 预测结果
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
    pos, neg = 0, 0
    for user_id in all_user_test_set:
        if user_id in all_user_set:
            pos += 1
        else:
            neg += 1
    all_user_acc = 1.0 * pos / (pos + neg)
    all_user_recall = 1.0 * pos / len(all_user_set)
    print '所有用户中预测购买用户的准确率为 ' + str(all_user_acc)
    print '所有用户中预测购买用户的召回率' + str(all_user_recall)

    pos, neg = 0, 0
    for user_item_pair in all_user_test_item_pair:
        if user_item_pair in all_user_item_pair:
            pos += 1
        else:
            neg += 1
    all_item_acc = 1.0 * pos / (pos + neg)
    all_item_recall = 1.0 * pos / len(all_user_item_pair)
    print '所有用户中预测购买商品的准确率为 ' + str(all_item_acc)
    print '所有用户中预测购买商品的召回率' + str(all_item_recall)

    F11 = 6.0 * all_user_recall * all_user_acc / \
        (5.0 * all_user_recall + all_user_acc)
    F12 = 5.0 * all_item_acc * all_item_recall / \
        (2.0 * all_item_recall + 3 * all_item_acc)
    score = 0.4 * F11 + 0.6 * F12
    print 'F11=' + str(F11)
    print 'F12=' + str(F12)
    print 'score=' + str(score)

#%%

if __name__ == '__main__':
    train_start_date = '2016-03-31'
    train_end_date = '2016-04-11'
    
    test_start_date = '2016-04-11'
    test_end_date = '2016-04-16'

    #get_Allactions()
    #store_set(train_start_date, train_end_date)
    #after_process()
#    make_test_set(train_start_date, train_end_date)
#    make_train_set(train_start_date, train_end_date, test_start_date, test_end_date)
#     print user.head(10)
#     print action.head(10)
#     print label.head(10)

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
