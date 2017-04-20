#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2017-03-30 05:42:31
# @Author  : Xielong (xielong@zju.edu.cn)
# @Link    : https://github.com/XieLongZju
# @Version : V0.1


import numpy as np
import pandas as pd
import time
import pickle
import os
import math
import datetime
#from datetime import timedelta

#%%
# Setting
# size = 100000
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
        pickle.dump(user,open(dump_path,'w'))
    return user

def get_basic_product_feat():
    #获得产品特征
    dump_path = '../cache/basic_product.pkl'
    if os.path.exists(dump_path):
        product = pickle.load(open(dump_path))
    else:
        product = pd.read_csv(product_path,encoding='gbk')
        attr1_df = pd.get_dummies(product["a1"],prefix="a1")
        attr2_df = pd.get_dummies(product["a2"],prefix="a2")
        attr3_df = pd.get_dummies(product["a3"],prefix="a3")
        frames = ['sku_id', 'cate', 'brand']
        product = pd.concat([product[frames], attr1_df, attr2_df, attr3_df], axis = 1)
        pickle.dump(product,open(dump_path,'w'))
    return product

def get_action_2():
    #获得2月用户行为
    actAll = pd.read_csv(action_2_path,iterator=True)
    loop=True
    chunksize=100000
    chunks=[]
    while loop:
        try:
            chunk = actAll.get_chunk(chunksize)
            chunks.append(chunk)
        except StopIteration:
            loop=False
            print "get_action_2 Iteration is stopped!"
    action=pd.concat(chunks,ignore_index=True)
    return action

def get_action_3():
    #获得3月用户行为
    actAll = pd.read_csv(action_3_path,iterator=True)
    loop=True
    chunksize=100000
    chunks=[]
    while loop:
        try:
            chunk = actAll.get_chunk(chunksize)
            chunks.append(chunk)
        except StopIteration:
            loop=False
            print "get_action_3 Iteration is stopped!"
    action=pd.concat(chunks,ignore_index=True)
    return action

def get_action_4():
    #获得4月用户行为
    actAll = pd.read_csv(action_4_path,iterator=True)
    loop=True
    chunksize=100000
    chunks=[]
    while loop:
        try:
            chunk = actAll.get_chunk(chunksize)
            chunks.append(chunk)
        except StopIteration:
            loop=False
            print "get_action_4 Iteration is stopped!"
    action=pd.concat(chunks,ignore_index=True)
    return action

def get_action(start_date, end_date):
    #获得2、3、4月用户行为
    dump_path = '../cache/all_action_%s_%s.pkl' % (start_date, end_date)
    if os.path.exists(dump_path):
        actions = pickle.load(open(dump_path))
    else:
        actions3 = get_action_3()
        actions = get_action_4()
        actions = pd.concat([actions3,actions])
        del actions3
        actions2 = get_action_2()
        actions = pd.concat([actions2,actions])
        del actions2
        actions = actions[(actions.time>=start_date)&(actions.time<end_date)]
        pickle.dump(actions,open(dump_path,'w'))
    return actions

def get_action_feat(start_date, end_date):
    dump_path='../cache/action_accumulate_%s_%s.pkl' % (start_date, end_date)
    if os.path.exists(open(dump_path)):
        actions = pickle.load(open(dump_path))
    else:
        actions = get_actions(start_date, end_date)#获得希望时间段内用户行为
        actions = actions[['user_id','sku_id','type']]
        user_action_type = pd.get_dummies(actions['type'],prefix='%s-%s-action' % (start_date, end_date))
        
























