
# coding: utf-8

# In[26]:

#!/usr/bin/env python

# @Date    : 2017-03-30 05:42:31
# @Author  : Xielong (xielong@zju.edu.cn)
# @Link    : https://github.com/XieLongZju
# @Version : V0.1

# coding: utf-8

import numpy as np
import pandas as pd
import os
from datetime import datetime
from datetime import timedelta
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
import xgboost as xgb

# Setting
basic_path = 'Cache/BasicFeat/'
com_path = 'Cache/Comments/'
action_path = 'Cache/Actions/'
sub_path = 'Cache/Submission/'
data_path = 'DataSet/'
user_path = data_path + 'JData_User.csv'
comment_path = data_path + 'JData_Comment.csv'
product_path = data_path + 'JData_Product.csv'
action_2_path = data_path + 'JData_Action_201602.csv'
action_3_path = data_path + 'JData_Action_201603.csv'
action_4_path = data_path + 'JData_Action_201604.csv'

comment_date = ["2016-02-01", "2016-02-08", "2016-03-15", "2016-02-22",
                "2016-02-19", "2016-03-07", "2016-03-14", "2016-03-21",
                "2016-03-28", "2016-04-04", "2016-04-11", "2016-04-15"]

# 注册基准时间，京东多媒体网开通时间
baseTime = datetime.strptime('2004-01-01', "%Y-%m-%d")

# type属性编码器
X_type = [[1], [2], [3], [4], [5], [6]]
encoder = OneHotEncoder(dtype=np.int, sparse=False)
encoder.fit(X_type)


# In[27]:


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
    dump_path = basic_path + name + '.csv'
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
        user['user_lv_reg'] = 1.0 * user['user_reg_tm'] / user['user_lv_cd']
        user.sort_values('user_id', inplace=True)
        user.to_csv(dump_path, header=True, index=False)
    return user

# temp = get_basic_user_feat()
# print temp.head(10)
# del temp


# In[28]:

def get_basic_product_feat():
    # 获得产品特征
    print "get_basic_product_feat start"
    name = 'basic_product'
    dump_path = basic_path + name + '.csv'
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

# temp = get_basic_product_feat()
# print temp.head(10)
# del temp


# In[29]:

def get_comments(start_date, end_date):
    
    # 获取商品评论信息
    comment_date_end = end_date
    comment_date_begin = comment_date[0]
    for date in reversed(comment_date):
        if date < comment_date_end:
            comment_date_begin = date
            break
    name = 'comments_%s_%s' % (comment_date_begin, comment_date_end)
    dump_path = com_path +  name + '.csv'
    
    print "get_comments(%s,%s) start" % (comment_date_begin, comment_date_end)

    if os.path.exists(dump_path):
        comments = pd.read_csv(dump_path)
    else:
        commentAll = pd.read_csv(comment_path)
        comments = commentAll[(commentAll.dt >= comment_date_begin) & (commentAll.dt < comment_date_end)]
        del commentAll
        comments.to_csv(dump_path, header=True, index=False)
    print "get_comments(%s, %s) finished" % (start_date, end_date)
    return comments

# temp = get_comments('2016-03-31','2016-04-11')
# print temp.head(10)
# del temp


# In[30]:

def get_user_action_feat(end_date):
    start_date = datetime.strptime(end_date, '%Y-%m-%d') - timedelta(days=3)
    start_date = start_date.strftime('%Y-%m-%d')
    # end_date = '2016-04-16'
    print "get_user_action_feat(%s, %s) start" % (start_date, end_date)
    name = 'diff_sku_cate_%s_%s' % (start_date, end_date)
    dump_path = basic_path + name + '.csv'
    
    if os.path.exists(dump_path):
        diff_sku_cate = pd.read_csv(dump_path)
    else:
        actAll = pd.read_csv(action_4_path)
        actions = actAll[(actAll.time >= start_date)& (actAll.time < end_date)]
        # actions['time'] = actions['time'].apply(lambda x: (datetime.strptime(x, "%Y-%m-%d") - baseTime).days)
        actions['user_id'] = actions['user_id'].astype('int')
        del actAll
        actions = actions.groupby(['user_id', 'sku_id', 'cate', 'brand'], as_index=False).sum()
        actions = actions[['user_id', 'sku_id', 'cate', 'brand']]
        diff_sku_cate = pd.DataFrame()
        # 一个用户看过多少不同的商品（reason：用户如果想购买会进行商品比对，会存在噪声，用户会不小心点到其他商品）
        # 一个用户看过多少不同品类的商品（reason：用户如果想购买会比对相同品类的商品，如果随意浏览则会对很多品类进行操作）
        print len(actions['user_id'].unique())
        for user in actions['user_id'].unique():
            user_action = pd.DataFrame()
            user_action = actions[actions['user_id'] == user]
            user_action['diff_sku'] = len(user_action['sku_id'].unique())
            user_action['diff_cate'] = len(user_action['cate'].unique())
            diff_sku_cate = pd.concat([diff_sku_cate, user_action], axis=0)
            del user_action
        diff_sku_cate = diff_sku_cate.groupby(['user_id'], as_index=False).mean()
        diff_sku_cate = diff_sku_cate[['user_id', 'diff_sku', 'diff_cate']]
        diff_sku_cate['ksu_cate_ratio'] = diff_sku_cate['diff_sku'] / diff_sku_cate['diff_cate']
        diff_sku_cate.to_csv(dump_path, header=True, index=False)
#     diff_sku_cate['ksu_cate_ratio'] = diff_sku_cate['diff_sku'] / diff_sku_cate['diff_cate']
#     diff_sku_cate.to_csv('Cache/BasicFeat/'+name+'.csv', header=True, index=False)
    return diff_sku_cate
        # 一个用户对该商品操作时间统计（reason：操作时间越长，越可能购买）
        # 可对model_id进行统计排序，然后进行编号（reason：点击搜索模块的更有倾向购买）
        

# temp = get_user_action_feat('2016-04-16')
# print temp.head(10)
# del temp


# In[7]:

def model_id_reset():
    act = pd.read_csv(action_4_path)
    act['label'] = 1
    act['model_id'] = act['model_id'].fillna(0)
    act = act.sort_values(['user_id'])
    act = act.groupby(['model_id'], as_index=False).sum()
    act = act[['model_id', 'label']]
    act_result = act.sort_values(['label']).reset_index().reset_index()
    act_result = act_result[['model_id', 'level_0']]
    del act
    return act_result


# In[31]:

def brand_reset():
    act = pd.read_csv(action_4_path)
    act = act.sort_values(['user_id'])
    act_type = pd.get_dummies(act['type'], prefix='action')
    act = pd.concat([act, act_type], axis=1)
    del act_type
    act = act.groupby(['brand'], as_index=False).sum()
    act = act[['brand', 'action_4']]
    act_result = act.sort_values(['action_4']).reset_index().reset_index()
    act_result = act_result[['brand', 'level_0']]
    act_result.rename(columns={'level_0':'brand_reset'}, inplace=True)
    del act
    return act_result

# temp = brand_reset()
# print temp
# del temp


# In[34]:

def get_action_feat(start_date, end_date):
    # 获得时间段内用户特征
    print "get_action_feat(%s, %s) start" % (start_date, end_date)
    name = 'action_accumulate_%s_%s' % (start_date, end_date)
    dump_path = action_path + name + '.csv'
    
    if os.path.exists(dump_path):
        actions = pd.read_csv(dump_path)
    else:
        print '1'
        actAll = pd.read_csv(action_4_path)
        actions = actAll[(actAll.time >= start_date)& (actAll.time < end_date)]
        del actAll
        actions['user_id'] = actions['user_id'].astype('int')
#         model_id_reset = model_id_reset()
#         actions = actions.merge(model_id_reset, on=['model_id'], how='left')
        user_action_type = pd.get_dummies(actions['type'], prefix='action')
        actions = pd.concat([actions, user_action_type], axis=1)
        del user_action_type
#         actions = actions.drop(['type'], axis=1)
        # 等会处理
        # actions.loc[actions['model_id'] > 0,'model_id'] = 1
        actions = actions.groupby(['user_id', 'sku_id', 'cate', 'brand'], as_index=False).sum()
        productsAll = get_basic_product_feat()
        actions = actions.merge(productsAll, on=['sku_id', 'cate', 'brand'], how='left')
        del productsAll
        usersAll = get_basic_user_feat()
        actions = actions.merge(usersAll, on=['user_id'], how='left')
        del usersAll
        
        # 不同品类累计特征
        cate_feat = actions.groupby(['cate'], as_index=False).sum()
        cate_feat = cate_feat[['cate', 'action_1', 'action_2', 'action_3', 'action_4', 'action_5', 'action_6']]
        cate_feat['ratio1'] = cate_feat['action_4']/cate_feat['action_1']
        cate_feat['ratio2'] = cate_feat['action_4']/cate_feat['action_2']
        cate_feat['ratio3'] = cate_feat['action_4']/cate_feat['action_3']
        cate_feat['ratio5'] = cate_feat['action_4']/cate_feat['action_5']
        cate_feat['ratio6'] = cate_feat['action_4']/cate_feat['action_6']
        # 不同品牌累计特征
        brand_feat = actions.groupby(['brand'], as_index=False).sum()
        brand_feat = brand_feat[['brand', 'action_1', 'action_2', 'action_3', 'action_4', 'action_5', 'action_6']]
        brand_feat['ratio1'] = brand_feat['action_4']/brand_feat['action_1']
        brand_feat['ratio2'] = brand_feat['action_4']/brand_feat['action_2']
        brand_feat['ratio3'] = brand_feat['action_4']/brand_feat['action_3']
        brand_feat['ratio5'] = brand_feat['action_4']/brand_feat['action_5']
        brand_feat['ratio6'] = brand_feat['action_4']/brand_feat['action_6']
        # 不同用户累积特征
        user_feat = actions.groupby(['user_id'], as_index=False).sum()
        user_feat = user_feat[['user_id', 'action_1', 'action_2', 'action_3', 'action_4', 'action_5', 'action_6']]
        user_feat['ratio1'] = user_feat['action_4']/user_feat['action_1']
        user_feat['ratio2'] = user_feat['action_4']/user_feat['action_2']
        user_feat['ratio3'] = user_feat['action_4']/user_feat['action_3']
        user_feat['ratio5'] = user_feat['action_4']/user_feat['action_5']
        user_feat['ratio6'] = user_feat['action_4']/user_feat['action_6']
        # 不同商品累积特征
        sku_feat = actions.groupby(['sku_id'], as_index=False).sum()
        sku_feat = sku_feat[['sku_id', 'action_1', 'action_2', 'action_3', 'action_4', 'action_5', 'action_6']]
        sku_feat['ratio1'] = sku_feat['action_4']/sku_feat['action_1']
        sku_feat['ratio2'] = sku_feat['action_4']/sku_feat['action_2']
        sku_feat['ratio3'] = sku_feat['action_4']/sku_feat['action_3']
        sku_feat['ratio5'] = sku_feat['action_4']/sku_feat['action_5']
        sku_feat['ratio6'] = sku_feat['action_4']/sku_feat['action_6']
        # 不同年龄段累计特征
        age_feat = actions.groupby(['age'], as_index=False).sum()
        age_feat = age_feat[['age', 'action_1', 'action_2', 'action_3', 'action_4', 'action_5', 'action_6']]
        age_feat['ratio1'] = age_feat['action_4']/age_feat['action_1']
        age_feat['ratio2'] = age_feat['action_4']/age_feat['action_2']
        age_feat['ratio3'] = age_feat['action_4']/age_feat['action_3']
        age_feat['ratio5'] = age_feat['action_4']/age_feat['action_5']
        age_feat['ratio6'] = age_feat['action_4']/age_feat['action_6']
        # 不同性别累计特征
        sex_feat = actions.groupby(['sex'], as_index=False).sum()
        sex_feat = sex_feat[['sex', 'action_1', 'action_2', 'action_3', 'action_4', 'action_5', 'action_6']]
        sex_feat['ratio1'] = sex_feat['action_4']/sex_feat['action_1']
        sex_feat['ratio2'] = sex_feat['action_4']/sex_feat['action_2']
        sex_feat['ratio3'] = sex_feat['action_4']/sex_feat['action_3']
        sex_feat['ratio5'] = sex_feat['action_4']/sex_feat['action_5']
        sex_feat['ratio6'] = sex_feat['action_4']/sex_feat['action_6']
        # 不同用户等级累计特征
        lv_feat = actions.groupby(['user_lv_cd'], as_index=False).sum()
        lv_feat = lv_feat[['user_lv_cd', 'action_1', 'action_2', 'action_3', 'action_4', 'action_5', 'action_6']]
        lv_feat['ratio1'] = lv_feat['action_4']/lv_feat['action_1']
        lv_feat['ratio2'] = lv_feat['action_4']/lv_feat['action_2']
        lv_feat['ratio3'] = lv_feat['action_4']/lv_feat['action_3']
        lv_feat['ratio5'] = lv_feat['action_4']/lv_feat['action_5']
        lv_feat['ratio6'] = lv_feat['action_4']/lv_feat['action_6']
        # 融合
        actions = actions.merge(cate_feat, on=['cate'], how='left')
        del cate_feat
        actions = actions.merge(brand_feat, on=['brand'], how='left')
        del brand_feat
        actions = actions.merge(user_feat, on=['user_id'], how='left')
        del user_feat
        actions = actions.merge(sku_feat, on=['sku_id'], how='left')
        del sku_feat
        actions = actions.merge(age_feat, on=['age'], how='left')
        del age_feat
        actions = actions.merge(sex_feat, on=['sex'], how='left')
        del sex_feat
        actions = actions.merge(lv_feat, on=['user_lv_cd'], how='left')
        del lv_feat
        # 可对品牌根据销量排序（reason：考虑品牌可靠度）
        temp = brand_reset()
        actions = actions.merge(temp, on=['brand'], how='left')
        del temp
#         temp = get_user_action_feat()
#         actions = actions.merge(temp, on=['user_id'], how='left')
#         del temp
        actions.to_csv(dump_path, mode='a', header=True, index=False)
    print "get_action_feat(%s, %s) finished" % (start_date, end_date)
    return actions

# temp = get_action_feat('2016-04-10','2016-04-11')
# print temp.columns
# print len(temp)
# print temp.head()
# del temp


# In[36]:

def store_set(start_date, end_date):
    print "store_set(%s, %s) start" % (start_date, end_date)
    name = 'store_set_%s_%s' % (start_date, end_date)
    dump_path = action_path + name + '.csv'
    
    if os.path.exists(dump_path):
        feat_set = pd.read_csv(dump_path)
    else:
        start_days = "2016-02-01"
        feat_set = pd.DataFrame()
        flag = 1
        for i in reversed(range(7)):
            start_days = datetime.strptime(end_date, '%Y-%m-%d') - timedelta(days=i+1)
            start_days = start_days.strftime('%Y-%m-%d')
            print start_days
            actions = get_action_feat(start_days, end_date)
            if flag == 1:
                feat_set = actions
                del actions
                flag = 2
            else:
                feat_set = feat_set.merge(actions, on=['user_id', 'sku_id', 'cate', 'brand'], how='left')
                del actions
        user_action = get_user_action_feat(end_date)
        feat_set = feat_set.merge(user_action, on=['user_id'], how='left')
        del user_action 
        print len(feat_set)
        comment = get_comments(start_date, end_date)
        feat_set = feat_set.merge(comment, on=['sku_id'], how='left')
        del comment 
        feat_set = feat_set.drop(['td'], axis=1)
        print len(feat_set)
        feat_set.to_csv(dump_path, mode='a', header=True, index=False)
    print "store_set(%s, %s) finished" % (start_date, end_date)
    return feat_set

# temp = store_set('2016-03-31','2016-04-11')
# print len(temp)
# print temp.head()
# del temp


# In[11]:

def get_labels(start_date, end_date):
    # 获取label标签
    # start_date—end_date时间段内有过购买行为的作为正样本
    print 'get_labels start'
    name = 'labels_%s_%s' % (start_date, end_date)
    dump_path = action_path + name + '.csv'
    if os.path.exists(dump_path):
        labels = pd.read_csv(dump_path)
    else:
        labelsAll = pd.read_csv(action_4_path)
        labels = labelsAll[(labelsAll.time >= start_date)& (labelsAll.time < end_date)]
        labels['user_id'] = labels['user_id'].astype('int')
        del labelsAll
        labels = labels[labels['type'] == 4]
        labels = labels.groupby(['user_id', 'sku_id'], as_index=False).sum()
        labels = labels[['user_id', 'sku_id']]
        labels['label'] = 1
        labels.to_csv(dump_path, mode='a',header=True, index=False)
    print 'get_labels finished'
    return labels


# In[12]:

#%%

def make_test_set(train_start_date, train_end_date):
    # 测试集
    # 前一个月中出现的所有用户商品对
    print "make_test_set start"
    name = 'store_set_%s_%s' % (train_start_date, train_end_date)
    dump_path = action_path + name + '.csv'
    path = action_path + 'test_set.csv'
    if os.path.exists(path):
        test_set = pd.read_csv(path)
    else:
        test_set = store_set(train_start_date, train_end_date)
        test_set = test_set.fillna(0)
        test_set.to_csv(path, mode='a',header=True, index=False)
    return test_set

#%%


def make_train_set(train_start_date, train_end_date, test_start_date, test_end_date, days=7):
    # 训练集
    print "make_train_set start"
    name = 'store_set_%s_%s' % (train_start_date, train_end_date)
    dump_path = action_path + name + '.csv'
    path = action_path + 'train_set.csv'
    
    if os.path.exists(path):
        train_set = pd.read_csv(path)
    else:
        #labels = get_labels(test_start_date, test_end_date)
        # generate 时间窗口
        # actions = get_accumulate_action_feat(train_start_date, train_end_date)
        train_set = store_set(train_start_date, train_end_date)
        train_labels = get_labels(test_start_date, test_end_date)
        train_labels = train_labels.convert_objects(convert_numeric=True)
        train_set = train_set.merge(train_labels, on=['user_id','sku_id'], how='left')
        del train_labels
        train_set = train_set.fillna(0)
        train_set.to_csv(path, mode='a',header=True, index=False)
    return train_set
    
    
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

    F11 = 6.0 * all_user_recall * all_user_acc /         (5.0 * all_user_recall + all_user_acc)
    F12 = 5.0 * all_item_acc * all_item_recall /         (2.0 * all_item_recall + 3 * all_item_acc)
    score = 0.4 * F11 + 0.6 * F12
    print 'F11=' + str(F11)
    print 'F12=' + str(F12)
    print 'score=' + str(score)

#%% 规则

def has_buyed(end_date):
    # 统计2、3、4月购买过第8类的用户
    print "has_buyed start"
    start_date = "2016-01-31"
    name = "has_buyed_%s_%s" % (start_date, end_date)
    dump_path = name + '.csv'
    if os.path.exists(dump_path):
        actions = pd.read_csv(dump_path)
    else:
        path = 'actions.csv'#get_actions(start_date,end_date)
        actionsAll = pd.read_csv(path, iterator=True)
        loop = True
        chunksize = 1000000
        chunks = []
        while loop:
            try:
                chunk = actionsAll.get_chunk(chunksize)
                chunk['user_id'] = chunk['user_id'].astype('int')
                chunk = chunk.sort_values(['user_id'])
                chunk = chunk[(chunk['type'] == 4) & (chunk['cate'] == 8)]
                if chunk.shape[0] != 0:
                    chunk = chunk.groupby(['user_id'], as_index=False).sum()
                    chunks.append(chunk)
            except StopIteration:
                loop = False
        actions = pd.concat(chunks)
        actions = actions.groupby(['user_id'], as_index=False).sum()
        actions['label'] = 1
        actions = actions[['user_id', 'label']]
        actions.to_csv(dump_path, mode='a', header=True, index=False)
    return actions


def del_has_buyed(result, end_date):
    # 删除2、3、4月已经购买过第8类的用于
    actions = has_buyed(end_date)
    
    insection_user = result.merge(actions, on='user_id', how='left')
    insection_user = insection_user.fillna(0)
    insection_user = insection_user[insection_user['label_y'] == 0]
    insection_user = insection_user[['user_id', 'sku_id', 'label_x']]
    insection_user.rename(columns={'label_x': 'label'}, inplace=True)


# In[13]:

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

    F11 = 6.0 * all_user_recall * all_user_acc /         (5.0 * all_user_recall + all_user_acc)
    F12 = 5.0 * all_item_acc * all_item_recall /         (2.0 * all_item_recall + 3 * all_item_acc)
    score = 0.4 * F11 + 0.6 * F12
    print 'F11=' + str(F11)
    print 'F12=' + str(F12)
    print 'score=' + str(score)


# In[14]:

#%% 规则

def has_buyed(end_date):
    # 统计2、3、4月购买过第8类的用户
    print "has_buyed start"
    start_date = "2016-01-31"
    name = "has_buyed_%s_%s" % (start_date, end_date)
    dump_path = basic_path + name + '.csv'
    if os.path.exists(dump_path):
        actions = pd.read_csv(dump_path)
    else:
        path = '../xlsx/action.csv'#get_actions(start_date,end_date)
        actionsAll = pd.read_csv(path, iterator=True)
        loop = True
        chunksize = 1000000
        chunks = []
        while loop:
            try:
                chunk = actionsAll.get_chunk(chunksize)
                chunk['user_id'] = chunk['user_id'].astype('int')
                chunk = chunk.sort_values(['user_id'])
                chunk = chunk[(chunk['type'] == 4) & (chunk['cate'] == 8)]
                if chunk.shape[0] != 0:
                    chunk = chunk.groupby(['user_id'], as_index=False).sum()
                    chunks.append(chunk)
            except StopIteration:
                loop = False
        actions = pd.concat(chunks)
        actions = actions.groupby(['user_id'], as_index=False).sum()
        actions['label'] = 1
        actions = actions[['user_id', 'label']]
        actions.to_csv(dump_path, mode='a', header=True, index=False)
    return actions


def del_has_buyed(result, end_date):
    # 删除2、3、4月已经购买过第8类的用户
    actions = has_buyed(end_date)
    
    insection_user = result.merge(actions, on='user_id', how='left')
    insection_user = insection_user.fillna(0)
    insection_user = insection_user[insection_user['label_y'] == 0]
    insection_user = insection_user[['user_id', 'sku_id', 'label_x']]
    insection_user.rename(columns={'label_x': 'label'}, inplace=True)
    return insection_user


# In[15]:

# train_start_date = '2016-03-31'
# train_end_date = '2016-04-11'

# test_start_date = '2016-04-11'
# test_end_date = '2016-04-16'

# sub_start_date = '2016-03-15'
# sub_end_date = '2016-04-16'

# train_set = make_train_set(train_start_date, train_end_date, test_start_date, test_end_date)
# print train_set.head()
# del train_set
# sub_training_data = make_test_set(sub_start_date, sub_end_date)
# print sub_training_data.head()
# del sub_training_data


# In[16]:

from sklearn.ensemble import RandomForestRegressor  
def RF_model():
    train_start_date = '2016-03-31'
    train_end_date = '2016-04-11'

    test_start_date = '2016-04-11'
    test_end_date = '2016-04-16'

    sub_start_date = '2016-03-15'
    sub_end_date = '2016-04-16'

    train_set = make_train_set(train_start_date, train_end_date, test_start_date, test_end_date)
    train_set = train_set.drop(['dt'], axis=1)
    train_set.replace([np.inf, -np.inf], 1, inplace=True)
    train_data = train_set.drop(['user_id', 'sku_id', 'label'], axis=1)
    label = train_set['label']
    del train_set
    # 使用默认参数
    rf=RandomForestRegressor(n_jobs=10)
    # 进行模型的训练 
    rf.fit(train_data[:],label[:])
    joblib.dump(rf, sub_path + '20170511model.pkl')
    print "Training model finished"


# In[17]:

def predict():
    sub_start_date = '2016-03-15'
    sub_end_date = '2016-04-16'
    rf = joblib.load(sub_path + '20170511model.pkl')
    sub_training_data = make_test_set(sub_start_date, sub_end_date)
    sub_training_data = sub_training_data.drop(['dt'], axis=1)
    sub_training_data.replace([np.inf, -np.inf], 1, inplace=True)
    # 去掉不是商品子集的用户商品对
    product = pd.read_csv(product_path)
    product['sku_id'] = product['sku_id'].astype(int)
    product = product[['sku_id', 'cate']]
    product.rename(columns={'cate':'cate_temp'}, inplace=True)
    sub_training_data = sub_training_data.merge(product, on=['sku_id'])
    sub_user_index = sub_training_data[['user_id', 'sku_id']]
    sub_training_data = sub_training_data.drop(['user_id', 'sku_id', 'cate_temp'], axis=1)
    result = rf.predict(sub_training_data)
    del sub_training_data
    sub_user_index['label'] = result
    sub_user_index = sub_user_index.sort_values(['label'], ascending=False)
# #     # 去掉以前购买过第8类的用户商品对
# #     sub_user_index = del_has_buyed(sub_user_index, sub_end_date)
#     sub_user_index['sku_id'] = sub_user_index['sku_id'].astype(int)
    
#     sub_user_index = sub_user_index.merge(product, on=['sku_id'])
    sub_user_index = sub_user_index[['user_id', 'sku_id', 'label']]
    sub_user_index = sub_user_index.groupby('user_id').first().reset_index()
    print len(sub_user_index)
    sub_user_index = sub_user_index.sort_values(['label'], ascending=False)
    print len(sub_user_index)
    sub_user_index.to_csv(sub_path + 'original.csv', index=False,index_label=False)
    print "Prediection has finished"
#     # 提交500个数据
#     pred = sub_user_index[:500]
#     del sub_user_index
#     pred = pred[['user_id', 'sku_id']]
#     pred['user_id'] = pred['user_id'].astype(int)
#     pred.to_csv(sub_path + 'submission.csv', index=False,index_label=False)
#     # del sub_training_data
#     # print pred.head()

# RF_model() 
# predict()


# In[25]:


def xgboost_make_submission():
    train_start_date = '2016-03-31'
    train_end_date = '2016-04-11'

    test_start_date = '2016-04-11'
    test_end_date = '2016-04-16'

    sub_start_date = '2016-03-15'
    sub_end_date = '2016-04-16'

    train_set = make_train_set(train_start_date, train_end_date, test_start_date, test_end_date)
    train_set = train_set.drop(['dt'], axis=1)
    # user_index = pd.read_csv(train_path[0])
    label = train_set['label']
    train_set = train_set.drop(['user_id', 'sku_id', 'label'],axis=1)
    
    X_train, X_test, y_train, y_test = train_test_split(train_set.values, label.values, test_size=0.2, random_state=0)
    del train_set
    del label
    dtrain = xgb.DMatrix(X_train, label=y_train)
    del X_train
    del y_train
    dtest = xgb.DMatrix(X_test, label=y_test)
    del X_test
    del y_test
    
    # scale_pos_weight正负样本不平衡
    param = {'learning_rate': 0.1, 'n_estimators': 1000, 'max_depth': 4,
             'min_child_weight': 5, 'gamma': 0, 'subsample': 1.0, 'colsample_bytree': 0.8,
             'scale_pos_weight': 217, 'eta': 0.05, 'silent': 1, 'objective': 'binary:logistic'}
    num_round = 283
    param['nthread'] = 4
    
    plst = param.items()
    plst += [('eval_metric','logloss')]
    evallist = [(dtest, 'eval'),(dtrain, 'train')]
#     bst = xgb.train(plst, dtrain, num_round, evallist)
    del dtest
    del dtrain
    del plst
    del evallist
    
#     joblib.dump(bst,'20170512model.pkl')
    
    sub_training_data = make_test_set(sub_start_date, sub_end_date)
    # 去掉不是商品子集的用户商品对
    product = pd.read_csv(product_path)
    product['sku_id'] = product['sku_id'].astype(int)
    product['temp'] = 1
    product = product[['sku_id', 'temp']]
    sub_training_data = sub_training_data.merge(product, on=['sku_id'])
    sub_user_index = sub_training_data[['user_id', 'sku_id']]
    sub_training_data = sub_training_data.drop(['user_id', 'sku_id', 'dt', 'temp'], axis=1)
    print sub_training_data.head()
    sub_training_data = xgb.DMatrix(sub_training_data.values)
    bst = joblib.load('20170512model.pkl')
    y = bst.predict(sub_training_data)
    del sub_training_data
    sub_user_index['label'] = y
    del y
    # 去掉以前购买过第8类的用户商品对
#     sub_user_index = del_has_buyed(sub_user_index, sub_end_date)
    # 去掉不是商品子集的用户商品对
#     product = pd.read_csv(product_path)
#     product['sku_id'] = product['sku_id'].astype(int)
#     sub_user_index = sub_user_index.merge(product, on=['sku_id'])
#     sub_user_index = sub_user_index[['user_id', 'sku_id', 'label']]
    sub_user_index = sub_user_index.sort_values(['label'], ascending=False)
    sub_user_index = sub_user_index.groupby('user_id').first().reset_index()
    sub_user_index = sub_user_index.sort_values(['label'], ascending=False)
    sub_user_index.to_csv('original_result_pro.csv', index=False,index_label=False)
    
    
#     #sub_user_index.reset_index()
#     # 提交500个数据
#     pred = sub_user_index[:500]
#     del sub_user_index
#     pred = pred[['user_id', 'sku_id']]
#     pred['user_id'] = pred['user_id'].astype(int)
#     pred.to_csv('1submission.csv', index=False,index_label=False)
    
    
#     pred = pred.sort_values(['label'])
#     pred = pred[['user_id','sku_id']]
#     pred = pred.groupby('user_id').first().reset_index()
#     pred['user_id'] = pred['user_id'].astype(int)

xgboost_make_submission()


# In[ ]:

if __name__ == '__main__':
    train_start_date = '2016-03-31'
    train_end_date = '2016-04-11'
    
    test_start_date = '2016-04-11'
    test_end_date = '2016-04-16'
    
    sub_train_start = '2016-04-09'
    sub_train_end = '2016-04-16'

