#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2017-04-30 13:00:33
# @Author  : Xielong (xielong@zju.edu.cn)
# @Link    : https://github.com/XieLongZju
# @Version : V1.0

# coding: utf-8

import pandas as pd
from datetime import datetime
from datetime import timedelta
import matplotlib.pyplot as plt

#%%
def actions():
    # 获得2016-03-22到2016-04-16的数据
    start_data = '2016-03-22'
    end_data = '2016-04-16'
    path = 'actions_%s_%s.csv' % (start_data, end_data)
    All = pd.read_csv('F:/Competition/JData/Learning/Analysis/action.csv', iterator=True)
    chunksize = 1000000
    loop = True
    flag = 1
    while loop:
        try:
            chunk = All.get_chunk(chunksize)
            chunk = chunk[(chunk.time >= start_data) & (chunk.time < end_data)]
            if chunk.shape[0] != 0:
                if flag == 1:
                    chunk.to_csv('F:/Competition/JData/Learning/Analysis/'+path, mode='a',header=True, index=False)
                    flag = 2
                else:
                    chunk.to_csv('F:/Competition/JData/Learning/Analysis/'+path, mode='a',header=False, index=False)
        except StopIteration:
            loop = False

#%%
start_data = '2016-03-22'
end_data = '2016-04-16'
action_4_path = 'F:/Competition/JData/Learning/Analysis/actions_%s_%s.csv' % (start_data, end_data)
path = 'F:/Competition/JData/DataSet/'
# action_4_path = path + 'JData_Action_201604.csv'
product_path = path + 'JData_Product.csv'
actionsAll = pd.read_csv(action_4_path)
actionsAll['user_id'] = actionsAll['user_id'].astype('int')
# 获取最后五天用户购买过第8类商品的用户商品对
UIP = pd.DataFrame()
UIP = actionsAll[(actionsAll['type'] == 4) & (actionsAll['cate'] == 8)]
UIP = UIP[(UIP.time >= '2016-04-11') & (UIP.time < '2016-04-16')]
UIP = UIP.groupby(['user_id', 'sku_id'], as_index=False).sum()
# 最后五天用户购买的商品在商品子集中
# 最后五天购买过商品子集商品的用户商品对为正样本
productAll = pd.read_csv(product_path)
productAll['label'] = 1
insection = pd.merge(UIP, productAll, on=['sku_id'], how='left')
del productAll
insection = insection.fillna(0)
insection = insection[insection['label'] == 1]
insection = insection[['user_id', 'sku_id', 'label']]
# 前1-20天用户交互样本中为考察日正样本数
end_date = '2016-04-11'
UIPAll = []
UIPBuy = []
UIPBrowse = []
UIPAdd = []
UIPDelete = []
UIPFollow = []
UIPClick = []
for i in range(20):
        start_days = datetime.strptime(end_date, '%Y-%m-%d') - timedelta(days=i+1)
        start_days = start_days.strftime('%Y-%m-%d')
        print start_days
        temp = actionsAll[(actionsAll.time >= start_days)& (actionsAll.time < end_date)]
        browse = temp[temp['type'] == 1]
        browse = browse.groupby(['user_id', 'sku_id'], as_index=False).sum()
        num = browse.shape[0]
        print browse.shape[0]
        browse = browse.merge(insection, on=['user_id', 'sku_id'])
        print browse.shape[0]
        UIPBrowse.append(1.0*browse.shape[0]/num)
        del browse
        add = temp[temp['type'] == 2]
        add = add.groupby(['user_id', 'sku_id'], as_index=False).sum()
        num = add.shape[0]
        #print num
        add = add.merge(insection, on=['user_id', 'sku_id'])
        UIPAdd.append(1.0*add.shape[0]/num)
        del add
        delete = temp[temp['type'] == 3]
        delete = delete.groupby(['user_id', 'sku_id'], as_index=False).sum()
        num = delete.shape[0]
        #print num
        delete = delete.merge(insection, on=['user_id', 'sku_id'])
        UIPDelete.append(1.0*delete.shape[0]/num)
        del delete
        follow = temp[temp['type'] == 5]
        follow = follow.groupby(['user_id', 'sku_id'], as_index=False).sum()
        num = follow.shape[0]
        follow = follow.merge(insection, on=['user_id', 'sku_id'])
        UIPFollow.append(1.0*follow.shape[0]/num)
        del follow
        click = temp[temp['type'] == 6]
        click = click.groupby(['user_id', 'sku_id'], as_index=False).sum()
        num = click.shape[0]
        #print num
        click = click.merge(insection, on=['user_id', 'sku_id'])
        UIPClick.append(1.0*click.shape[0]/num)
        del click
        UIPAll.append(temp.shape[0])
        has_buyed = temp.merge(insection, on=['user_id', 'sku_id'])
        UIPBuy.append(has_buyed.shape[0])
        #print temp.shape[0]
        #print has_buyed.shape[0]
        
plt.plot(UIPAll,'b*--')
plt.plot(UIPBuy,'b*--')
plt.plot(UIPBrowse,'b*--')
plt.plot(UIPAdd,'y*-')
plt.plot(UIPDelete,'g*-.')
plt.plot(UIPFollow,'r*-')
plt.plot(UIPClick,'k*-')
# 前1-20天购买转换率
end_date = '2016-04-11'
SUIPAll = []
SUIPBuy = []
SUIPBrowse = []
SUIPAdd = []
SUIPDelete = []
SUIPFollow = []
SUIPClick = []
for i in range(20):
        start_days = datetime.strptime(end_date, '%Y-%m-%d') - timedelta(days=i+1)
        end_days = datetime.strptime(end_date, '%Y-%m-%d') - timedelta(days=i)
        start_days = start_days.strftime('%Y-%m-%d')
        end_days = end_days.strftime('%Y-%m-%d')
        print start_days
        temp = actionsAll[(actionsAll.time >= start_days) & (actionsAll.time <= end_days)]
        browse = temp[temp['type'] == 1]
        browse = browse.groupby(['user_id', 'sku_id'], as_index=False).sum()
        num = browse.shape[0]
        # print browse.shape[0]
        browse = browse.merge(insection, on=['user_id', 'sku_id'])
        # print browse.shape[0]
        SUIPBrowse.append(1.0*browse.shape[0]/num)
        del browse
        add = temp[temp['type'] == 2]
        add = add.groupby(['user_id', 'sku_id'], as_index=False).sum()
        num = add.shape[0]
        #print num
        add = add.merge(insection, on=['user_id', 'sku_id'])
        SUIPAdd.append(1.0*add.shape[0]/num)
        del add
        delete = temp[temp['type'] == 3]
        delete = delete.groupby(['user_id', 'sku_id'], as_index=False).sum()
        num = delete.shape[0]
        #print num
        delete = delete.merge(insection, on=['user_id', 'sku_id'])
        SUIPDelete.append(1.0*delete.shape[0]/num)
        del delete
        follow = temp[temp['type'] == 5]
        follow = follow.groupby(['user_id', 'sku_id'], as_index=False).sum()
        num = follow.shape[0]
        follow = follow.merge(insection, on=['user_id', 'sku_id'])
        SUIPFollow.append(1.0*follow.shape[0]/num)
        del follow
        click = temp[temp['type'] == 6]
        click = click.groupby(['user_id', 'sku_id'], as_index=False).sum()
        num = click.shape[0]
        #print num
        click = click.merge(insection, on=['user_id', 'sku_id'])
        SUIPClick.append(1.0*click.shape[0]/num)
        del click
        SUIPAll.append(temp.shape[0])
        has_buyed = temp.merge(insection, on=['user_id', 'sku_id'])
        SUIPBuy.append(has_buyed.shape[0])
        del temp
        del has_buyed
        #print temp.shape[0]
        #print has_buyed.shape[0]
        
plt.plot(SUIPAll,'b*--')
plt.plot(SUIPBuy,'b*--')
plt.plot(SUIPBrowse,'b*--')
plt.plot(SUIPAdd,'y*-')
plt.plot(SUIPDelete,'g*-.')
plt.plot(SUIPFollow,'r*-')
plt.plot(SUIPClick,'k*-')
