#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2017-03-30 05:42:31
# @Author  : Xielong (xielong@zju.edu.cn)
# @Link    : https://github.com/XieLongZju
# @Version : V0.1
# 备注     ：循环不输出列名，需要手动添加列名

# coding: utf-8

import pandas as pd
import datetime

# 配置
size = 100000
# 读取文件
userAll = pd.read_csv('JData_User.csv', encoding='gbk',
                      parse_dates=['user_reg_dt'], chunksize=size)
# 注册基准时间
baseTime = datetime.datetime.strptime('2010-01-01', "%Y-%m-%d")
# 处理
for user in userAll:
    # user = userAll.get_chunk(chunksize)
    # 年龄
    user.loc[user["age"] == "-1", "age"] = 30
    user.loc[user["age"] == u'56\u5c81\u4ee5\u4e0a', "age"] = 55
    user.loc[user["age"] == u'46-55\u5c81', "age"] = 50
    user.loc[user["age"] == u'36-45\u5c81', "age"] = 40
    user.loc[user["age"] == u'26-35\u5c81', "age"] = 30
    user.loc[user["age"] == u'16-25\u5c81', "age"] = 20
    user.loc[user["age"] == u'15\u5c81\u4ee5\u4e0b', "age"] = 15
    user.loc[user["age"] == u'56\u5c81\u4ee5\u4e0a', "age"] = 55
    user['age'] = user['age'].astype('int')
    # 注册时间
    user['user_reg_dt'] = user['user_reg_dt'].astype('str')
    user['user_reg_dt'] = user['user_reg_dt'].apply(lambda x: (
        datetime.datetime.strptime(x, "%Y-%m-%d") - baseTime).days)
    # print user.describe()
    # user.to_csv('user.csv', mode='a', header=False, index=False)
