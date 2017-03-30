#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2017-03-30 05:42:31
# @Author  : Xielong (xielong@zju.edu.cn)
# @Link    : https://github.com/XieLongZju
# @Version : V0.1

import pandas as pd

user = pd.read_csv('JData_User.csv', encoding='gbk',
                   parse_dates=['user_reg_dt'])
user.loc[user["age"] == "-1", "age"] = 0
user.loc[user["age"] == u'56\u5c81\u4ee5\u4e0a', "age"] = 55
user.loc[user["age"] == u'46-55\u5c81', "age"] = 50
user.loc[user["age"] == u'36-45\u5c81', "age"] = 40
user.loc[user["age"] == u'26-35\u5c81', "age"] = 30
user.loc[user["age"] == u'16-25\u5c81', "age"] = 20
user.loc[user["age"] == u'15\u5c81\u4ee5\u4e0b', "age"] = 15
user.loc[user["age"] == u'56\u5c81\u4ee5\u4e0a', "age"] = 55
user['age'] = user['age'].astype('int')
print user.describe()
