#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2017-03-30 17:48:10
# @Author  : Xielong (xielong@zju.edu.cn)
# @Link    : https://github.com/XieLongZju
# @Version : V0.1

import pandas as pd
chunksize = 100
# action2All = pd.read_csv("JData_Action_201602.csv", encoding="gbk")
# print action2All.describe()
action2All = pd.read_csv("JData_Action_201602.csv",
                         encoding="gbk", iterator=True)
action = action2All.get_chunk(chunksize)
print action.describe()
