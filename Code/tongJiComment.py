#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2017-03-30 17:12:46
# @Author  : Xielong (xielong@zju.edu.cn)
# @Link    : https://github.com/XieLongZju
# @Version : V0.1

import pandas as pd

chunksize = 100
commentAll = pd.read_csv('JData_Comment.csv', encoding='gbk')
# comment = commentAll.get_chunk(chunksize)
print commentAll.describe()