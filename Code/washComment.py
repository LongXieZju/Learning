#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2017-03-30 17:10:22
# @Author  : Xielong (xielong@zju.edu.cn)
# @Link    : https://github.com/XieLongZju
# @Version : $Id$

import pandas as pd

# 配置
size = 100000
# 读取文件
commentAll = pd.read_csv('JData_Comment.csv', encoding='gbk', chunksize=size)
# 处理
for user in userAll:

