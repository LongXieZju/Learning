#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2017-03-30 20:51:33
# @Author  : Xielong (xielong@zju.edu.cn)
# @Link    : https://github.com/XieLongZju
# @Version : V0.1

import pandas as pd

# 配置
size = 100000
# 读取文件
action3All = pd.read_csv('JData_Action_201603.csv',
                         encoding='gbk', chunksize=size)
for action in action3All:
    action.to_csv('action3.csv', mode='a', header=False, index=False, columns=[
                  'user_id', 'sku_id', 'type', 'cate', 'brand'])
