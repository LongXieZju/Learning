#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2017-03-30 20:24:32
# @Author  : Xielong (xielong@zju.edu.cn)
# @Link    : https://github.com/XieLongZju
# @Version : V0.1

import pandas as pd

# 配置
size = 100000
# 读取文件
action2All = pd.read_csv('JData_Action_201602.csv',
                         encoding='gbk', chunksize=size)
for action in action2All:
    action.to_csv('action2.csv', mode='a', header=False, index=False, columns=[
                  'user_id', 'sku_id', 'type', 'cate', 'brand'])
