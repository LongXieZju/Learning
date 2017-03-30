#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2017-03-30 17:12:46
# @Author  : Xielong (xielong@zju.edu.cn)
# @Link    : https://github.com/XieLongZju
# @Version : $Id$

import pandas as pd 
commentAll = pd.read_csv('JData_Comment.csv', encoding='gbk', chunksize=size)
