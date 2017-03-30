#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2017-03-30 05:42:31
# @Author  : Xielong (xielong@zju.edu.cn)
# @Link    : https://github.com/XieLongZju
# @Version : V0.1

# coding: gbk

import pandas as pd
# 配置
size = 1000
loop = True
# 读取文件
productAll = pd.read_csv('JData_Product.csv', encoding='gbk', chunksize=size)
# 处理
for product in productAll:
    product.loc[product["attr1"] == -1, "attr1"] = 0
    product.loc[product["attr2"] == -1, "attr2"] = 0
    product.loc[product["attr3"] == -1, "attr3"] = 0
    product.to_csv('product.csv', mode='a', header=False, index=False)
