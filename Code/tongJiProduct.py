#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2017-03-30 05:42:31
# @Author  : Xielong (xielong@zju.edu.cn)
# @Link    : https://github.com/XieLongZju
# @Version : V0.1

import pandas as pd
productAll = pd.read_csv('JData_Product.csv', encoding='gbk', iterator=True)
product = productAll.get_chunk(100)
print product
print product.describe()
