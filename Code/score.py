#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2017-04-28 10:06:40
# @Author  : Xielong (xielong@zju.edu.cn)
# @Link    : https://github.com/XieLongZju
# @Version : V0.1

# coding: utf-8

# predata与trudata只包含user_id和sku_id两列
def score(predata, truedata):
    print "score start"
    subnums = predata.shape[0]
    allnums = truedata.shape[0]
    insection_user = predata.merge(truedata, on='user_id')
    insection_user_sku = predata.merge(truedata, on=['user_id', 'sku_id'])
    correct_user_sum = insection_user.shape[0]
    correct_user_sku_num = insection_user_sku.shape[0]

    # F11查准率和召回率
    F11_p = 1.0 * correct_user_sum / subnums
    F11_r = 1.0 * correct_user_sum / allnums
    # F11总分
    F11 = 6.0 * F11_r * F11_p / (5 * F11_r + F11_p)
    print "F11：" + str(F11)

    # F12查准率和召回率
    F12_p = 1.0 * correct_user_sku_num / subnums
    F12_r = 1.0 * correct_user_sku_num / allnums
    # F12总分
    F12 = 5.0 * F12_r * F12_p / (2 * F12_r + 3 * F12_p)
    print "F12：" + str(F12)
    # 总分
    score = 0.4 * F11 + 0.6 * F12
    print "总分：" + str(score)
