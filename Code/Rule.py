#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2017-04-28 13:45:40
# @Author  : Xielong (xielong@zju.edu.cn)
# @Link    : https://github.com/XieLongZju
# @Version : V1.0

# coding: utf-8

import pandas as pd
import os
from gen_feat import get_actions

def has_buyed(end_date):
    # 统计2、3、4月购买过第8类的用户
    print "has_buyed start"
    start_date = "2016-01-31"
    name = "has_buyed_%s_%s" % (start_date, end_date)
    dump_path = name + '.csv'
    if os.path.exists(dump_path):
        actions = pd.read_csv(dump_path)
    else:
        path = get_actions(start_date,end_date)
        actionsAll = pd.read_csv(path, iterator=True)
        loop = True
        chunksize = 1000000
        chunks = []
        while loop:
            try:
                chunk = actionsAll.get_chunk(chunksize)
                chunk['user_id'] = chunk['user_id'].astype('int')
                chunk = chunk.sort_values(['user_id'])
                chunk = chunk[(chunk['type'] == 4) & (chunk['cate'] == 8)]
                if chunk.shape[0] != 0:
                    chunk = chunk.groupby(['user_id'], as_index=False).sum()
                    chunks.append(chunk)
            except StopIteration:
                loop = False
        actions = pd.concat(chunks)
        actions = actions.groupby(['user_id'], as_index=False).sum()
        actions['label'] = 1
        actions = actions[['user_id', 'label']]
        actions.to_csv(dump_path, mode='a', header=True, index=False)
    return actions


def del_has_buyed(result, end_date):
    # 删除2、3、4月已经购买过第8类的用于
    actions = has_buyed(end_date)
    
    insection_user = result.merge(actions, on='user_id', how='left')
    insection_user = insection_user.fillna(0)
    insection_user = insection_user[insection_user['label'] == 0]
    insection_user = insection_user[['user_id', 'sku_id']]
    return insection_user
    
    
    