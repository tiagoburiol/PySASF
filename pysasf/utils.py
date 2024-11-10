#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on jul 2024

@author: tiagoburiol
@coworker: buligonl; josue@sehnem.com
"""

import pandas as pd
import numpy as np




#https://learner-cares.medium.com/handy-python-pandas-for-data-normalization-and-scaling-9658846de8fc
'''
não esá sendo usado para nada
'''
def minmax_scale(bd):
    from sklearn.preprocessing import MinMaxScaler
    import copy
    
    bd_n = copy.deepcopy(bd)
    df_dict = bd.df_dict
    df_norm_dict = {}
    for key in df_dict.keys():
        df = df_dict[key]
        scaler = MinMaxScaler()
        df_norm = scaler.fit_transform(df)
        cols = df_dict[key].columns
        df_norm = pd.DataFrame(df_norm, columns=cols)
        df_norm_dict[key] = df_norm
    bd_n.df_dict = df_norm_dict
    return(bd_n)
