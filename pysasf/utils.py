#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on jul 2024

@author: tiagoburiol
@coworker: buligonl; josue@sehnem.com
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from random import randrange
from scipy.linalg import solve
import seaborn as sns
from matplotlib.legend import Legend
import h5py

#https://learner-cares.medium.com/handy-python-pandas-for-data-normalization-and-scaling-9658846de8fc

def minmax_scale(bd):
    from sklearn.preprocessing import MinMaxScaler
    import copy
    
    bd_n = copy.deepcopy(bd)
    df_dict = bd.df_dict
    # numero de fontes
    n = len(df_dict.keys())
    df_norm_dict = {}
    # para cada fonte
    for key in df_dict.keys():
        # pega o dataframe
        df = df_dict[key]
        # normaliza a escala 
        scaler = MinMaxScaler()
        df_norm = scaler.fit_transform(df)
        # pega o nome das colunas do dataframe
        cols = df_dict[key].columns
        # monta o dataframe normalizado
        df_norm = pd.DataFrame(df_norm, columns=cols)
        df_norm_dict[key] = df_norm
    bd_n.df_dict = df_norm_dict
    return(bd_n)
