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

from pysasf import distances
'''
Just get basic infos and print
'''
def infos(bd):
    info = pd.DataFrame()
    for key in list(bd.df_dict):
        size = bd.df_dict[key].shape[0]
        values = np.full(len(bd.cols),size).reshape(1,len(bd.cols))
        df_info= pd.DataFrame(values, columns=bd.cols)
        df_info = df_info.rename(index={0: key})
        info = pd.concat((info, df_info))
    info.columns.name = 'Sample Sizes'
    #display(info)
    return(info)

def means(bd):
    means = pd.DataFrame()
    for key in list(bd.df_dict):
        df_mean = bd.df_dict[key].mean(axis=0).to_frame().transpose()
        df_mean = df_mean.rename(index={0: key})
        means = pd.concat((means, df_mean))
    means.columns.name = 'Means'
    #display(means.round(2))
    return(means.astype(float).round(2))

           
def std(bd):
    std = pd.DataFrame()
    for key in list(bd.df_dict):
        df_std = bd.df_dict[key].std(axis=0).to_frame().transpose()
        df_std = df_std.rename(index={0: key})
        std = pd.concat((std, df_std))
    std.columns.name = 'STD'
    #display(std.round(2))
    return(std.astype(float).round(2))


def randon_props_subsamples(bd, key, n, only_feasebles=False):
    Ps = bd.props
    combs = bd.combs
    key_idx = list(bd.df_dict.keys()).index(key)
    size = len(bd.df_dict[key])
    rand = np.random.choice(np.arange(size), n, replace=False)
    
    if only_feasebles == False:
        selected_combs = combs[np.where(np.isin(combs[:,key_idx],rand))]
        selected_Ps = Ps[np.where(np.isin(combs[:,key_idx],rand))]
    else:
        #print('randon_props_subsamples->only_feasebles')
        selected_combs = combs[np.where(np.isin(combs[:,key_idx],rand)  & bd.feas)]
        selected_Ps = Ps[np.where(np.isin(combs[:,key_idx],rand) & bd.feas)]
    return selected_combs, selected_Ps

'''
Calculate de confidence region
'''    
def confidence_region(P, p = 95, space_dist='mahalanobis'):
    Pm = np.mean(P, axis=0)
    if space_dist=='mahalanobis':
        dist = distances.mahalanobis_dist(P, Pm)
    elif space_dist=='mahalanobis0':
        dist = distances.mahalanobis0_dist(P, Pm)
    else:
        print('The confidence region space distance',space_dist,'not avaliable')

    
    sorted_idx = np.argsort(dist)
    Psorted = P[sorted_idx]

    # em ordem crescente
    end_idx = int((p/100)*len(Psorted))
    #print ("Os 95% mais próximos:", Psorted[:,:end_idx])
    return (Psorted[:end_idx,:])


