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

import distances
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

'''
Calculate de confidence region
'''    
def confidence_region(P, p = 95, spacedist= 'mahalanobis'):
        if P.shape[0]>2:
            P=P[0:2,:]
        Pm = np.mean(P, axis=1)
        if spacedist=='mahalanobis':
            dist = distances.mahalanobis_dist(P, Pm)
        if spacedist=='euclidean':
            dist = distances.euclidean_dist(P)
        
        #dist = self.mahalanobis_dist(P, Pm)
        sorted_idx = np.argsort(dist)
        Psorted = P.T[sorted_idx].T
        end_idx = int((p/100)*len(Psorted.T))
        return (Psorted[:,:end_idx])

