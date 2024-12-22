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
def minmax_scale(bd):
    """
    This function is currently not in use.
    Scale the features of a dataset to the range [0, 1] using Min-Max scaling.

    Parameters
    --------------------
    bd : object
        An object containing a dictionary of DataFrames in its `df_dict` attribute.

    Returns
    --------------------
    bd_n : object
        A new object with the same structure as `bd`, but with the DataFrames in `df_dict` scaled to the range [0, 1] using Min-Max scaling.

    Raises
    --------------------
    None
    """
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
