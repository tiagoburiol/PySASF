#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on jul 2024

@author: tiagoburiol
@coworker: buligonl; josue@sehnem.com
"""

import pandas as pd
import numpy as np

from pysasf import distances
'''
Just get basic infos and print
'''
def infos(bd):
    """
    Generate a DataFrame containing the sample sizes for each key in the input object.

    This function iterates through the dictionary of DataFrames contained in the
    `bd` object and creates a summary DataFrame that lists the number of samples
    (rows) for each DataFrame associated with the keys in `bd.df_dict`. The resulting
    DataFrame has the keys as index and the sample sizes as values.

    Parameters
    ----------
    bd : object
        An object that contains a dictionary of DataFrames in the attribute `df_dict`
        and a list of tracer names in the attribute `tracers`.

    Returns
    -------
    pd.DataFrame
        A DataFrame where the index represents the keys from `bd.df_dict` and the
        values represent the sample sizes (number of rows) for each corresponding DataFrame.

    Raises
    ------
    AttributeError
        If `bd` does not have the attributes `df_dict` or `tracers`.

    Notes
    -----
    #TODO: Need to provide more comments for these!

    """
    info = pd.DataFrame()
    for key in list(bd.df_dict):
        size = bd.df_dict[key].shape[0]
        values = np.full(len(bd.tracers),size).reshape(1,len(bd.tracers))
        df_info= pd.DataFrame(values, columns=bd.tracers)
        df_info = df_info.rename(index={0: key})
        info = pd.concat((info, df_info))
    info.columns.name = 'Sample Sizes'
    #display(info)
    return(info)

def means(bd):
    '''
    Calculates the mean values for each column in the dataframes stored in the bd.df_dict dictionary
    (average) value of each element in each source.

    Parameters
    --------------------
    bd : object
        An instance of a BasinData class Basin. Contains a dictionary of dataframes, bd.df_dict.

    Returns
    --------------------
    means : pandas.DataFrame
        A dataframe containing the mean values for each column in the dataframes stored in bd.df_dict
        (average value of each element in each source -- average of the element value of all samples from the same source).

    Raises
    --------------------
    None
    '''
    means = pd.DataFrame()
    for key in list(bd.df_dict):
        df_mean = bd.df_dict[key].mean(axis=0).to_frame().transpose()
        df_mean = df_mean.rename(index={0: key})
        means = pd.concat((means, df_mean))
    means.columns.name = 'Means'
    #display(means.round(2))
    return(means.astype(float).round(2))

           
def std(bd):
    '''
    Calculates the standard deviation values for each column in the dataframes stored in the bd.df_dict dictionary.

    Parameters
    --------------------
    bd : object
        An instance of a BasinData class Basin. Contains a dictionary of dataframes, bd.df_dict.

    Returns
    --------------------
    std : pandas.DataFrame
        A dataframe containing the standard deviation values for each column in the dataframes stored in bd.df_dict.

    Raises
    --------------------
    None
    '''
    std = pd.DataFrame()
    for key in list(bd.df_dict):
        df_std = bd.df_dict[key].std(axis=0).to_frame().transpose()
        df_std = df_std.rename(index={0: key})
        std = pd.concat((std, df_std))
    std.columns.name = 'STD'
    #display(std.round(2))
    return(std.astype(float).round(2))


def randon_props_subsamples(bd, key, n, only_feasebles=False, target=None):
    '''
    Selects a random subsample of n rows from the dataframe stored in bd.df_dict[key], and returns the corresponding rows from the bd.combs and bd.props arrays.

    Parameters
    --------------------
    bd : object
        An instance of a class that contains a dictionary of dataframes (bd.df_dict), a combinations array (bd.combs), and a properties array (bd.props).
    key : str
        The key corresponding to the dataframe in bd.df_dict from which the subsample will be selected.
    n : int
        The number of rows to select in the subsample.
    only_feasibles : bool, optional
        If True, only selects rows that are feasible (i.e., where bd.feas is True).Defaults to False.

    Returns
    --------------------
    selected_combs : numpy.ndarray
        A 2D numpy array containing the rows from bd.combs that correspond to the selected subsample.
    selected_Ps : numpy.ndarray
        A 1D numpy array containing the rows from bd.props that correspond to the selected subsample.

    Raises
    --------------------
    None
    '''
    Ps = bd.props
    combs = bd.combs
    key_idx = list(bd.df_dict.keys()).index(key)
    size = len(bd.df_dict[key])
    rand = np.random.choice(np.arange(size), n, replace=False)

    # if 'bytarget' pick only the target selected
    targets_idx = list(bd.df_dict.keys()).index('Y')
    if target!=None:
        Ps = bd.props[np.where(np.isin(combs[:,targets_idx],target))]
        combs = bd.combs[np.where(np.isin(combs[:,targets_idx],target))]
        feas = bd.feas[np.where(np.isin(combs[:,targets_idx],target))]
    else:
        feas = bd.feas
    
    if only_feasebles == False:
        selected_combs = combs[np.where(np.isin(combs[:,key_idx],rand))]
        selected_Ps = Ps[np.where(np.isin(combs[:,key_idx],rand))]
    else:
        #print('randon_props_subsamples->only_feasebles')
        selected_combs = combs[np.where(np.isin(combs[:,key_idx],rand) & feas)]
        selected_Ps = Ps[np.where(np.isin(combs[:,key_idx],rand) & feas)]
    return selected_combs, selected_Ps

'''
Calculate the confidence region
'''    
def confidence_region(P, p = 95, space_dist='mahalanobis'):
    '''
    Calculates the confidence region for the given set of points P, based on the specified confidence level
    (default is 95%) and distance metric (default is 'mahalanobis').

    Parameters
    --------------------
    P : numpy.ndarray
        A 2D numpy array containing the set of points for which the confidence region is to be calculated.
    p : {int, optional}
        The confidence level, expressed as a percentage. Defaults to 95.
    space_dist : str, optional
        The distance metric to use for calculating the confidence region. Can be either 'mahalanobis' or 'mahalanobis2d'. Defaults to 'mahalanobis'.

    Returns
    --------------------
    Psorted_cropped : numpy.ndarray: A numpy array containing the points that fall within the specified confidence region, within the mahalanobis distance metric.

    Raises
    --------------------
    None
    '''
    P0 = np.mean(P, axis=0)
    #print('Pm=',Pm)
    if space_dist=='mahalanobis':
        dist = distances.mahalanobis_dist(P, P0)
    elif space_dist=='mahalanobis2d':
        dist = distances.mahalanobis2d_dist(P, P0)
    else:
        print('The confidence region space distance',space_dist,'not avaliable')

    
    sorted_idx = np.argsort(dist)
    Psorted = P[sorted_idx]

    # em ordem crescente
    end_idx = int((p/100)*len(Psorted))
    #print ("Os 95% mais próximos:", Psorted[:,:end_idx])
    return (Psorted[:end_idx,:])


def random_subsamples(bd,nlist):
    '''
    Get a random subsample set
    n_list is a list containing a number of elements of each source
    '''
    df_dict = bd.df_dict
    ss_dict = {}
    for i, key in enumerate(df_dict.keys()):
        data = df_dict[key].values.astype(float)
        ss_dict[key]=data[np.random.choice(len(data), nlist[i], replace=False)]
    return ss_dict

'''
Get a randon props subset
'''
def random_props_subset(arr,n):
    '''
    Get a random props subset
    '''
    subset = arr[np.random.choice(len(arr), n, replace=False)]
    return subset
