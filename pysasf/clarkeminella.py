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
from IPython.display import clear_output
from scipy.spatial import ConvexHull
import concurrent.futures
import time


#PySASF imports
from pysasf import solvers
from pysasf import stats


#def confidence_region(P, p = 95, space_dist='mahalanobis0'):
#    d = space_dist
#    return stats.confidence_region(P, p = 95, space_dist=d)
    
#def cm_feasebles(Ps):
    # '''
    # Checks for feasible solutions in a larger set of solutions.
    #
    # Parameters
    # --------------------
    # Ps : numpy.ndarray
    #     A list of proportions from a certain source, each row represents a point in the solution space.
    #
    # Returns
    # --------------------
    # Ps_feas : numpy.ndarray
    #     A 2D array containing only the feasible solutions, in which all values are greater than zero (the basin can't have a negative amount of a certain element).
    #
    # Raises
    # --------------------
    # None
    # '''
#    Ps_feas = Ps[[np.all(P>0) for P in Ps]]
#    return np.array(Ps_feas)

#########################################################################################3


def run_repetitions_and_reduction (bd, key, reductions, percents = False,
                                   repetitions = 50, target = None):
    '''
    Runs multiple repetitions of subsampling and computes statistical measures for these samples.

    Parameters
    --------------------
    bd : object
        Basin dataset.

    key : str
        The key identifying the specific data source to be analyzed (C,E,L,Y).

    reductions : list
        A list of integers representing the number of subsamples to be taken in each reduction.

    repetitions : int, optional
        The number of repetitions for each reduction (default is 50).

    Returns
    --------------------
    df_out : pandas.DataFrame

    None
        The function updates the internal state of the `bd` object with computed statistics.

    Raises
    --------------------
    None
    '''
    inicio = time.time()
    cv = lambda x: np.std(x) / np.mean(x) *100
    CVs = []
    areas_medias = []
    filename = bd.filename+'_'+str(key)
    
    # create the columns names for return dataframe
    df_out_cols = ['nSamp','CV','Mean','Std', 'Total', 'Feas']
    nSources = len(bd.sources)
    for i in range(nSources):
        df_out_cols.append('MeanP'+str(i+1))
    df_out_data = []


    # apenas para calcular o numero total de Ps (Melhorar isso)
    prod =1
    for k in bd.sources:
        if k!=key:
            prod = prod*len(bd.df_dict[k])

    #calcula as reduções com base nos percentuais
    if percents == True:
        percents_reductions = reductions 
        ssize = len(bd.df_dict[key])
        percent = ssize/100
        reductions = np.round(percent*np.array(reductions))
        reductions = list(reductions.astype(int))
        print ("Number of samples:", key, reductions)

    def compute_area(pts):
        '''
        Forms a ConvexHull around the points

        Parameters
        --------------------
        pts :
            points around which you want to draw the Convex Hull and calculate its area.

        Returns
        --------------------
        area : float
            float value that represents the area of the Convex Hull that encompasses the desired set of points.

        Raises
        --------------------
        TypeError
            If pts is not a set of numbers.
        '''
        hull = ConvexHull(pts)
        area = hull.volume # area for 2d points
        if isinstance(pts, (str, int, float)):
            raise TypeError("Input pts must be a set of points.")
        return area
    
    
    #AQUI ACONTECEM AS REDUÇÕES
    for n in reductions:
        #Total=prod*n
        points_set = []
        areas = []
        filename = filename+'-'+str(n)

        t1 = time.time()
        for i in range(repetitions):
            # apenas para imprimir o texto
            t2 = time.time()
            if t2-t1>0.5:
                #clear_output(wait=True)
                print ('Processing for', n, 'subsamples of',key, 
                   ', repetition number', i+1,end='\r', flush=True)
                t1=t2

            
            #_,Ptot = stats.randon_props_subsamples(bd, key, n, only_feasebles=False)
            #Pfea =  cm.cm_feasebles(Ptot[:,0:2])
            _,Pfea = stats.randon_props_subsamples(bd, key, n, 
                                                   only_feasebles=True, target=target)

            
            if Pfea.shape[0]>=4:
                Pcr = stats.confidence_region(Pfea[:,0:2], p = 95)
                #<<-------------------------------
                #hull = ConvexHull(Pcr[:,0:2])    #Aqui
                #areas.append(hull.volume)
                #>>-------------------------------
                points_set.append(Pcr)
                #-------------------------------
       
        #Aqui: otimização/paralelização
        #>>-------------------------------
        with concurrent.futures.ThreadPoolExecutor() as executor:
             areas = list(executor.map(compute_area, points_set)) 
        #-------------------------------

        
        # insert data for df_out
        if percents == True:
            idx = np.argwhere(reductions == n)[0][0]
            #print('IDX',idx)
            nSamp = percents_reductions[idx]
        else:
            nSamp = n
            
        CV = np.round(cv(areas),4)
        Mean = np.round(np.mean(areas),4)
        Std = np.round(np.std(areas),4)
        Total=prod*n
        Feas = len(Pfea)
        df_out_data_n = [nSamp,CV,Mean,Std,Total,Feas]
        for i in range(nSources):
            df_out_data_n.append(np.mean(Pfea, axis=0)[i])
        df_out_data.append(df_out_data_n)
        bd.cm_df = df_out_data

        CVs.append(cv(areas))
        areas_medias.append(np.mean(areas))

    # Clean the terminal and print some infos
    print('Done!')
    clear_output(wait=True)
    fim = time.time()
    print ("Time for all runs:",fim-inicio)

    # create return dataframe
    df_out = pd.DataFrame(df_out_data, columns=df_out_cols)
    bd.cm_df = df_out

    # Saving file in cvs
    print ('Saving in', filename+'.csv')
    df_out.to_csv(bd.output_folder+'/'+filename+'.csv')
    return (df_out)
###########################################################################################3



#'''
#Esta função processa o cálculo das porcentagens usando as médias
#Faz as contas usando reps repetições tomando aletoriamente um conjunto
#de subamostras com tamanhos definidos por n_list. Tem opção de plotar. 
#'''

#def random_subsamples(bd,nlist):
    # '''
    # Computes proportions from random subsamples by calculating their means and optionally plots the results.
    #
    # Parameters
    # --------------------
    # bd : object
    #     An object containing the data sources and related methods for processing, including a dictionary of dataframes.
    #
    # reps : int
    #     The number of repetitions for generating random subsamples.
    #
    # n_list : list
    #     A list of integers specifying the sizes of the subsamples to be taken from the data. The length of this list must match the number of keys in `df_dict`.
    #
    # plot : bool, optional
    #     A flag indicating whether to plot the results (default is True).
    #
    # Returns
    # --------------------
    # numpy.ndarray
    #     An array of properties calculated from the subsamples, where each row corresponds to a repetition and contains the calculated properties.
    #
    # Raises
    # --------------------
    # ValueError
    #     If the length of `n_list` does not match the number of data sources in `df_dict`.
    # '''
#    df_dict = bd.df_dict
#    if len(nlist) != len(df_dict.keys()):
#         raise ValueError("The length of nlist must match the number of keys in df_dict.")
#    ss_dict = {}
#    for i, key in enumerate(df_dict.keys()):
#        data = df_dict[key].values.astype(float)
#        ss_dict[key]=data[np.random.choice(len(data), nlist[i], replace=False)]
#    return ss_dict
    
#'''
#'''
#def get_props_from_subsamples_means(bd,reps,n_list, plot=True):
    # '''
    # Gets proportions from subsamples by calculating means and optionally plots the results.
    #
    # Parameters
    # --------------------
    # bd : object
    #     BasinData class instance that contains its information.
    #
    # reps : int
    #     The number of repetitions for generating random subsamples.
    #
    # n_list : list
    #     A list of integers specifying the sizes of the subsamples to be taken from the data.
    #
    # plot : bool, optional
    #     A flag indicating whether to plot the results (default is True).
    #
    # Returns
    # --------------------
    # Ps : numpy.ndarray
    #     Proportions. An array of proportions calculated from the subsamples, where each row corresponds to a repetition.
    #
    # Raises
    # --------------------
    # ValueError
    #     If the length of `n_list` does not match the number of data sources in `bd`.
    # '''
#    df_dict = bd.df_dict
#    Y = bd.df_dict['Y'].values.astype(float)
#    S_inv = np.linalg.inv(np.cov(Y.T))
#    
#    if len(df_dict.keys()) != len(n_list):
#     raise ValueError("The n_list needs to be of the same size as the number of data sources in bd, keys in df_dict.")
#    else:
#        Ps = []
#        for i in range(reps):
#            ss_dict = random_subsamples(bd,n_list)
#            X = []
#            for key in bd.sources:
#                X.append(ss_dict[key].mean(axis=0))
#            X = np.array(X).T
#            y = ss_dict['Y'].mean(axis=0)
#            P = solvers.solve_gls_4x4(y,X,S_inv)
#            #print(P)
#            if np.all(np.array(P[0:3])>=0):
#                #print (P[0:3], 'soma:', np.sum(P[0:3]))    
#                Ps.append(list(P))
#        Ps = np.array(Ps)
#        if plot:
#            plt.plot(Ps[:,0],Ps[:,1],'.')
#            plt.xlabel("P1")
#            plt.ylabel("P2")
#            plt.xlim((0,1))
#            plt.ylim((0,1))
#            plt.plot(Ps[:,0],Ps[:,1],'.')
#            plt.show()
#        return(Ps)































