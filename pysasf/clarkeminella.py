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
from itertools import product
from IPython.display import clear_output
from scipy.spatial import ConvexHull
import time

#PySASF imports
import distances 
import solvers
import stats

'''
moved to basindata.py
################################################################################
def props_from_all_combinations(bd, solve_opt='ols',save=True):
    from itertools import product
    df_dict = bd.df_dict
    S_inv = bd.S_inv
    
    filename = ''
    idx = []
    keys = list(df_dict.keys())
    for key in keys:
        size = len(df_dict[key])
        filename=filename+key+str(size)
        idx.append(list(np.arange(size)))
    bd.filename = filename
    
    # Gera todas as combinações possíveis
    combs = list(product(*idx))
    total = len(combs)

    #cria um array vazio para todos os Ps
    Ps = np.empty((len(combs),len(keys)-1)).astype(float)

    for k, comb in enumerate(combs):
        X = []
        for i in range(len(comb)-1):
            key = keys[i]
            pos = comb[i]
            data = df_dict[key].values[pos]
            X.append(data.astype(float)) 
        y = df_dict['Y'].values[comb[-1]].astype(float)
        X = np.array(X).T
        #SOLVE
        if solve_opt == 'gls':
            P = solvers.solve_gls_4x4(y,X,S_inv)
        if solve_opt == 'ols':
            P = solvers.solve_ols_4x4(y,X)
        if solve_opt == 'opt':
            P = solvers.solve_minimize(y,X)
        Ps[k] = P
    return combs, Ps
################################################################################
'''

'''
moved to stats.py
def randon_props_subsamples(bd, key, n):
    Ps = bd.props
    combs = bd.combs
    key_idx = list(bd.df_dict.keys()).index(key)
    size = len(bd.df_dict[key])
    rand = np.random.choice(np.arange(size), n, replace=False)
    selected_combs = combs[np.where(np.isin(combs[:,key_idx],rand))]
    selected_Ps = Ps[np.where(np.isin(combs[:,key_idx],rand))]
    return selected_combs, selected_Ps

################################################################################
'''
def cm_feasebles(Ps):
    Ps_feas = Ps[[np.all(P>0) for P in Ps]]
   # Ps_feas = []
   # for P in Ps:
   #     if np.all(P>0):# and np.sum(P)<=1:
   #        Ps_feas.append(P)
            
    return np.array(Ps_feas)

#########################################################################################3
def run_repetitions_and_reduction (bd, key, 
                                        reductions,
                                        repetitions = 50):
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

    # get data from df for reductions
    data = bd.df_dict[key].values.astype(float)
    
    for n in reductions:
        areas = []
        filename = filename+'-'+str(n)
        for i in range(repetitions):
            clear_output(wait=True)
            print ('Processing for', n, 'subsamples of',key, ', repetition number', i)

            _,Ptot = stats.randon_props_subsamples(bd, key, n)
            Pfea =  cm_feasebles(Ptot) 
            if Pfea.shape[0]>=4:
                Pcr = stats.confidence_region(Pfea[:,0:2], p = 95)
                hull = ConvexHull(Pcr)
                areas.append(hull.volume)
        
        # insert data for df_out
        nSamp = n
        CV = np.round(cv(areas),4)
        Mean = np.round(np.mean(areas),4)
        Std = np.round(np.std(areas),4)
        Total = len(Ptot)
        Feas = len(Pfea)
        df_out_data_n = [nSamp,CV,Mean,Std,Total,Feas]
        for i in range(nSources):
            df_out_data_n.append(np.mean(Pfea, axis=0)[i])
        df_out_data.append(df_out_data_n)
        bd.cm_df = df_out_data

        CVs.append(cv(areas))
        areas_medias.append(np.mean(areas))
    
    print('Done!')
    clear_output(wait=True)
    fim = time.time()
    print ("Time for all runs:",fim-inicio)
    
    df_out = pd.DataFrame(df_out_data, columns=df_out_cols)
    display(df_out)
    bd.cm_df = df_out

    # Saving file in cvs
    print ('Saving in', filename+'.csv')
    df_out.to_csv(bd.output_folder+'/'+filename+'.csv')
    return (df_out)
###########################################################################################3


def confidence_region(P):
    return stats.confidence_region(P, p = 95, space_dist='mahalanobis0')
#    Pm = np.mean(P, axis=0)
#    dist = distances.mahalanobis0_dist(P, Pm)
#    #dist = distances.mahalanobis_dist(P, Pm)
#    
#    sorted_idx = np.argsort(dist)
#    Psorted = P[sorted_idx]#
#
#    # em ordem crescente
#    end_idx = int((p/100)*len(Psorted))
#    #print ("Os 95% mais próximos:", Psorted[:,:end_idx])
#    return (Psorted[:end_idx,:])



'''
Esta função processa o cálculo das porcentagens usando as médias
Faz as contas usando reps repetições tomando aletoriamente um conjunto
de subamostras com tamanhos definidos por n_list. Tem opção de plotar. 
'''
def get_props_from_subsamples_means(bd,reps,n_list, plot=True):
    df_dict = bd.df_dict
    Y = bd.df_dict['Y'].values.astype(float)
    S_inv = np.linalg.inv(np.cov(Y.T))
    
    if len(df_dict.keys())!=len(n_list):
        print("The n_list needs to be of the same size of lands data.")
    else:
        Ps = []
        for i in range(reps):
            ss_dict = random_subsamples(bd,n_list)
            X = []
            for key in bd.sources:
                X.append(ss_dict[key].mean(axis=0))
            X = np.array(X).T
            y = ss_dict['Y'].mean(axis=0)
            P = solvers.solve_gls_4x4(y,X,S_inv)
            #print(P)
            if np.all(np.array(P[0:3])>=0):
                #print (P[0:3], 'soma:', np.sum(P[0:3]))    
                Ps.append(list(P))
        Ps = np.array(Ps)
        if plot:
            plt.plot(Ps[:,0],Ps[:,1],'.')
            plt.xlabel("P1")
            plt.ylabel("P2")
            plt.xlim((0,1))
            plt.ylim((0,1))
            plt.plot(Ps[:,0],Ps[:,1],'.')
            plt.show()
        return(Ps)































