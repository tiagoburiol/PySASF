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
    
def cm_feasebles(Ps):
    Ps_feas = Ps[[np.all(P>0) for P in Ps]]
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


    # apenas para calcular o numero total de Ps (Melhorar isso)
    prod =1
    for k in bd.sources:
        if k!=key:
            prod = prod*len(bd.df_dict[k])


    def compute_area(pts):
        hull = ConvexHull(pts)
        area = hull.volume # area for 2d points
        return area
    
    
    #AQUI ACONTECEM AS REDUÇÕES
    for n in reductions:
        #Total=prod*n
        points_set = []
        areas = []
        filename = filename+'-'+str(n)

        t1 = time.time()
        for i in range(repetitions):
            t2 = time.time()
            if t2-t1>0.5:
                #clear_output(wait=True)
                print ('Processing for', n, 'subsamples of',key, 
                   ', repetition number', i+1,end='\r', flush=True)
                t1=t2

            
            #_,Ptot = stats.randon_props_subsamples(bd, key, n, only_feasebles=False)
            #Pfea =  cm.cm_feasebles(Ptot[:,0:2])
            _,Pfea = stats.randon_props_subsamples(bd, key, n, only_feasebles=True)

            if Pfea.shape[0]>=4:
                Pcr = stats.confidence_region(Pfea[:,0:2], p = 95)
                #<<-------------------------------
                #hull = ConvexHull(Pcr[:,0:2])    #Aqui
                #areas.append(hull.volume)
                #>>-------------------------------
                points_set.append(Pcr)
                #-------------------------------
        #Aqui
        #>>-------------------------------
        with concurrent.futures.ThreadPoolExecutor() as executor:
             areas = list(executor.map(compute_area, points_set)) 
        #-------------------------------

        
        # insert data for df_out
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



'''
Esta função processa o cálculo das porcentagens usando as médias
Faz as contas usando reps repetições tomando aletoriamente um conjunto
de subamostras com tamanhos definidos por n_list. Tem opção de plotar. 
'''

def random_subsamples(bd,nlist):
    df_dict = bd.df_dict
    ss_dict = {}
    for i, key in enumerate(df_dict.keys()):
        data = df_dict[key].values.astype(float)
        ss_dict[key]=data[np.random.choice(len(data), nlist[i], replace=False)]
    return ss_dict

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































